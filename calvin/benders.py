"""Benders decomposition for the two-stage CVaR capacity problem.

The monolithic extensive form (`extensive_form.build_ef`) couples S operational
blocks into one LP; at full 82-yr horizon that LP is intractable (S x ~3M cols,
superlinear simplex). Benders splits it back apart:

  * a small **master** over the here-and-now build x (the `('cap',f)` levels), the
    CVaR variables (eta, z_s), and one recourse-epigraph variable theta_s per
    sample, plus accumulating optimality cuts;
  * S independent **subproblems** Q_s(x), each a single operational block with the
    build fixed to the master's proposal. Subproblems are independent (parallel
    across the machine) and warm-start across iterations (only the fixed cap bounds
    move), so after the first pass each re-solve is cheap.

Because every block is feasible for any x >= 0 (priced-soft slacks, and more
capacity only relaxes), we need **optimality cuts only** -- no feasibility cuts.
The cut for sample s at build x_hat is the supporting hyperplane of Q_s:

    theta_s >= Q_s(x_hat) + sum_f pi_{s,f} (x_f - x_hat_f),

where pi_{s,f} = dQ_s/dx_f is the reduced cost of the (fixed) `('cap',f)` column
once its capital coefficient is zeroed (capital lives only in the master). Cuts
approximate Q_s(x), which does not depend on lambda, so a single cut pool is shared
across the whole lambda sweep: solve lambda=0 first, then reuse and extend the pool
for the risk-averse points.

CVaR objective (Rockafellar-Uryasev), matching build_ef:

    min  sum_f capcoeff_f x_f  +  (1-lam)/S sum_s theta_s
         +  lam ( eta + 1/((1-beta) S) sum_s z_s )
    s.t. z_s >= theta_s - eta,  z_s >= 0,  theta_s >= (cuts),  x in [0, xcap_max],
         group caps.

`solve_benders` returns, per lambda, the build, RP = capital + E[Q], the CVaR, and
the convergence trace. Validate against `extensive_form.build_ef`/`solve_ef` on the
smoke network (RP must match to tolerance); production runs go to full horizon at S
the coupled form can't reach.
"""
import os
from time import perf_counter

import numpy as np
import highspy

from calvin import CALVINCap
from calvin.extensive_form import build_ef
from calvin.futures import apply_futures
from calvin.scenario import apply_gw_export_closures

INF = highspy.kHighsInf


def cvar_of(Q, beta):
  """Rockafellar-Uryasev CVaR_beta of a finite sample {Q_s} and its argmin eta.

  CVaR = min_eta eta + 1/((1-beta) S) sum_s max(Q_s - eta, 0); the minimizer is the
  empirical VaR (the ceil(beta S)-th smallest). Returns (cvar, eta)."""
  Q = np.sort(np.asarray(Q, dtype=float))
  S = len(Q)
  # eta candidates are the order statistics; evaluate the R-U objective at each.
  best, best_eta = np.inf, Q[-1]
  for eta in Q:
    val = eta + np.maximum(Q - eta, 0.0).sum() / ((1.0 - beta) * S)
    if val < best:
      best, best_eta = val, eta
  return float(best), float(best_eta)


# ---------------------------------------------------------------------------
# Precompute: which floors actually bind (so blocks price only those as soft)
# ---------------------------------------------------------------------------
def _binding_for_sample(task):
  """Binding floor arcs for one sample at zero build. Top-level for pickling."""
  links, sample, catalog, enforce_alpha, options = task
  from calvin.relax import relaxable_floors, binding_floors
  cal = CALVINCap(links, scenario=None, enforce_alpha=enforce_alpha,
                  expansions_csv=None, **catalog)
  apply_futures(cal.df, sample, log=None)
  apply_gw_export_closures(cal.df, log=None)
  floors = relaxable_floors(cal.df)
  cal.create_highs_model(debug_mode=False)
  m = cal.hmodel
  cap_idx = np.array([m.extra_cols[k] for k in m.extra_cols
                      if isinstance(k, tuple) and k and k[0] in ('cap', 'exp')],
                     dtype=np.int32)
  if len(cap_idx):                                  # zero build = maximal shortage
    z = np.zeros(len(cap_idx))
    m.set_col_bounds(cap_idx, z, z)
  return {tuple(a) for a in binding_floors(m, floors, options=options)}


def precompute_soft_floors(links, samples, *, catalog, enforce_alpha=True,
                           options=None, workers=8, log=print):
  """Union of floors that bind across the ensemble (at zero build), for
  `build_ef(soft_floor_arcs=...)`. One phase-1 solve per sample, in parallel."""
  import multiprocessing as mp
  opts = options or {'solver': 'simplex'}
  tasks = [(links, s, catalog, enforce_alpha, opts) for s in samples]
  t0 = perf_counter()
  ctx = mp.get_context('spawn')
  with ctx.Pool(processes=min(workers, len(samples))) as pool:
    sets = pool.map(_binding_for_sample, tasks)
  union = set().union(*sets) if sets else set()
  if log:
    log('[benders] precomputed %d binding floors (union of %d..%d per sample) '
        'in %.0fs' % (len(union), min(len(x) for x in sets),
                      max(len(x) for x in sets), perf_counter() - t0))
  return union


# ---------------------------------------------------------------------------
# Subproblem: Q_s(x) and its subgradient
# ---------------------------------------------------------------------------
class Subproblem:
  """One sample's operational block. Built once; `solve(x_hat)` fixes the build,
  re-solves warm, and returns (Q_s, {f: dQ_s/dx_f})."""

  def __init__(self, links, sample, T, beta, *, env_flow, catalog,
               wtp_tranches=8, enforce_alpha=True, soft_floor_arcs=None,
               options=None, log=None):
    m, meta = build_ef(links, [sample], T, 0.0, beta, env_flow=env_flow,
                       catalog=catalog, enforce_alpha=enforce_alpha,
                       wtp_tranches=wtp_tranches, share_build=True,
                       soft_alpha=enforce_alpha, soft_floor_arcs=soft_floor_arcs,
                       log=log)
    self.m = m
    self.fac_names = meta['fac_names']
    self.cap_col = {f: m.extra_cols[('cap', f)] for f in self.fac_names}
    self._idx = np.array([self.cap_col[f] for f in self.fac_names], dtype=np.int32)
    self._q_col = ('Q', 0)
    self.options = options or {'solver': 'simplex'}
    # capital lives in the master; the subproblem objective is pure recourse Q_s.
    m.set_col_costs(self._idx, np.zeros(len(self._idx)))

  def solve(self, x_hat):
    m = self.m
    lo = np.array([x_hat[f] for f in self.fac_names], dtype=float)
    m.set_col_bounds(self._idx, lo, lo)          # fix build = x_hat (warm re-solve)
    m.solve(need_duals=True, options=self.options, raise_on_infeasible=True)
    Q = float(m.cap_values()[self._q_col])
    cd = m._col_dual
    pi = {f: float(cd[self.cap_col[f]]) for f in self.fac_names}
    return Q, pi


# ---------------------------------------------------------------------------
# Parallel subproblem backend (persistent warm workers)
# ---------------------------------------------------------------------------
def _sp_worker(links, indices, samples, cfg, in_q, out_q):
  """Long-lived worker: builds its blocks once, then serves solve(x_hat) requests
  warm until sent None. Returns {sample_idx: (Q, pi)} per request."""
  subs = {i: Subproblem(links, samples[i], cfg['T'], cfg['beta'],
                        env_flow=cfg['env_flow'], catalog=cfg['catalog'],
                        wtp_tranches=cfg['wtp_tranches'],
                        enforce_alpha=cfg['enforce_alpha'],
                        soft_floor_arcs=cfg['soft_floor_arcs'],
                        options=cfg['options'], log=None)
          for i in indices}
  out_q.put(('ready', list(indices)))
  while True:
    x_hat = in_q.get()
    if x_hat is None:
      break
    out_q.put(('result', {i: subs[i].solve(x_hat) for i in indices}))


class ParallelSubproblems:
  """Persistent parallel subproblem backend. Each of `workers` processes owns a
  static, dryness-balanced slice of the S blocks (built once, warm-started across
  iterations). Use as `solve_benders(..., solve_sub=backend.solve_all)`; call
  `close()` when done. Provides the same `list of (Q, pi)` contract as the serial
  path."""

  def __init__(self, links, samples, *, T, beta, env_flow, catalog,
               wtp_tranches=8, enforce_alpha=True, soft_floor_arcs=None,
               options=None, workers=8, log=print):
    import multiprocessing as mp
    self.S = len(samples)
    W = min(workers, self.S)
    self.W = W
    cfg = dict(T=T, beta=beta, env_flow=env_flow, catalog=catalog,
               wtp_tranches=wtp_tranches, enforce_alpha=enforce_alpha,
               soft_floor_arcs=soft_floor_arcs,
               options=options or {'solver': 'simplex'})
    # spread the slow (dry) blocks across workers: sort by dryness, round-robin.
    order = sorted(range(self.S),
                   key=lambda i: samples[i]['warm_shift']['wa_target'])
    chunks = [order[w::W] for w in range(W)]

    ctx = mp.get_context('spawn')
    self.in_qs, self.out_qs, self.procs = [], [], []
    for w in range(W):
      inq, outq = ctx.Queue(), ctx.Queue()
      p = ctx.Process(target=_sp_worker,
                      args=(links, chunks[w], samples, cfg, inq, outq),
                      daemon=True)
      p.start()
      self.in_qs.append(inq)
      self.out_qs.append(outq)
      self.procs.append(p)
    t0 = perf_counter()
    for w in range(W):                              # barrier: all blocks built
      tag, _ = self.out_qs[w].get()
      assert tag == 'ready'
    if log:
      log('[benders] %d workers built %d subproblems in %.0fs'
          % (W, self.S, perf_counter() - t0))

  def solve_all(self, x_hat):
    for inq in self.in_qs:
      inq.put(x_hat)
    merged = {}
    for outq in self.out_qs:
      tag, res = outq.get()
      merged.update(res)
    return [merged[i] for i in range(self.S)]

  def close(self):
    for inq in self.in_qs:
      inq.put(None)
    for p in self.procs:
      p.join(timeout=15)


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------
class BendersMaster:
  """Master LP over (x, theta, eta, z) with a growing cut pool. The lambda-weighted
  objective is re-costed in place between frontier points; the cuts (lambda-free)
  persist."""

  def __init__(self, fac_names, cap_coeff, xcap_max, groups, S, beta,
               theta_lb=-1e11):
    self.fac = list(fac_names)
    self.nf = len(self.fac)
    self.S = S
    self.beta = beta
    self.cap_coeff = cap_coeff
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    self.h = h

    # column layout: x_f | theta_s | eta | z_s
    self.x0 = 0
    self.t0 = self.nf
    self.eta = self.nf + S
    self.z0 = self.nf + S + 1
    self.ncol = self.nf + 2 * S + 1

    for f in self.fac:                                    # x_f
      h.addCol(cap_coeff[f], 0.0, float(xcap_max[f]), 0, [], [])
    for _ in range(S):                                    # theta_s (recourse epi)
      h.addCol(0.0, theta_lb, INF, 0, [], [])
    h.addCol(0.0, -INF, INF, 0, [], [])                   # eta
    for _ in range(S):                                    # z_s
      h.addCol(0.0, 0.0, INF, 0, [], [])

    # CVaR rows: z_s + eta - theta_s >= 0
    for s in range(S):
      h.addRow(0.0, INF, 3, [self.z0 + s, self.eta, self.t0 + s], [1.0, 1.0, -1.0])
    # group caps: sum_f sumprofile_f x_f <= ceil_g
    for g, (ceil, coeffs) in groups.items():
      idx = [self.x0 + self.fac.index(f) for f in coeffs]
      val = [coeffs[f] for f in coeffs]
      h.addRow(-INF, float(ceil), len(idx), idx, val)

  def set_lambda(self, lam):
    """Re-cost theta/eta/z for a new lambda (cuts unchanged)."""
    S = self.S
    idx = ([self.t0 + s for s in range(S)] + [self.eta]
           + [self.z0 + s for s in range(S)])
    cost = ([(1.0 - lam) / S] * S + [lam]
            + [lam / ((1.0 - self.beta) * S)] * S)
    self.h.changeColsCost(len(idx), np.array(idx, dtype=np.int32),
                          np.array(cost, dtype=float))

  def add_cut(self, s, Q_hat, pi, x_hat):
    """theta_s - sum_f pi_f x_f >= Q_hat - sum_f pi_f x_hat_f."""
    rhs = Q_hat - sum(pi[f] * x_hat[f] for f in self.fac)
    idx = [self.t0 + s] + [self.x0 + i for i in range(self.nf)]
    val = [1.0] + [-pi[f] for f in self.fac]
    self.h.addRow(float(rhs), INF, len(idx), idx, val)

  def solve(self):
    self.h.run()
    sol = self.h.getSolution()
    cv = np.asarray(sol.col_value, dtype=float)
    x = {f: float(cv[self.x0 + i]) for i, f in enumerate(self.fac)}
    theta = cv[self.t0:self.t0 + self.S].astype(float)
    obj = float(self.h.getObjectiveValue())
    return x, theta, obj


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _registries(links, catalog, enforce_alpha):
  """A CALVINCap just for the build registries the master needs (sample-free)."""
  cal = CALVINCap(links, scenario=None, enforce_alpha=enforce_alpha,
                  expansions_csv=None, **catalog)
  fac = list(cal.facilities.index)
  groups = {}
  cg = getattr(cal, 'cap_groups', {}) or {}
  for g, ceil in cg.items():
    coeffs = {f: cal._sum_profile[f] for f in fac
              if cal.facilities.cap_group[f] == g}
    if coeffs:
      groups[g] = (ceil, coeffs)
  return fac, dict(cal._cap_coeff), dict(cal._xcap_max), groups


def solve_benders(links, samples, T, lambdas, beta, *, env_flow, catalog,
                  wtp_tranches=8, enforce_alpha=True, soft_floor_arcs=None,
                  max_iter=40, tol=1e-4, solve_sub=None, options=None, log=print):
  """Solve the two-stage CVaR problem by Benders for each lambda in `lambdas`,
  sharing one (lambda-free) cut pool across the sweep.

  :param solve_sub: optional callable(x_hat) -> list of (Q_s, pi_s) over samples,
    for a parallel subproblem backend. Default solves the S `Subproblem`s serially
    (warm). :returns: list of per-lambda result dicts.
  """
  S = len(samples)
  fac, cap_coeff, xcap_max, groups = _registries(links, catalog, enforce_alpha)

  # subproblem backend: injected (e.g. parallel persistent workers) or serial.
  if solve_sub is not None:
    _solve_all = solve_sub
  else:
    t0 = perf_counter()
    subs = [Subproblem(links, samples[s], T, beta, env_flow=env_flow,
                       catalog=catalog, wtp_tranches=wtp_tranches,
                       enforce_alpha=enforce_alpha,
                       soft_floor_arcs=soft_floor_arcs, options=options)
            for s in range(S)]
    if log:
      log('[benders] built %d subproblems in %.0fs' % (S, perf_counter() - t0))
    _solve_all = lambda x_hat: [sub.solve(x_hat) for sub in subs]

  master = BendersMaster(fac, cap_coeff, xcap_max, groups, S, beta)
  results = []
  x_hat = {f: 0.0 for f in fac}                # first proposal: no build

  for lam in lambdas:
    master.set_lambda(lam)
    lb, it = -np.inf, 0
    best_ub, best = np.inf, None                # incumbent (best build seen)
    trace = []
    for it in range(1, max_iter + 1):
      qp = _solve_all(x_hat)                    # S subproblem solves at x_hat
      Q = np.array([q for q, _ in qp])
      for s, (Qs, pis) in enumerate(qp):
        master.add_cut(s, Qs, pis, x_hat)
      capital = sum(cap_coeff[f] * x_hat[f] for f in fac)
      cvar, eta = cvar_of(Q, beta)
      ub = capital + (1.0 - lam) / S * Q.sum() + lam * cvar   # cost of THIS build
      if ub < best_ub:                          # keep the best build, not the last
        best_ub = ub
        best = dict(build=dict(x_hat), capital=capital, Q=Q, cvar=cvar, eta=eta)
      x_hat, theta, lb = master.solve()         # new proposal + valid lower bound
      gap = abs(best_ub - lb) / max(1.0, abs(best_ub))
      trace.append((it, lb, best_ub, gap))
      if log:
        log('[lam=%.2f it=%2d] LB=%.4g UB=%.4g gap=%.2e'
            % (lam, it, lb, best_ub, gap))
      if gap <= tol:
        break

    Q = best['Q']
    built = {f: v for f, v in best['build'].items() if v > 1e-4}
    results.append(dict(lam=lam, iters=it, gap=gap, capital=best['capital'],
                        expected_Q=float(Q.mean()),
                        RP=best['capital'] + float(Q.mean()), cvar=best['cvar'],
                        eta=best['eta'], Q=Q, build=best['build'],
                        n_fac=len(built), trace=trace))
  return results
