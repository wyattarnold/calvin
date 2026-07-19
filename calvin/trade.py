"""Institutional water-trading limit as a per-water-year budget.

In CALVIN, trading is not a variable — the network-flow LP reallocates water to
its highest-value use by default, so the unconstrained solve *is* the
frictionless market. This module limits that reallocation with a tunable budget:

  per water year wy:
    Σ_{ag delivery arcs in wy} X  +  budget_wy  +  overflow_wy  ≥  Ent_wy

where ``Ent_wy`` is the full ag entitlement (Σ_k upper_bound over the curated ag
delivery arcs, per the CALVIN demand convention) — the rights baseline. Total ag
shortfall below entitlement (= water reallocated away from ag) is split into:

  - ``budget_wy ∈ [0, T]`` — voluntary market reallocation, free up to the annual
    trade limit T (TAF/yr). The row dual is the marginal value of relaxing the
    limit: the *price of the release valve*, what building competes against.
  - ``overflow_wy ∈ [0, Ent_wy]`` — involuntary curtailment beyond the limit,
    priced at ``overflow_penalty`` so the LP fills the free budget first and only
    overflows when a drought forces shortfall past T (keeps dry years feasible;
    overflow_wy is the drought-shortfall report).

T=0 is rigid rights (no trade); T→∞ is the free market. Sweeping T traces the
value of new supply against the trading limit. The penalty is set above any real
supply cost so it backstops feasibility without capping the build decision.

Aggregate (system-wide) budget per year, not per node, so the LP spends the
budget on the highest-value trades first (lowest-WTP ag sells first) — the market
outcome. Per-node trade is recovered post-hoc from (ub − flow).

Backend: HiGHS only (the two-stage capacity path). Structure mirrors
``calvin/env_flow.py``: df-pure RHS constants, backend-neutral column/row specs.

Design: my-models/two-stage-cap/notes/01-design/cost-of-inaction-study-design.md
"""
from collections import defaultdict

from calvin.env_flow import _base, _water_year


def ag_entitlement(df, ag_links):
  """Full ag entitlement per water year and the ag delivery arcs each sums.

  :param df: links DataFrame (columns i, j, k, upper_bound, ...)
  :param ag_links: iterable of base ``'i-j'`` names (``scenario.demand_links``)
  :returns: ``({wy: entitlement TAF}, {wy: [(i, j, k) arc keys]})`` — entitlement
    is Σ_k upper_bound over the ag delivery arcs whose base name is in
    ``ag_links``, grouped by the water year of the arc's timestamp.
  """
  ag = set(ag_links)
  bi = df['i'].map(_base)
  bj = df['j'].map(_base)
  ij = bi + '-' + bj
  mask = ij.isin(ag)
  sub = df[mask]

  ent = defaultdict(float)
  arcs = defaultdict(list)
  for row in sub.itertuples():
    wy = _water_year(row.i)
    if wy is None:
      continue
    ent[wy] += row.upper_bound
    arcs[wy].append((row.i, row.j, row.k))
  return dict(ent), dict(arcs)


def trade_budget_rows(df, config, log=None):
  """Backend-neutral trade-budget column + row specs.

  :param config: dict —
    ``T_tafy`` (float, annual trade limit; default 0.0 = rigid rights),
    ``overflow_penalty`` (float, per-TAF cost of curtailment beyond the limit;
    default 10x the max absolute arc cost so it never caps the build),
    ``ag_links`` (optional iterable of base 'i-j'; default
    ``scenario.demand_links('ag')``).
  :returns: ``(col_specs, row_specs, meta)`` where ``col_specs`` is a list of
    ``(key, lower, upper, cost)`` for the budget + overflow columns, ``row_specs``
    a list of ``(coeffs, sense, rhs, label)`` per water year, and ``meta`` the
    ``{wy: entitlement}`` dict for reporting.
  """
  if 'ag_links' in config and config['ag_links'] is not None:
    ag_links = config['ag_links']
  else:
    from calvin.scenario import demand_links
    ag_links = demand_links('ag')

  T = float(config.get('T_tafy', 0.0))
  penalty = config.get('overflow_penalty')
  if penalty is None:
    penalty = 10.0 * float(df['cost'].abs().max())
  penalty = float(penalty)

  ent, arcs = ag_entitlement(df, ag_links)
  wys = sorted(ent)

  col_specs = []
  row_specs = []
  for wy in wys:
    budget_key = ('trade_budget', wy)
    overflow_key = ('trade_overflow', wy)
    col_specs.append((budget_key, 0.0, T, 0.0))
    col_specs.append((overflow_key, 0.0, ent[wy], penalty))
    coeffs = {a: 1.0 for a in arcs[wy]}
    coeffs[budget_key] = 1.0
    coeffs[overflow_key] = 1.0
    row_specs.append((coeffs, '>=', ent[wy], ('trade_budget', wy)))

  if log:
    total_ent = sum(ent.values())
    log.info('trade budget: %d water years, ag entitlement %.0f TAF/yr, '
             'T=%.0f TAF/yr, overflow penalty %.3g/TAF'
             % (len(wys), total_ent / max(len(wys), 1), T, penalty))
  return col_specs, row_specs, ent


def add_trade_budget(model, df, config, log=None):
  """Add the per-water-year trade-budget columns + rows to a built model.

  HiGHS only (dispatch on ``add_rows``); columns first, then rows. Returns the
  ``{wy: entitlement}`` meta dict.
  """
  col_specs, row_specs, ent = trade_budget_rows(df, config, log=log)
  if not hasattr(model, 'add_rows'):
    raise NotImplementedError('trade budget is implemented for the HiGHS backend '
                              'only')
  model.add_columns(col_specs)     # budget + overflow columns first
  model.add_rows(row_specs)        # rows reference them by key
  if log:
    log.info('trade budget: added %d columns + %d rows'
             % (len(col_specs), len(row_specs)))
  return ent
