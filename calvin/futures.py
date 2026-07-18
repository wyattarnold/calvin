"""
Scenario-tree driver levers for the cost-of-inaction futures.

Two supply-side drivers applied on top of the institutional baseline
(``scenario.apply_scenario``) and the environmental-flow lever
(``env_flow``), matching the study design's three varying axes
(``notes/01-design/cost-of-inaction-study-design.md`` §4):

1. Colorado cut  — reduce California's Colorado River import (the single
   ``SR_CR3`` inflow, 4,400 TAF/yr = the 4.4-maf apportionment) by a fixed
   volume. Report driver: likely 0.5 / worse 1.0 maf/yr.
2. Climate warm-shift — perturb the surface rim inflows by the California
   Fourth Assessment GCM monthly multipliers (Herman et al. 2018, shipped in
   ``data/fourth-assessment-data``), rescaled to hit a target water-availability
   change (WA) and, optionally, a winter index (WI). Report driver: WA likely
   -1.5 (~ -4.2%) / worse -3 (~ -8.5%) maf/yr.

Both mutate a time-expanded links DataFrame in place, editing the forced
``INFLOW -> node`` arc bounds (lower_bound == upper_bound for a natural
inflow), and return an audit DataFrame in the ``scenario.bound_adjustments``
schema so a run can record exactly what changed.

The functions are df-pure (no model/solver dependency), so they compose with
the scenario/env layers and later with the COSVF single-year networks.
"""
import os

import numpy as np
import pandas as pd

from calvin.scenario import _split_base, bound_adjustments, BOUND_DECIMALS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GCM_DIR = os.path.join(BASE_DIR, 'data', 'fourth-assessment-data',
                       'rim-inflow-multipliers')

# California's Colorado River apportionment enters at SR_CR3 (4,400 TAF/yr, an
# October lump the reservoir then releases). SR_CR1/CR2 are the priority-4a
# storage/routing nodes; SR_CRW (Owens) and the salt sinks are separate.
COLORADO_NODE = 'SR_CR3'

# Surface rim inflows excluded from the climate multiplier: groundwater basins
# (not rim runoff), the artificial Desal/CN2 backstop sources, the Colorado
# import, and the salt-import placeholders. Everything else fed by an INFLOW
# arc is a natural surface rim inflow (~37 MAF/yr across 92 nodes, ~ the design's
# 35.3-MAF anchor).
RIM_EXCLUDE_PREFIXES = ('GW_', 'SR_CR')
RIM_EXCLUDE_NODES = frozenset({'Desal', 'CN2', 'C146', 'C148'})


# ---------------------------------------------------------------------------
# Orchestrator (the [futures] axis)
# ---------------------------------------------------------------------------
def apply_futures(df, futures, log=None):
  """
  Apply the supply-side future drivers to a links DataFrame, in order:
  Colorado cut, then climate warm-shift. Mirrors ``scenario.apply_scenario``.

  Composition note: these change the exogenous supply, so they must be applied
  BEFORE the market reference solve of a given future — the institutional
  delivery floors (``scenario.apply_scenario``) anchor to that future's own
  market run, not the historical one. The environmental-flow lever
  (``env_flow``) is applied at model-build time and is orthogonal here.

  :param df: links DataFrame (mutated in place)
  :param futures: dict with optional keys:
    ``colorado_cut_taf`` (float, annual TAF cut on the Colorado import),
    ``warm_shift`` (dict: member, rcp, wa_target, winter_index).
  :returns: (audit DataFrame of every bound changed, info dict)
  """
  known = {'enabled', 'colorado_cut_taf', 'warm_shift'}
  unknown = set(futures) - known
  if unknown:
    raise ValueError('Unknown [futures] keys %s; known: %s'
                     % (sorted(unknown), sorted(known)))

  init_bounds = df[['lower_bound', 'upper_bound']].copy()
  info = {}

  cut = futures.get('colorado_cut_taf', 0.0)
  if cut:
    apply_colorado_cut(df, cut, log=log)
    info['colorado_cut_taf'] = cut

  ws = futures.get('warm_shift') or None
  if ws:
    _, ws_info = apply_warm_shift(
        df, member=ws.get('member'), rcp=ws.get('rcp'),
        wa_target=ws.get('wa_target'), winter_index=ws.get('winter_index'),
        log=log)
    info['warm_shift'] = ws_info

  return bound_adjustments(df, init_bounds), info


# ---------------------------------------------------------------------------
# Colorado import cut
# ---------------------------------------------------------------------------
def apply_colorado_cut(df, cut_taf, node=COLORADO_NODE, log=None):
  """
  Reduce the annual Colorado River import by ``cut_taf`` TAF/yr.

  Scales every ``INFLOW -> node`` arc within each water year by
  ``(annual - cut_taf) / annual`` so the annual delivered volume drops by
  ``cut_taf`` regardless of how the inflow is distributed across months
  (SR_CR3 delivers the whole 4,400 TAF in October, but the proportional edit
  is robust to any pattern). Both bounds move together (forced inflow).

  :param df: links DataFrame (mutated in place)
  :param cut_taf: annual reduction in TAF/yr (e.g. 500 likely, 1000 worse);
    must be < the annual import
  :param node: Colorado import node base name (default ``SR_CR3``)
  :returns: audit DataFrame of every bound changed
  """
  init_bounds = df[['lower_bound', 'upper_bound']].copy()

  bi, bj = _split_base(df.i), _split_base(df.j)
  mask = bi.eq('INFLOW') & bj.eq(node)
  if not mask.any():
    raise ValueError('No INFLOW -> %s import arcs found in this network' % node)

  wy = _water_year(df.i[mask])
  annual = df.loc[mask].groupby(wy)['upper_bound'].transform('sum')
  if (cut_taf >= annual).any():
    worst = float(annual.min())
    raise ValueError('Colorado cut %.0f TAF/yr exceeds the annual import '
                     '(min year = %.0f TAF/yr)' % (cut_taf, worst))

  factor = (annual - cut_taf) / annual
  for col in ('lower_bound', 'upper_bound'):
    df.loc[mask, col] = _round(df.loc[mask, col] * factor)

  if log:
    realized = (init_bounds.loc[mask, 'upper_bound'].sum()
                - df.loc[mask, 'upper_bound'].sum()) / _n_years(df.i[mask])
    log.info('Colorado cut: -%.0f TAF/yr on %s (%d arc-months, realized '
             '-%.0f TAF/yr)'
             % (cut_taf, node, int(mask.sum()), realized))
  return bound_adjustments(df, init_bounds)


# ---------------------------------------------------------------------------
# Climate warm-shift
# ---------------------------------------------------------------------------
# CALVIN rim-inflow node -> Fourth-Assessment projection basin (the 11 columns
# of the multiplier CSVs). Majors mapped by river/reservoir correspondence;
# rim inflows not listed fall back to the ensemble-mean shape across all 11
# basins (``_ENSEMBLE``), which still carries the statewide winter-up /
# spring-down signal and is then rescaled to the WA target.
_ENSEMBLE = '_ensemble'
RIM_BASIN_MAP = {
    # Sacramento River / Shasta / upper Sac tributaries
    'SR_SHA': 'SAC_B', 'SR_CLE': 'SAC_B', 'SR_WHI': 'SAC_B', 'SR_BLB': 'SAC_B',
    'C2': 'SAC_B', 'C5': 'SAC_B', 'C86': 'SAC_B', 'C87': 'SAC_B',
    'D74': 'SAC_B', 'D75': 'SAC_B', 'D76B': 'SAC_B', 'D77': 'SAC_B',
    'D94': 'SAC_B',
    # Feather / Oroville
    'SR_ORO': 'OROVI', 'C77': 'OROVI', 'C23': 'OROVI', 'D43A': 'OROVI',
    # Yuba (Smartville / New Bullards Bar) + Bear
    'SR_BUL': 'SMART', 'C27': 'SMART', 'C28': 'SMART', 'C29': 'SMART',
    'SR_RLL_CMB': 'BEARC',
    # American / Folsom
    'SR_FOL': 'FOL_I', 'D17': 'FOL_I',
    # Cosumnes/Dry (unregulated, group with American shape)
    'C37': 'FOL_I', 'C38': 'FOL_I',
    # Mokelumne / Pardee
    'SR_PAR': 'PRD-C', 'SR_BER': 'PRD-C', 'SR_CLK_INV': 'PRD-C',
    # Calaveras / New Hogan
    'SR_NHG': 'N_HOG',
    # Stanislaus / New Melones
    'SR_NML': 'N_MEL',
    # Tuolumne / Don Pedro
    'SR_DNP': 'DPR_I', 'SR_HTH': 'DPR_I', 'SR_LL_ENR': 'DPR_I',
    'SR_SCAGG': 'DPR_I',
    # Merced / Lake McClure + southern-SJ correlated tributaries
    'SR_MCR': 'LK_MC', 'SR_TRM': 'LK_MC',
    # San Joaquin / Millerton + Fresno/Chowchilla/Kings/Kern (SJ-valley shape)
    'SR_MIL': 'MILLE', 'SR_HID': 'MILLE', 'SR_BUC': 'MILLE',
    'SR_PNF': 'MILLE', 'SR_ISB': 'MILLE', 'SR_SCC': 'MILLE',
    # Eastern Sierra (Owens/Mono) — group with San Joaquin shape (xlsx corr)
    'SR_CRW': 'MILLE', 'SR_GNT': 'MILLE', 'C116': 'MILLE',
}

WINTER_MONTHS = (11, 12, 1, 2, 3, 4)   # Nov-Apr, the winter-index window


def load_gcm_multipliers(member=None, rcp=None):
  """
  Load a 12-month x 11-basin rim-inflow multiplier table.

  :param member: GCM file stem (e.g. ``CNRM-CM5``); None -> ensemble mean over
    the selected members. The design's picks: an RCP4.5 central member for the
    likely future, an RCP8.5 hot-dry tail member for worse.
  :param rcp: ``'rcp45'`` / ``'rcp85'`` to restrict the ensemble; None -> all.
  :returns: DataFrame indexed by month 1..12, columns = the 11 basin codes.
  """
  if member is not None:
    suffix = '.%s' % rcp if rcp else ''
    path = os.path.join(GCM_DIR, '%s%s.csv' % (member, suffix))
    if not os.path.isfile(path):
      raise FileNotFoundError('GCM multiplier file not found: %s' % path)
    tbl = pd.read_csv(path, index_col=0)
    tbl.index = tbl.index.astype(int)
    return tbl.sort_index()

  # ensemble mean over the per-GCM files (optionally one RCP)
  frames = []
  for fn in sorted(os.listdir(GCM_DIR)):
    if not fn.endswith('.csv') or fn == 'overall.csv':
      continue
    if rcp and rcp not in fn:
      continue
    t = pd.read_csv(os.path.join(GCM_DIR, fn), index_col=0)
    t.index = t.index.astype(int)
    frames.append(t.sort_index())
  if not frames:
    raise ValueError('No GCM multiplier files matched (rcp=%r)' % rcp)
  return sum(frames) / len(frames)


def rim_inflow_mask(df):
  """Boolean mask of the surface-rim-inflow arcs (climate-perturbable)."""
  bi, bj = _split_base(df.i), _split_base(df.j)
  is_inflow = bi.eq('INFLOW')
  excluded = bj.isin(RIM_EXCLUDE_NODES)
  for pre in RIM_EXCLUDE_PREFIXES:
    excluded |= bj.str.startswith(pre)
  return is_inflow & ~excluded


def _node_month_factors(gcm_table):
  """
  {(node_base, month): multiplier} for every mapped rim node, plus the
  ensemble-mean shape under key node_base ``_ENSEMBLE`` for the fallback.
  """
  ens = gcm_table.mean(axis=1)          # 12-vector, mean across the 11 basins
  factors = {(_ENSEMBLE, m): float(ens[m]) for m in gcm_table.index}
  for node, basin in RIM_BASIN_MAP.items():
    for m in gcm_table.index:
      factors[(node, m)] = float(gcm_table.at[m, basin])
  return factors


def apply_warm_shift(df, member=None, rcp=None, wa_target=None,
                     winter_index=None, log=None):
  """
  Perturb the surface rim inflows by a GCM monthly multiplier, rescaled to the
  target water-availability change.

  Construction (design §4.2): take the basin-specific monthly SHAPE from the
  Fourth-Assessment GCM cloud, apply it per rim node by
  :data:`RIM_BASIN_MAP` (unmapped nodes -> ensemble-mean shape), then RESCALE
  so the total annual rim inflow changes by exactly ``wa_target`` and,
  optionally, the Nov-Apr winter fraction hits ``winter_index`` x historical.

  :param df: links DataFrame (mutated in place)
  :param member/rcp: GCM selection (see :func:`load_gcm_multipliers`)
  :param wa_target: signed fractional change in TOTAL rim inflow (e.g. -0.042
    likely, -0.085 worse). None -> apply the raw GCM multipliers with no
    aggregate rescale.
  :param winter_index: target Nov-Apr fraction as a multiple of the historical
    winter fraction (Herman Warm1/2/3 = 1.05/1.10/1.15). None -> keep the GCM
    shape's own seasonality (WA rescale only).
  :returns: (audit DataFrame, info dict with realized wa / winter_index)
  """
  init_bounds = df[['lower_bound', 'upper_bound']].copy()
  mask = rim_inflow_mask(df)
  if not mask.any():
    raise ValueError('No surface rim inflow arcs found in this network')

  gcm = load_gcm_multipliers(member=member, rcp=rcp)
  factors = _node_month_factors(gcm)

  bj = _split_base(df.j)[mask]
  month = df.loc[mask, 'i'].str.split('.').str[1].str.split('-').str[1].astype(int)
  base_ub = df.loc[mask, 'upper_bound'].to_numpy()

  # per-arc raw GCM multiplier (mapped basin shape, else ensemble shape)
  raw = np.array([factors.get((n, m), factors[(_ENSEMBLE, m)])
                  for n, m in zip(bj, month)])

  hist_total = base_ub.sum()
  hist_winter = base_ub[np.isin(month.to_numpy(), WINTER_MONTHS)].sum()

  # WA rescale: uniform factor so the perturbed total hits (1 + wa_target)
  if wa_target is not None:
    perturbed_total = (base_ub * raw).sum()
    scale = (1.0 + wa_target) * hist_total / perturbed_total
    raw = raw * scale

  # WI tilt: multiply winter months by t, non-winter by s, solving
  # t*W' = wi*Whist and s*(T'-W') = T'-wi*Whist while holding the WA total T'.
  realized_wi = None
  if winter_index is not None:
    if wa_target is None:
      raise ValueError('winter_index requires wa_target (rescale to a total '
                       'before tilting its seasonal split)')
    is_w = np.isin(month.to_numpy(), WINTER_MONTHS)
    total = (base_ub * raw).sum()                    # the WA-fixed total T'
    wprime = (base_ub * raw)[is_w].sum()
    target_w = winter_index * hist_winter
    if not (0 < target_w < total):
      raise ValueError('winter_index %.3f infeasible: target winter volume '
                       '%.0f vs total %.0f TAF' % (winter_index, target_w, total))
    t = target_w / wprime
    s = (total - target_w) / (total - wprime)
    raw = raw * np.where(is_w, t, s)
    realized_wi = target_w / hist_winter

  new_ub = _round(base_ub * raw)
  df.loc[mask, 'upper_bound'] = new_ub
  df.loc[mask, 'lower_bound'] = new_ub          # forced inflow: bounds move together

  realized_wa = new_ub.sum() / hist_total - 1.0
  info = {
      'hist_total_tafy': hist_total / _n_years(df.i[mask]),
      'new_total_tafy': new_ub.sum() / _n_years(df.i[mask]),
      'realized_wa': realized_wa,
      'realized_winter_index': (new_ub[np.isin(month.to_numpy(), WINTER_MONTHS)].sum()
                                / hist_winter),
      'n_rim_nodes': int(bj.nunique()),
  }
  if log:
    log.info('Warm-shift (member=%s rcp=%s): rim inflow %.0f -> %.0f TAF/yr '
             '(WA %+.1f%%, winter index %.3f), %d rim nodes'
             % (member or 'ensemble', rcp or 'all', info['hist_total_tafy'],
                info['new_total_tafy'], 100 * realized_wa,
                info['realized_winter_index'], info['n_rim_nodes']))
  return bound_adjustments(df, init_bounds), info


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _round(values):
  return np.round(values, BOUND_DECIMALS)


def _water_year(node_series):
  """Water year (Oct-Sep) from timestamped node names."""
  dt = pd.to_datetime(node_series.str.split('.').str[1]).dt
  return np.where(dt.month >= 10, dt.year + 1, dt.year)


def _n_years(node_series):
  return len(np.unique(_water_year(node_series)))
