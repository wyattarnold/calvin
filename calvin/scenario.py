"""
Institutionally constrained scenario for CALVIN runs.

Applies three bound edits to a time-expanded links DataFrame, all anchored
to the flows of a completed UNCONSTRAINED reference ("market") run so the
reference solution remains feasible by construction (floors rounded down
below reference, caps rounded up above it):

1. Delivery floors — per-arc monthly lower bounds at fraction x reference
   flow on demand arcs (ag by default), spread across piecewise k segments
   filling k=0 (highest-value water) first. Limits ag->urban reallocation
   to roughly the observed 2-5% market volume.
2. GW pumping caps — upper bounds at factor x reference flow on the
   GW_xx -> HGPxx pumping choke arcs. Closes the implicit GW-substitution
   channel (pump extra local groundwater to free surface water for export).
3. GW export closures — zero the configured native-GW-to-conveyance export
   arcs; real operating banks/exports stay open by default.
4. Delivery ceilings (optional, ag_ceiling_fraction > 0) — upper bounds at
   fraction x reference flow, pinning deliveries into the [floor, ceiling]
   x reference band. The Policy-3 analog: the baseline then mimics the
   reference operations instead of merely never undercutting them.

Market-anchored bounds cannot bind the market run itself (floors sit below
its flows, caps above); they bind when another constraint creates
reallocation pressure — canonically constrain_ending="gw" in the matrix
builder (no net GW overdraft), which removes the mined supply while the
floors/caps block the trading and GW-substitution escapes.

This is the modern analog of Howitt et al. (1999) Policy 2 ("Water Market
with Minimum Deliveries"): minimum deliveries per contracts/rights with
dry-year deficiencies (inherited automatically by scaling the reference
run's monthly flows), operations otherwise free. The pumping cap borrows
from their Policy 3, relaxed from equality to an upper bound.

Functions are pure (df, ref_flows, params) so they can later be reused for
COSVF single-year networks via a re-dated ref_flows adapter.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOUND_DECIMALS = 3  # matches calvin.capacity / matrix conventions

# Known native-GW-to-conveyance export arcs (base i-j names, no date suffix)
NATIVE_GW_EXPORT_ARCS = (
    'HGP21-C98',     # Tulare native GW into the Friant-Kern/aqueduct system
    'HGPKRN-D865',   # Kern water bank return to Metropolitan
    'HGP19-D851',    # Semitropic arc: nominally a bank return, but GW_19 is
                     # the real Semitropic basin and the market runs export
                     # through it with ZERO prior HAR19 recharge — in-model
                     # it is native-GW export, so it defaults to closed.
                     # (A recovery <= cumulative-recharge constraint would be
                     # the faithful banking representation; Phase 3 work.)
    'HGPHF-PMP_JH',  # Hayfield storage to Julian Hinds (Colorado R. Aqueduct)
    'HGPOW-C120',    # Owens Valley GW into the LA Aqueduct (LADWP)
)
# Kern (GW_KRN) and Hayfield (GW_HF) are dedicated bank constructs that start
# empty, so their recovery is self-limited to prior deposits — institutionally
# faithful, kept open. Owens is LADWP's real, ongoing native export — kept.
DEFAULT_EXPORT_CLOSE = ('HGP21-C98', 'HGP19-D851')
DEFAULT_EXPORT_KEEP = ('HGPKRN-D865', 'HGPHF-PMP_JH', 'HGPOW-C120')

DEFAULT_DEMAND_NODES_CSV = os.path.join(BASE_DIR, 'data', 'demand_nodes.csv')


def _floor_round(values):
  return np.floor(np.clip(values, 0.0, None) * 10 ** BOUND_DECIMALS) / 10 ** BOUND_DECIMALS


def _ceil_round(values):
  return np.ceil(np.clip(values, 0.0, None) * 10 ** BOUND_DECIMALS) / 10 ** BOUND_DECIMALS


def load_reference_flows(results_dir):
  """
  Read flow.csv from a completed unconstrained run.

  :param results_dir: (string) results directory of the reference run
  :returns: DataFrame indexed by date string (YYYY-MM-DD), columns are
    hyphen-joined 'i-j' base link names (k-summed by the postprocessor)
  """
  path = os.path.join(results_dir, 'flow.csv')
  if not os.path.isfile(path):
    raise FileNotFoundError(
        'Reference flows not found: %s. Run the unconstrained market '
        'baseline first (a calvin-pf-cap run with the [scenario] table '
        'absent or enabled = false).' % path)
  return pd.read_csv(path, index_col=0)


def demand_links(demand_type, demand_nodes_csv=None):
  """Base 'i-j' link names of the given type ('ag' or 'urban') from demand_nodes.csv."""
  path = demand_nodes_csv or DEFAULT_DEMAND_NODES_CSV
  nodes = pd.read_csv(path)
  return sorted(nodes.loc[nodes['type'].str.strip() == demand_type, 'link'].str.strip())


def _split_base(series):
  """Strip the .YYYY-MM-DD suffix from a node-name Series."""
  return series.str.split('.').str[0]


def _reference_series(ref_flows, dates, ij_names):
  """
  Long-form reference lookup: Series indexed by (date, ij).

  Raises ValueError when the reference run does not cover this network's
  links or dates (reference drift protection).
  """
  missing_cols = sorted(set(ij_names) - set(ref_flows.columns))
  if missing_cols:
    raise ValueError(
        'Reference results do not match this network; flow.csv is missing '
        'link columns such as %s' % missing_cols[:5])
  missing_dates = sorted(set(dates) - set(ref_flows.index))
  if missing_dates:
    raise ValueError(
        'Reference results do not cover this run period; missing dates '
        'such as %s' % missing_dates[:5])
  long = ref_flows[sorted(set(ij_names))].stack()
  long.index.names = ['date', 'ij']
  return long


def apply_delivery_floors(df, ref_flows, links, fraction, log=None):
  """
  Set monthly delivery lower bounds at ``fraction`` x reference flow.

  The per-(link, month) floor is spread across piecewise k segments filling
  k=0 first (lb_k = min(remaining, ub_k)), which binds identically to an
  aggregate floor because optimal solutions always fill the most-negative-
  cost segments first. Upper bounds are never touched, so shortage
  accounting (ub - flow) is unaffected.

  :param df: links DataFrame (mutated in place)
  :param ref_flows: reference flow table from :func:`load_reference_flows`
  :param links: iterable of base 'i-j' link names to floor
  :param fraction: floor as a fraction of reference flow (rounded down)
  :returns: number of (link, month) groups floored
  """
  links = set(links)
  if not links or fraction <= 0:
    return 0

  i_base = _split_base(df.i)
  j_base = _split_base(df.j)
  ij = i_base + '-' + j_base
  mask = ij.isin(links)
  if not mask.any():
    raise ValueError('None of the %d floor links exist in this network' % len(links))

  sub = df.loc[mask, ['k', 'upper_bound']].copy()
  sub['ij'] = ij[mask]
  sub['date'] = df.loc[mask, 'i'].str.split('.').str[1]

  ref = _reference_series(ref_flows, sub['date'].unique(), sub['ij'].unique())
  ref_vals = ref.reindex(
      pd.MultiIndex.from_arrays([sub['date'], sub['ij']])).to_numpy()
  if np.isnan(ref_vals).any():
    raise ValueError('Reference flows contain gaps for floored links')
  sub['floor'] = _floor_round(ref_vals * fraction)

  sub.sort_values(['ij', 'date', 'k'], inplace=True)
  grouped = sub.groupby(['ij', 'date'], sort=False)
  prev_cum = grouped['upper_bound'].cumsum() - sub['upper_bound']
  lb_new = np.minimum(np.clip(sub['floor'] - prev_cum, 0.0, None), sub['upper_bound'])

  leftover = sub['floor'] - grouped['upper_bound'].transform('sum')
  bad = leftover > 1e-6
  if bad.any():
    worst = sub.loc[bad, ['ij', 'date']].drop_duplicates().head(5)
    raise ValueError(
        'Delivery floors exceed total demand capacity (reference drift?) '
        'on e.g. %s' % worst.to_dict('records'))

  # Explicit ordering: pull existing lbs in lb_new's index order, compare
  # positionally, write back against that same order (avoids ufunc
  # index-alignment surprises on the shuffled sort order).
  existing = df['lower_bound'].loc[lb_new.index].to_numpy()
  df.loc[lb_new.index, 'lower_bound'] = np.maximum(existing, lb_new.to_numpy())

  n_groups = grouped.ngroups
  if log:
    log.info('Scenario: floored %d delivery arcs at %.0f%% of reference '
             'across %d (link, month) groups'
             % (len(links & set(sub.ij)), fraction * 100, n_groups))
  return n_groups


def apply_delivery_ceilings(df, ref_flows, links, fraction, log=None):
  """
  Cap monthly deliveries at ``fraction`` x reference flow.

  The Policy-3 analog ("mimic current operations"): where floors keep a
  region from losing its reference deliveries, ceilings keep it from
  gaining beyond them, so deliveries on the given links are pinned into
  the [floor_fraction, ceiling_fraction] x reference band. The aggregate
  ceiling is spread across piecewise k segments filling k=0 first,
  mirroring :func:`apply_delivery_floors`, which binds identically to an
  aggregate ceiling at any LP optimum.

  Call after apply_delivery_floors; raises if a ceiling falls below the
  floors already on the segments.

  NOTE: the postprocessor's shortage metrics measure ub - flow, so under
  ceilings they understate unmet demand relative to FULL demand. Recover
  full-demand shortage from scenario_adjustments.csv (ub_init vs flow).

  :param df: links DataFrame (mutated in place)
  :param ref_flows: reference flow table from :func:`load_reference_flows`
  :param links: iterable of base 'i-j' link names to cap
  :param fraction: ceiling as a fraction of reference flow (rounded up)
  :returns: number of (link, month) groups capped
  """
  links = set(links)
  if not links or fraction <= 0:
    return 0

  i_base = _split_base(df.i)
  j_base = _split_base(df.j)
  ij = i_base + '-' + j_base
  mask = ij.isin(links)
  if not mask.any():
    raise ValueError('None of the %d ceiling links exist in this network' % len(links))

  sub = df.loc[mask, ['k', 'lower_bound', 'upper_bound']].copy()
  sub['ij'] = ij[mask]
  sub['date'] = df.loc[mask, 'i'].str.split('.').str[1]

  ref = _reference_series(ref_flows, sub['date'].unique(), sub['ij'].unique())
  ref_vals = ref.reindex(
      pd.MultiIndex.from_arrays([sub['date'], sub['ij']])).to_numpy()
  if np.isnan(ref_vals).any():
    raise ValueError('Reference flows contain gaps for ceiling links')
  sub['ceiling'] = _ceil_round(ref_vals * fraction)

  sub.sort_values(['ij', 'date', 'k'], inplace=True)
  grouped = sub.groupby(['ij', 'date'], sort=False)
  lb_sum = grouped['lower_bound'].transform('sum')
  bad = sub['ceiling'] < lb_sum - 1e-9
  if bad.any():
    worst = sub.loc[bad, ['ij', 'date']].drop_duplicates().head(5)
    raise ValueError(
        'Delivery ceilings fall below existing floors on e.g. %s — is the '
        'ceiling fraction below the floor fraction?' % worst.to_dict('records'))

  # same prefix-fill as the floors: both fill k=0 first from the original
  # segment ubs, and ceiling >= floor total, so each segment keeps
  # ub >= lb by construction
  prev_cum = grouped['upper_bound'].cumsum() - sub['upper_bound']
  ub_new = np.minimum(np.clip(sub['ceiling'] - prev_cum, 0.0, None),
                      sub['upper_bound'])
  ub_new = np.maximum(ub_new, sub['lower_bound'])

  existing = df['upper_bound'].loc[ub_new.index].to_numpy()
  df.loc[ub_new.index, 'upper_bound'] = np.minimum(existing, ub_new.to_numpy())

  n_groups = grouped.ngroups
  if log:
    log.info('Scenario: capped %d delivery arcs at %.0f%% of reference '
             'across %d (link, month) groups'
             % (len(links & set(sub.ij)), fraction * 100, n_groups))
  return n_groups


def apply_gw_pump_caps(df, ref_flows, factor=1.0, log=None, protect=None):
  """
  Cap GW_xx -> HGPxx pumping choke arcs at ``factor`` x reference flow.

  :param df: links DataFrame (mutated in place)
  :param factor: cap as a multiple of reference flow (rounded up)
  :param protect: optional Index of df row labels to leave uncapped
    (injected expansion segments — new infrastructure, not the existing
    system the caps freeze)
  :returns: number of arc rows capped
  """
  i_base = _split_base(df.i)
  j_base = _split_base(df.j)
  mask = i_base.str.startswith('GW_') & j_base.str.startswith('HGP')
  if protect is not None and len(protect):
    mask &= ~df.index.isin(protect)
  if not mask.any():
    raise ValueError('No GW->HGP pumping choke arcs found in this network')

  ij = (i_base + '-' + j_base)[mask]
  dates = df.loc[mask, 'i'].str.split('.').str[1]

  ref = _reference_series(ref_flows, dates.unique(), ij.unique())
  caps = _ceil_round(ref.reindex(
      pd.MultiIndex.from_arrays([dates, ij])).to_numpy() * factor)

  df.loc[mask, 'upper_bound'] = np.minimum(df.loc[mask, 'upper_bound'], caps)

  bad = df.loc[mask, 'lower_bound'] > df.loc[mask, 'upper_bound']
  if bad.any():
    raise ValueError('GW pumping caps violate existing lower bounds on %s'
                     % df.loc[mask].loc[bad].index[:5].tolist())
  if log:
    log.info('Scenario: capped %d GW pumping arc rows (%d basins) at '
             '%.2f x reference' % (mask.sum(), ij.nunique(), factor))
  return int(mask.sum())


def apply_gw_export_closures(df, close=DEFAULT_EXPORT_CLOSE,
                             keep=DEFAULT_EXPORT_KEEP, log=None):
  """
  Zero the upper bound on configured GW export arcs.

  :param close: base 'i-j' arc names to close (upper bound -> 0)
  :param keep: arcs deliberately left open (validated for explicitness)
  :returns: number of rows zeroed
  """
  close, keep = set(close), set(keep)
  known = set(NATIVE_GW_EXPORT_ARCS)
  unknown = (close | keep) - known
  if unknown:
    raise ValueError('Unknown GW export arcs %s; known: %s'
                     % (sorted(unknown), sorted(known)))
  overlap = close & keep
  if overlap:
    raise ValueError('Arcs in both close and keep: %s' % sorted(overlap))
  unlisted = known - close - keep
  if unlisted and log:
    log.warning('Scenario: GW export arcs neither closed nor kept '
                '(left open): %s' % sorted(unlisted))

  i_base = _split_base(df.i)
  j_base = _split_base(df.j)
  ij = i_base + '-' + j_base
  n_rows = 0
  for arc in sorted(close):
    mask = ij == arc
    if not mask.any():
      raise ValueError('Export arc %s not found in this network' % arc)
    if (df.loc[mask, 'lower_bound'] > 0).any():
      raise ValueError('Export arc %s carries a positive lower bound' % arc)
    df.loc[mask, 'upper_bound'] = 0.0
    n_rows += int(mask.sum())
    if log:
      log.info('Scenario: closed GW export arc %s (%d rows)' % (arc, mask.sum()))
  return n_rows



def bound_adjustments(df, initial_bounds):
  """
  Diff current vs snapshot bounds (same columns as
  CALVIN.get_bound_adjustments); changed rows only.
  """
  cur = df[['i', 'j', 'k', 'lower_bound', 'upper_bound']].copy()
  cur = cur.join(initial_bounds.rename(
      columns={'lower_bound': 'lb_init', 'upper_bound': 'ub_init'}))
  cur['lb_final'] = cur['lower_bound']
  cur['ub_final'] = cur['upper_bound']
  cur['lb_delta'] = cur['lb_final'] - cur['lb_init']
  cur['ub_delta'] = cur['ub_final'] - cur['ub_init']
  changed = cur[(cur['lb_delta'] != 0) | (cur['ub_delta'] != 0)]
  return changed[['i', 'j', 'k', 'lb_init', 'lb_final', 'lb_delta',
                  'ub_init', 'ub_final', 'ub_delta']].copy()



def apply_scenario(df, scenario, log=None, protect=None):
  """
  Apply the institutionally constrained scenario to a links DataFrame.

  :param df: links DataFrame (mutated in place)
  :param scenario: dict with keys: reference_results (path to the market
    run's results dir), ag_floor_fraction (default 0.95),
    urban_floor_fraction (default 0.0 = off), ag_ceiling_fraction
    (default 0.0 = off; must be >= ag_floor_fraction when set),
    gw_pump_cap_factor (default 1.0; 0 disables),
    gw_export_close / gw_export_keep (arc name lists)
  :param protect: optional Index of df row labels the scenario must not
    touch (capacity-expansion segments injected by CALVINCap)
  :returns: audit DataFrame of every bound changed (see bound_adjustments)
  """
  known = {'enabled', 'reference_results', 'ag_floor_fraction',
           'urban_floor_fraction', 'ag_ceiling_fraction',
           'gw_pump_cap_factor', 'gw_export_close', 'gw_export_keep'}
  unknown = set(scenario) - known
  if unknown:
    raise ValueError(
        'Unknown [scenario] keys %s. Retired keys (delta_outflow_floor, '
        'delta_outflow_value, urban_sw_floor, delta_export_cap, '
        'ag_floor_source, conveyance_relax) no longer exist; see '
        'tmp/notes/dcr-retirement.md.' % sorted(unknown))

  pre = df[['lower_bound', 'upper_bound']].copy()
  ref = load_reference_flows(scenario['reference_results'])

  ag_fraction = scenario.get('ag_floor_fraction', 0.95)
  if ag_fraction > 0:
    apply_delivery_floors(df, ref, demand_links('ag'), ag_fraction, log=log)

  urban_fraction = scenario.get('urban_floor_fraction', 0.0)
  if urban_fraction > 0:
    apply_delivery_floors(df, ref, demand_links('urban'), urban_fraction,
                          log=log)

  ceiling_fraction = scenario.get('ag_ceiling_fraction', 0.0)
  if ceiling_fraction > 0:
    if ceiling_fraction < ag_fraction:
      raise ValueError('ag_ceiling_fraction (%s) must be >= '
                       'ag_floor_fraction (%s)'
                       % (ceiling_fraction, ag_fraction))
    apply_delivery_ceilings(df, ref, demand_links('ag'), ceiling_fraction,
                            log=log)

  gw_factor = scenario.get('gw_pump_cap_factor', 1.0)
  if gw_factor and gw_factor > 0:
    apply_gw_pump_caps(df, ref, gw_factor, log=log, protect=protect)

  apply_gw_export_closures(
      df,
      close=scenario.get('gw_export_close', DEFAULT_EXPORT_CLOSE),
      keep=scenario.get('gw_export_keep', DEFAULT_EXPORT_KEEP),
      log=log)

  return bound_adjustments(df, pre)
