"""
COSVF data preparation.

Reads the calvin-network-data repository and produces the five input files
required by the COSVF limited-foresight CALVIN model:

1. ``links.csv`` — single water-year network matrix
2. ``r-dict.json`` — reservoir dictionary with penalty properties
3. ``inflows.csv`` — external inflows for the full period of analysis
4. ``variable-constraints.csv`` — time-varying link bounds
5. ``cosvf-params.csv`` — default penalty parameters (initial EA values)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import Bound, Network, NetworkNode, load_network
from .matrix import (
    DEFAULT_MAX_UB,
    _aggregate_bounds_annual,
    _date_str,
    _generate_annual_steps,
    _generate_steps,
    _lookup_timeseries_value,
    _reconcile_step_cost,
    _resolve_bounds,
    _resolve_costs,
    _resolve_costs_annual,
    _water_year_months,
    build_annual_matrix,
    build_matrix,
)

logger = logging.getLogger(__name__)

# Default reservoir type lists -------------------------------------------

# Type 1: Surface reservoirs with quadratic COSVF penalties
DEFAULT_R_TYPE1 = [
    'SR_BER', 'SR_BUC', 'SR_BUL', 'SR_CLE', 'SR_CLK_INV', 'SR_CMN', 'SR_DNP',
    'SR_EBMUD', 'SR_FOL', 'SR_HTH', 'SR_ISB', 'SR_LL_ENR', 'SR_LVQ', 'SR_MCR',
    'SR_MIL', 'SR_NHG', 'SR_NML', 'SR_ORO', 'SR_PAR', 'SR_PNF', 'SR_RLL_CMB',
    'SR_SHA', 'SR_SNL', 'SR_SFAGG', 'SR_GNT', 'SR_WHI',
]

# Type 2: Groundwater basins with linear COSVF penalties.
# GW_HF and GW_KRN are excluded — their pumping links have UBC=0 (blocked).
DEFAULT_R_TYPE2 = [
    'GW_01', 'GW_02', 'GW_03', 'GW_04', 'GW_05', 'GW_06', 'GW_07',
    'GW_08', 'GW_09', 'GW_10', 'GW_11', 'GW_12', 'GW_13', 'GW_14', 'GW_15',
    'GW_16', 'GW_17', 'GW_18', 'GW_19', 'GW_20', 'GW_21',
    'GW_AV', 'GW_CH', 'GW_EW', 'GW_IM', 'GW_MJ', 'GW_MWD',
    'GW_OW', 'GW_SBV', 'GW_SC', 'GW_SD', 'GW_VC',
]


# Network cost / bound constants -----------------------------------------

# Small negative cost applied to SR_ k=0 carryover links to gently persuade
# the solver to carry water forward rather than leave storage empty.
STORAGE_PERSUASION_COST = -0.02

# Cost applied to SINK and OUTBOUND connector links to gently discourage
# unnecessary flow through these links.
CONNECTOR_COST = 0.01

# Per-node lower bound overrides applied after build_matrix/_annual_matrix.
# These set an absolute lb that supersedes the LBT operational targets in the
# network data, allowing drought drawdown to the minimum operational pool.
NODE_LB_OVERRIDES: dict[str, float] = {
    'SR_SHA': 650,  # TAF — minop pool overrides LBT=2000
    'SR_CLE': 500,  # TAF — minop pool overrides LBT=1100
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_cosvf(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    start: str = '1921-10',
    stop: str = '2003-09',
    wy1_start: str = '1921-10',
    wy1_stop: str = '1922-09',
    r_type1: list[str] | None = None,
    r_type2: list[str] | None = None,
    network: Network | None = None,
    node_lb_overrides: dict[str, float] | None = None,
    storage_persuasion_cost: float = STORAGE_PERSUASION_COST,
    connector_cost: float = CONNECTOR_COST,
) -> Path:
    """Prepare all input files for a COSVF limited-foresight CALVIN run.

    Reads directly from ``calvin-network-data`` via the network module and
    writes the five required files into *output_dir*.

    Parameters
    ----------
    data_path : str or Path
        Path to the ``data/`` directory in ``calvin-network-data``.
    output_dir : str or Path
        Destination directory for the generated files.  Created if it
        does not already exist.
    start : str
        Start of the full period of analysis, YYYY-MM (default 1921-10).
    stop : str
        End of the full period of analysis, YYYY-MM (default 2003-09).
    wy1_start : str
        Start of the first (template) water year, YYYY-MM (default 1921-10).
    wy1_stop : str
        End of the first water year, YYYY-MM (default 1922-09).
    r_type1 : list of str, optional
        Surface reservoirs with quadratic COSVF penalties.
        Defaults to :data:`DEFAULT_R_TYPE1`.
    r_type2 : list of str, optional
        Groundwater basins with linear COSVF penalties.
        Defaults to :data:`DEFAULT_R_TYPE2`.
    network : Network, optional
        Pre-loaded network object.  If ``None``, the network will be loaded
        from *data_path*.

    Returns
    -------
    Path
        The resolved *output_dir* path.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if r_type1 is None:
        r_type1 = DEFAULT_R_TYPE1
    if r_type2 is None:
        r_type2 = DEFAULT_R_TYPE2
    node_lb_overrides = node_lb_overrides or {}

    # Load network
    if network is None:
        logger.info('Loading network from %s ...', data_path)
        network = load_network(data_path)
    logger.info('  %d nodes, %d links', len(network.nodes), len(network.links))

    # ------------------------------------------------------------------
    # 1. Reservoir dictionary (r-dict.json)
    # ------------------------------------------------------------------
    r_dict = _build_reservoir_dict(network, r_type1, r_type2, wy1_start, wy1_stop,
                                    node_lb_overrides=node_lb_overrides)
    r_dict_path = output_dir / 'r-dict.json'
    with open(r_dict_path, 'w') as f:
        json.dump(r_dict, f, sort_keys=False, indent=4, separators=(',', ': '))
    logger.info('  Saved %s (%d reservoirs)', r_dict_path.name, len(r_dict))

    # ------------------------------------------------------------------
    # 2. Inflows (inflows.csv)
    # ------------------------------------------------------------------
    inflows = _extract_inflows(network)
    inflows.to_csv(output_dir / 'inflows.csv')
    logger.info('  Saved inflows.csv (%d records, %d nodes)',
                len(inflows), inflows.j.nunique())

    # ------------------------------------------------------------------
    # 3. Single-year links (links.csv) — built before variable constraints
    #    so we can derive segment counts per edge
    # ------------------------------------------------------------------
    links = _build_links(network, r_dict, r_type1, wy1_start, wy1_stop,
                         node_lb_overrides=node_lb_overrides,
                         storage_persuasion_cost=storage_persuasion_cost,
                         connector_cost=connector_cost)
    links.to_csv(output_dir / 'links.csv', index=False)
    logger.info('  Saved links.csv (%d rows)', len(links))

    # ------------------------------------------------------------------
    # 4. Variable constraints (variable-constraints.csv)
    # ------------------------------------------------------------------
    vc = _extract_variable_constraints(network, r_type1,
                                       {n: network.nodes[n] for n in r_dict},
                                       start, stop,
                                       node_lb_overrides=node_lb_overrides)
    vc.to_csv(output_dir / 'variable-constraints.csv', index=False)
    logger.info('  Saved variable-constraints.csv (%d records)', len(vc))

    # ------------------------------------------------------------------
    # 5. Default COSVF parameters (cosvf-params.csv)
    # ------------------------------------------------------------------
    params = _build_cosvf_params(r_dict)
    params.to_csv(output_dir / 'cosvf-params.csv', index=False)
    logger.info('  Saved cosvf-params.csv')

    return output_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_reservoir_dict(
    network: Network,
    r_type1: list[str],
    r_type2: list[str],
    wy1_start: str,
    wy1_stop: str,
    node_lb_overrides: dict[str, float] | None = None,
) -> dict:
    """Build the reservoir dictionary with penalty type assignments.

    The EA requires that type-2 (linear/GW) reservoirs are indexed before
    type-1 (quadratic/SR) reservoirs in ``cosvf_param_index``.  We enforce
    this by iterating reservoirs in the order: type-2 → type-1 → type-0.

    ``r_dict['lb']`` is set to ``min(WY1_Sep_LBT, node_lb_overrides[r])`` so
    the penalty curve lower endpoint matches the actual reachable minimum
    storage — consistent with the September variable constraints.
    """
    node_lb_overrides = node_lb_overrides or {}
    reservoirs = {name: node for name, node in network.nodes.items()
                  if node.initialstorage is not None}

    steps_wy1 = _generate_steps(wy1_start, wy1_stop)
    sep_idx = next(idx for idx, s in enumerate(steps_wy1) if s.month == 9)

    # Partition reservoirs so ALL GW nodes precede ALL SR nodes (the EA
    # uses the index where SR type-1 params begin to enforce pmax > pmin).
    # Within each group, parametrised types come first: type-2 GW before
    # type-0 GW, type-1 SR before type-0 SR.  Sub-order within each
    # category is alphabetical for determinism.
    r_type2_set = set(r_type2)
    r_type1_set = set(r_type1)
    is_gw = lambda r: r.startswith('GW_')
    ordered_names: list[str] = (
        sorted([r for r in reservoirs if is_gw(r) and r in r_type2_set])
        + sorted([r for r in reservoirs if is_gw(r) and r not in r_type2_set])
        + sorted([r for r in reservoirs if not is_gw(r) and r in r_type1_set])
        + sorted([r for r in reservoirs if not is_gw(r) and r not in r_type1_set])
    )

    r_dict: dict = {}
    i = 0  # running counter for cosvf_param_index
    for r_name in ordered_names:
        node = reservoirs[r_name]
        storage_bounds = _resolve_bounds(node.bounds, steps_wy1)
        lb_9_raw = storage_bounds[sep_idx].lb
        override = node_lb_overrides.get(r_name)
        lb_9 = (min(lb_9_raw, override)
                if override is not None and lb_9_raw is not None
                else lb_9_raw)
        ub_9 = storage_bounds[sep_idx].ub if storage_bounds[sep_idx].ub is not None else DEFAULT_MAX_UB

        if r_name in r_type1_set:
            r_type, cosvf_param_index, k_count = 1, [i, i + 1], 15
            i += 2
        elif r_name in r_type2_set:
            r_type, cosvf_param_index, k_count = 2, i, 2
            i += 1
        else:
            r_type, cosvf_param_index, k_count = 0, None, 1

        r_dict[r_name] = {
            'eop_init': node.initialstorage,
            'lb': lb_9,
            'ub': ub_9,
            'type': r_type,
            'cosvf_param_index': cosvf_param_index,
            'k_count': k_count,
        }

    return r_dict


def _extract_inflows(network: Network) -> pd.DataFrame:
    """Extract all inflow timeseries from network nodes."""
    records: list[dict] = []
    for name, node in network.nodes.items():
        if not node.inflows:
            continue
        for _inflow_name, inflow_data in node.inflows.items():
            if not inflow_data or len(inflow_data) < 2:
                continue
            for row in inflow_data[1:]:  # skip header
                if len(row) >= 2:
                    try:
                        records.append({
                            'date': pd.Timestamp(row[0]),
                            'j': name,
                            'flow_taf': float(row[1]),
                        })
                    except (ValueError, TypeError):
                        continue

    df = pd.DataFrame(records)
    return df.set_index('date').sort_index()


def _sb_different(a, b) -> bool:
    """Return True if two StepBounds have meaningfully different lb or ub."""
    def _diff(v1, v2):
        if v1 is None and v2 is None:
            return False
        if v1 is None or v2 is None:
            return True
        return abs(v1 - v2) > 1e-9
    return _diff(a.lb, b.lb) or _diff(a.ub, b.ub)


def _varies_across_years(
    full_bounds,
    n_per_year: int,
    skip_month_pos: int | None = None,
) -> bool:
    """Return True if any step's bounds differ from the template (first year).

    The first ``n_per_year`` entries are the template.  For each subsequent
    step at position *i*, the corresponding template position is ``i %
    n_per_year``.  Steps at ``skip_month_pos`` are ignored (if provided).
    """
    for i in range(n_per_year, len(full_bounds)):
        month_pos = i % n_per_year
        if skip_month_pos is not None and month_pos == skip_month_pos:
            continue
        if _sb_different(full_bounds[month_pos], full_bounds[i]):
            return True
    return False


def _extract_variable_constraints(
    network: Network,
    r_type1: list[str],
    reservoirs: dict[str, NetworkNode],
    start: str,
    stop: str,
    node_lb_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Extract links/sinks/storage with time-varying bounds.

    Only links whose bounds actually change from one water year to the next
    are included.  Constraints that repeat the same seasonal pattern every
    year are omitted — they are already captured correctly in ``links.csv``
    and do not need to be updated by the rolling COSVF solver.

    For piecewise links, VCs are emitted for **all** k segments so that the
    time-varying physical bounds are correctly split across the cost
    breakpoints — matching the Node.js builder behaviour.
    """
    steps_full = _generate_steps(start, stop)
    n_per_year = 12
    max_ub = DEFAULT_MAX_UB
    node_lb_overrides = node_lb_overrides or {}
    records: list[dict] = []

    # Links with timeseries bounds — emit all k segments.
    for link in network.links:
        if not any(b.type in ('LBT', 'UBT', 'EQT') for b in link.bounds):
            continue
        if link.origin.startswith('INFLOW') or link.origin.startswith('INITIAL'):
            continue
        full_bounds = _resolve_bounds(link.bounds, steps_full)
        if not _varies_across_years(full_bounds, n_per_year):
            continue
        full_costs = _resolve_costs(link.costs, full_bounds, steps_full, max_ub)
        is_eq = any(b.is_equality for b in link.bounds)
        for idx, step in enumerate(steps_full):
            sb = full_bounds[idx]
            cost_segs = full_costs[idx]
            phys = {"LB": sb.lb, "UB": sb.ub}
            for k, seg in enumerate(cost_segs):
                clb, cub = _reconcile_step_cost(
                    seg, phys, len(cost_segs), max_ub
                )
                if is_eq:
                    clb = cub
                records.append({
                    'date': pd.Timestamp(step.isoformat()),
                    'i': f'{link.origin}.{_date_str(step)}',
                    'j': f'{link.terminus}.{_date_str(step)}',
                    'k': k,
                    'lower_bound': clb,
                    'upper_bound': cub,
                })

    # Sinks with timeseries bounds
    for name, node in network.nodes.items():
        if not node.sinks:
            continue
        for sink in node.sinks:
            sink_bounds_list = sink.get('bounds', [])
            if not any(b.type in ('LBT', 'UBT', 'EQT')
                       for b in sink_bounds_list if isinstance(b, Bound)):
                continue
            full_bounds = _resolve_bounds(sink_bounds_list, steps_full)
            if not _varies_across_years(full_bounds, n_per_year):
                continue
            for idx, step in enumerate(steps_full):
                sb = full_bounds[idx]
                records.append({
                    'date': pd.Timestamp(step.isoformat()),
                    'i': f'{name}.{_date_str(step)}',
                    'j': f'SINK.{_date_str(step)}',
                    'k': 0,
                    'lower_bound': sb.lb,
                    'upper_bound': sb.ub if sb.ub is not None else DEFAULT_MAX_UB,
                })

    # Storage nodes with timeseries bounds.
    # Storage links in the matrix go from i.T → i.T+1, so the variable
    # constraint for bounds at timestep T must reference the next step as j.
    # September IS included: the COSVF penalty curve lower endpoint is set to
    # min(WY1_Sep_LBT, lb_override), so the September hard lb must be consistent
    # with that and must relax in drought years as the network LBT does.
    for r_name, node in reservoirs.items():
        if not any(b.type in ('LBT', 'UBT', 'EQT') for b in node.bounds):
            continue
        full_bounds = _resolve_bounds(node.bounds, steps_full)
        if not _varies_across_years(full_bounds, n_per_year):
            continue
        lb_override = node_lb_overrides.get(r_name)
        for idx, step in enumerate(steps_full):
            if idx + 1 >= len(steps_full):
                continue  # no next step to form the link target
            next_step = steps_full[idx + 1]
            sb = full_bounds[idx]
            lb = (min(sb.lb, lb_override) if lb_override is not None and sb.lb is not None
                  else sb.lb)
            records.append({
                'date': pd.Timestamp(step.isoformat()),
                'i': f'{r_name}.{_date_str(step)}',
                'j': f'{r_name}.{_date_str(next_step)}',
                'k': 0,
                'lower_bound': lb,
                'upper_bound': sb.ub if sb.ub is not None else DEFAULT_MAX_UB,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(['date', 'i', 'j', 'k'])
    return df


def _build_links(
    network: Network,
    r_dict: dict,
    r_type1: list[str],
    wy1_start: str,
    wy1_stop: str,
    *,
    node_lb_overrides: dict[str, float] | None = None,
    storage_persuasion_cost: float = STORAGE_PERSUASION_COST,
    connector_cost: float = CONNECTOR_COST,
) -> pd.DataFrame:
    """Build the single water-year links file for COSVF."""
    node_lb_overrides = node_lb_overrides or {}
    links = build_matrix(network, start=wy1_start, stop=wy1_stop, add_debug=True,
                          constrain_ending='all', node_lb_overrides=node_lb_overrides)

    # Add helper columns (only for non-debug links)
    links = links.copy()
    is_dbug = links.i.str.contains('DBUG') | links.j.str.contains('DBUG')
    links['i_node'] = ''
    links['j_node'] = ''
    links['edge'] = ''
    links.loc[~is_dbug, 'i_node'] = links.loc[~is_dbug, 'i'].str.split('.').str[0]
    links.loc[~is_dbug, 'j_node'] = links.loc[~is_dbug, 'j'].str.split('.').str[0]
    links.loc[~is_dbug, 'edge'] = links.loc[~is_dbug, 'i_node'] + '_' + links.loc[~is_dbug, 'j_node']

    # Collapse multi-k surface reservoir storage links to single k=0 with full capacity
    for r in r_type1:
        edge = f'{r}_{r}'
        mask = links.edge == edge
        if not mask.any():
            continue
        storage_links = links[mask].copy()
        k0_mask = mask & (links.k == 0)

        total_ub = storage_links.groupby(['i', 'j'])['upper_bound'].sum()
        links.loc[k0_mask, 'cost'] = storage_persuasion_cost
        for idx in links[k0_mask].index:
            key = (links.at[idx, 'i'], links.at[idx, 'j'])
            if key in total_ub.index:
                links.at[idx, 'upper_bound'] = total_ub[key]

        links = links[~(mask & (links.k > 0))]

    # Initial storage → r_dict values
    for idx in links[links.i_node == 'INITIAL'].index:
        r = links.at[idx, 'j_node']
        if r in r_dict:
            links.at[idx, 'lower_bound'] = r_dict[r]['eop_init']
            links.at[idx, 'upper_bound'] = r_dict[r]['eop_init']

    # Final storage → r_dict lb/ub
    # NOTE: When constrain_ending='all' is passed to build_matrix above, the
    # →FINAL links already have lb=ub=initial_storage (the COSVF steady-state
    # assumption).  We intentionally skip this override to preserve those
    # constraints.  This block is retained (commented) for reference in case
    # constrain_ending is ever set to 'none'.
    # for idx in links[links.j_node == 'FINAL'].index:
    #     r = links.at[idx, 'i_node']
    #     if r in r_dict:
    #         links.at[idx, 'lower_bound'] = r_dict[r]['lb']
    #         links.at[idx, 'upper_bound'] = r_dict[r]['ub']

    # Sink/outbound connector costs
    links.loc[links.i_node.isin(['SINK', 'OUTBOUND']), 'cost'] = connector_cost

    return links[['i', 'j', 'k', 'cost', 'amplitude', 'lower_bound', 'upper_bound']]


def _build_cosvf_params(r_dict: dict) -> pd.DataFrame:
    """Build default COSVF penalty parameters (initial EA values)."""
    param_labels = ['pmin', 'pmax']
    rtype1_list = [k for k, v in r_dict.items() if v['type'] == 1]
    rtype2_list = [k for k, v in r_dict.items() if v['type'] == 2]

    r_col = rtype2_list + list(np.repeat(rtype1_list, len(param_labels)))
    param_col = ['p'] * len(rtype2_list) + param_labels * len(rtype1_list)
    value_col = (list(np.repeat([-1e2], len(rtype2_list)))
                 + list(np.tile([-1e2, -7e2], len(rtype1_list))))

    return pd.DataFrame({'r': r_col, 'param': param_col, 'value': value_col})


# ---------------------------------------------------------------------------
# Annual-step helpers
# ---------------------------------------------------------------------------

def _extract_inflows_annual(
    network: Network,
    start: str,
    stop: str,
) -> pd.DataFrame:
    """Extract annual inflow sums from network nodes.

    Groups monthly inflow records by water year (Oct–Sep) and sums the monthly
    values.  Output rows are indexed by the September 30 end-date of each WY.
    """
    annual_steps = _generate_annual_steps(start, stop)
    records: list[dict] = []

    for name, node in network.nodes.items():
        if not node.inflows:
            continue
        for _inflow_name, inflow_data in node.inflows.items():
            if not inflow_data or len(inflow_data) < 2:
                continue
            for annual_step in annual_steps:
                monthly_steps = _water_year_months(annual_step)
                annual_val = sum(
                    _lookup_timeseries_value(inflow_data, s) or 0.0
                    for s in monthly_steps
                )
                records.append({
                    'date': pd.Timestamp(annual_step.isoformat()),
                    'j': name,
                    'flow_taf': annual_val,
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index('date').sort_index()
    return df


def _build_links_annual(
    network: Network,
    r_dict: dict,
    r_type1: list[str],
    wy1_start: str,
    wy1_stop: str,
    *,
    node_lb_overrides: dict[str, float] | None = None,
    storage_persuasion_cost: float = STORAGE_PERSUASION_COST,
    connector_cost: float = CONNECTOR_COST,
) -> pd.DataFrame:
    """Build the single water-year annual links file for COSVF astep."""
    node_lb_overrides = node_lb_overrides or {}
    links = build_annual_matrix(network, start=wy1_start, stop=wy1_stop,
                                add_debug=True, constrain_ending='all',
                                node_lb_overrides=node_lb_overrides)

    # Add helper columns (only for non-debug links)
    links = links.copy()
    is_dbug = links.i.str.contains('DBUG') | links.j.str.contains('DBUG')
    links['i_node'] = ''
    links['j_node'] = ''
    links['edge'] = ''
    links.loc[~is_dbug, 'i_node'] = links.loc[~is_dbug, 'i'].str.split('.').str[0]
    links.loc[~is_dbug, 'j_node'] = links.loc[~is_dbug, 'j'].str.split('.').str[0]
    links.loc[~is_dbug, 'edge'] = (links.loc[~is_dbug, 'i_node'] + '_'
                                   + links.loc[~is_dbug, 'j_node'])

    # Collapse multi-k surface reservoir storage links to single k=0 with full capacity
    for r in r_type1:
        edge = f'{r}_{r}'
        mask = links.edge == edge
        if not mask.any():
            continue
        storage_links = links[mask].copy()
        k0_mask = mask & (links.k == 0)

        total_ub = storage_links.groupby(['i', 'j'])['upper_bound'].sum()
        links.loc[k0_mask, 'cost'] = storage_persuasion_cost
        for idx in links[k0_mask].index:
            key = (links.at[idx, 'i'], links.at[idx, 'j'])
            if key in total_ub.index:
                links.at[idx, 'upper_bound'] = total_ub[key]

        links = links[~(mask & (links.k > 0))]

    # Initial storage → r_dict values
    for idx in links[links.i_node == 'INITIAL'].index:
        r = links.at[idx, 'j_node']
        if r in r_dict:
            links.at[idx, 'lower_bound'] = r_dict[r]['eop_init']
            links.at[idx, 'upper_bound'] = r_dict[r]['eop_init']

    # Sink/outbound connector costs
    links.loc[links.i_node.isin(['SINK', 'OUTBOUND']), 'cost'] = connector_cost

    return links[['i', 'j', 'k', 'cost', 'amplitude', 'lower_bound', 'upper_bound']]


def _extract_variable_constraints_annual(
    network: Network,
    r_type1: list[str],
    reservoirs: dict[str, NetworkNode],
    start: str,
    stop: str,
    node_lb_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Extract annual variable constraints from network links, sinks, and storage.

    Like ``_extract_variable_constraints()`` but at annual (water-year) resolution.
    Bounds for links and sinks are aggregated by summing 12 monthly values per WY.
    Storage bounds use the September 30 value only.

    Only links whose aggregated bounds actually change from one water year to the
    next are included — constraints that are identical across all years are omitted.
    """
    annual_steps = _generate_annual_steps(start, stop)
    max_ub = DEFAULT_MAX_UB
    node_lb_overrides = node_lb_overrides or {}
    records: list[dict] = []

    # Links with timeseries bounds
    for link in network.links:
        if not any(b.type in ('LBT', 'UBT', 'EQT') for b in link.bounds):
            continue
        if link.origin.startswith('INFLOW') or link.origin.startswith('INITIAL'):
            continue
        is_eq = any(b.is_equality for b in link.bounds)

        # Pre-compute aggregated bounds for all years for the year-constant check.
        all_annual_bounds = [
            _aggregate_bounds_annual(link.bounds, _water_year_months(s), max_ub)
            for s in annual_steps
        ]
        if not _varies_across_years(all_annual_bounds, n_per_year=1):
            continue

        for idx, annual_step in enumerate(annual_steps):
            annual_bounds = all_annual_bounds[idx]
            annual_costs = _resolve_costs_annual(link.costs, annual_bounds,
                                                 _water_year_months(annual_step), max_ub)
            phys = {"LB": annual_bounds.lb, "UB": annual_bounds.ub}
            for k, seg in enumerate(annual_costs):
                clb, cub = _reconcile_step_cost(seg, phys, len(annual_costs), max_ub)
                if is_eq:
                    clb = cub
                records.append({
                    'date': pd.Timestamp(annual_step.isoformat()),
                    'i': f'{link.origin}.{_date_str(annual_step)}',
                    'j': f'{link.terminus}.{_date_str(annual_step)}',
                    'k': k,
                    'lower_bound': clb,
                    'upper_bound': cub,
                })

    # Sinks with timeseries bounds
    for name, node in network.nodes.items():
        if not node.sinks:
            continue
        for sink in node.sinks:
            sink_bounds_list = sink.get('bounds', [])
            if not any(b.type in ('LBT', 'UBT', 'EQT')
                       for b in sink_bounds_list if isinstance(b, Bound)):
                continue
            all_annual_bounds = [
                _aggregate_bounds_annual(sink_bounds_list, _water_year_months(s), max_ub)
                for s in annual_steps
            ]
            if not _varies_across_years(all_annual_bounds, n_per_year=1):
                continue
            for idx, annual_step in enumerate(annual_steps):
                annual_bounds = all_annual_bounds[idx]
                records.append({
                    'date': pd.Timestamp(annual_step.isoformat()),
                    'i': f'{name}.{_date_str(annual_step)}',
                    'j': f'SINK.{_date_str(annual_step)}',
                    'k': 0,
                    'lower_bound': annual_bounds.lb,
                    'upper_bound': annual_bounds.ub if annual_bounds.ub is not None else DEFAULT_MAX_UB,
                })

    # Storage nodes with timeseries bounds — September 30 value per WY.
    # Storage link at annual step T: node.T → node.T+1; skip last step (no next).
    for r_name, node in reservoirs.items():
        if not any(b.type in ('LBT', 'UBT', 'EQT') for b in node.bounds):
            continue
        all_sep_bounds = _resolve_bounds(node.bounds, annual_steps)
        if not _varies_across_years(all_sep_bounds, n_per_year=1):
            continue
        lb_override = node_lb_overrides.get(r_name)
        for idx, annual_step in enumerate(annual_steps):
            if idx + 1 >= len(annual_steps):
                continue  # no next step to form the link target
            next_annual_step = annual_steps[idx + 1]
            sb = all_sep_bounds[idx]
            lb = (min(sb.lb, lb_override) if lb_override is not None and sb.lb is not None
                  else sb.lb)
            records.append({
                'date': pd.Timestamp(annual_step.isoformat()),
                'i': f'{r_name}.{_date_str(annual_step)}',
                'j': f'{r_name}.{_date_str(next_annual_step)}',
                'k': 0,
                'lower_bound': lb,
                'upper_bound': sb.ub if sb.ub is not None else DEFAULT_MAX_UB,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(['date', 'i', 'j', 'k'])
    return df


# ---------------------------------------------------------------------------
# Annual-step public API
# ---------------------------------------------------------------------------

def prepare_pf_astep(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    start: str = '1921-10',
    stop: str = '2003-09',
    network: Network | None = None,
    node_lb_overrides: dict[str, float] | None = None,
    connector_cost: float = CONNECTOR_COST,
) -> Path:
    """Prepare annual-step perfect foresight links file.

    Builds a links.csv with 82 annual timesteps (Sep 30 end-dates) covering
    the full CALVIN period of analysis.

    Parameters
    ----------
    data_path : str or Path
        Path to the ``data/`` directory in ``calvin-network-data``.
    output_dir : str or Path
        Destination directory for the generated ``links.csv``.  Created if
        it does not already exist.
    start : str
        Start of the analysis period, YYYY-MM (default 1921-10).
    stop : str
        End of the analysis period, YYYY-MM (default 2003-09).
    network : Network, optional
        Pre-loaded network object.

    Returns
    -------
    Path
        The resolved *output_dir* path.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if network is None:
        logger.info('Loading network from %s ...', data_path)
        network = load_network(data_path)
    logger.info('  %d nodes, %d links', len(network.nodes), len(network.links))

    node_lb_overrides = node_lb_overrides or {}

    links = build_annual_matrix(network, start=start, stop=stop,
                                add_debug=True, constrain_ending='none',
                                node_lb_overrides=node_lb_overrides)

    # Set connector costs (same as COSVF convention)
    links = links.copy()
    is_dbug = links.i.str.contains('DBUG') | links.j.str.contains('DBUG')
    links['i_node'] = ''
    links.loc[~is_dbug, 'i_node'] = links.loc[~is_dbug, 'i'].str.split('.').str[0]
    links.loc[links.i_node.isin(['SINK', 'OUTBOUND']), 'cost'] = connector_cost
    links = links.drop(columns=['i_node'])

    links.to_csv(output_dir / 'links.csv', index=False)
    logger.info('  Saved links.csv (%d rows)', len(links))

    return output_dir


def prepare_cosvf_astep(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    start: str = '1921-10',
    stop: str = '2003-09',
    wy1_start: str = '1921-10',
    wy1_stop: str = '1922-09',
    r_type1: list[str] | None = None,
    r_type2: list[str] | None = None,
    network: Network | None = None,
    node_lb_overrides: dict[str, float] | None = None,
    storage_persuasion_cost: float = STORAGE_PERSUASION_COST,
    connector_cost: float = CONNECTOR_COST,
) -> Path:
    """Prepare all input files for an annual-step COSVF limited-foresight run.

    Produces the same five-file schema as ``prepare_cosvf()`` but with
    annual timesteps (one water year per step, Sep 30 end-dates).

    Parameters
    ----------
    data_path : str or Path
        Path to the ``data/`` directory in ``calvin-network-data``.
    output_dir : str or Path
        Destination directory.  Created if it does not already exist.
    start : str
        Start of the full period of analysis, YYYY-MM (default 1921-10).
    stop : str
        End of the full period of analysis, YYYY-MM (default 2003-09).
    wy1_start : str
        Start of the first (template) water year, YYYY-MM (default 1921-10).
    wy1_stop : str
        End of the first water year, YYYY-MM (default 1922-09).
    r_type1 : list of str, optional
        Surface reservoirs with quadratic COSVF penalties.
    r_type2 : list of str, optional
        Groundwater basins with linear COSVF penalties.
    network : Network, optional
        Pre-loaded network object.

    Returns
    -------
    Path
        The resolved *output_dir* path.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if r_type1 is None:
        r_type1 = DEFAULT_R_TYPE1
    if r_type2 is None:
        r_type2 = DEFAULT_R_TYPE2
    node_lb_overrides = node_lb_overrides or {}

    if network is None:
        logger.info('Loading network from %s ...', data_path)
        network = load_network(data_path)
    logger.info('  %d nodes, %d links', len(network.nodes), len(network.links))

    # ------------------------------------------------------------------
    # 1. Reservoir dictionary (r-dict.json) — reuse existing helper
    # ------------------------------------------------------------------
    r_dict = _build_reservoir_dict(network, r_type1, r_type2, wy1_start, wy1_stop,
                                    node_lb_overrides=node_lb_overrides)
    r_dict_path = output_dir / 'r-dict.json'
    with open(r_dict_path, 'w') as f:
        json.dump(r_dict, f, sort_keys=False, indent=4, separators=(',', ': '))
    logger.info('  Saved %s (%d reservoirs)', r_dict_path.name, len(r_dict))

    # ------------------------------------------------------------------
    # 2. Annual inflows (inflows.csv)
    # ------------------------------------------------------------------
    inflows = _extract_inflows_annual(network, start, stop)
    inflows.to_csv(output_dir / 'inflows.csv')
    logger.info('  Saved inflows.csv (%d records, %d nodes)',
                len(inflows), inflows.j.nunique() if not inflows.empty else 0)

    # ------------------------------------------------------------------
    # 3. Single-step annual links (links.csv)
    # ------------------------------------------------------------------
    links = _build_links_annual(network, r_dict, r_type1, wy1_start, wy1_stop,
                                node_lb_overrides=node_lb_overrides,
                                storage_persuasion_cost=storage_persuasion_cost,
                                connector_cost=connector_cost)
    links.to_csv(output_dir / 'links.csv', index=False)
    logger.info('  Saved links.csv (%d rows)', len(links))

    # ------------------------------------------------------------------
    # 4. Annual variable constraints (variable-constraints.csv)
    # ------------------------------------------------------------------
    vc = _extract_variable_constraints_annual(
        network, r_type1,
        {n: network.nodes[n] for n in r_dict},
        start, stop,
        node_lb_overrides=node_lb_overrides,
    )
    vc.to_csv(output_dir / 'variable-constraints.csv', index=False)
    logger.info('  Saved variable-constraints.csv (%d records)', len(vc))

    # ------------------------------------------------------------------
    # 5. Default COSVF parameters (cosvf-params.csv) — reuse existing helper
    # ------------------------------------------------------------------
    params = _build_cosvf_params(r_dict)
    params.to_csv(output_dir / 'cosvf-params.csv', index=False)
    logger.info('  Saved cosvf-params.csv')

    return output_dir
