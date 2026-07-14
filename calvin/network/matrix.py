"""
Matrix builder.

Transforms a Network object into the time-expanded i,j,k,cost,amplitude,
lower_bound,upper_bound matrix consumed by the CALVIN Pyomo solver.

This is the Python replacement for the entire nodejs/matrix/ directory
in calvin-network-tools.
"""

from __future__ import annotations

import calendar
import csv
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from .loader import (
    Bound,
    CostDef,
    Network,
    NetworkLink,
    NetworkNode,
    MONTHS,
)

logger = logging.getLogger(__name__)

# Default time range for CALVIN (water years 1922–2003 = Oct 1921 to Sep 2003)
DEFAULT_START = "1921-10"
DEFAULT_STOP = "2003-09"

# Default large upper bound for unconstrained physical links (TAF).
# 1e6 TAF >> any physically plausible monthly flow in California (~20,000 TAF/month peak),
# while keeping the LP coefficient range numerically tractable for HiGHS.
DEFAULT_MAX_UB = 1_000_000.0

# Upper bound for SOURCE→INITIAL and FINAL→SINK aggregate connectors.
# These carry the *sum* of all reservoir initial/final conditions simultaneously,
# which can exceed DEFAULT_MAX_UB, so they get a separate (larger) cap.
AGGREGATE_MAX_UB = 1_000_000_000.0  # 1e9

# Rounding precision
COST_AMP_DECIMALS = 5
BOUND_DECIMALS = 3


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _end_of_month(year: int, month: int) -> date:
    """Return the last day of the given month."""
    _, last_day = calendar.monthrange(year, month)
    return date(year, month, last_day)


def _generate_steps(
    start: str = DEFAULT_START,
    stop: str = DEFAULT_STOP,
) -> list[date]:
    """Generate monthly end-of-month dates from start to stop (inclusive).

    Parameters
    ----------
    start : str
        Start date as YYYY-MM (first month included).
    stop : str
        Stop date as YYYY-MM (last month included).
    """
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in stop.split("-"))

    steps = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        steps.append(_end_of_month(y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return steps


def _date_str(d: date) -> str:
    """Format a date as YYYY-MM-DD for node IDs."""
    return d.isoformat()


def _month_index(d: date) -> int:
    """0-based month index (0=JAN, 11=DEC)."""
    return d.month - 1


def _month_name(d: date) -> str:
    """Three-letter uppercase month name."""
    return MONTHS[_month_index(d)]


# ---------------------------------------------------------------------------
# Bound computation
# ---------------------------------------------------------------------------

@dataclass
class StepBounds:
    """Resolved bounds for a single timestep."""
    lb: float = 0.0
    ub: float | None = None  # None = unbounded
    lb_defined: bool = False


def _resolve_bounds(
    bounds: list[Bound],
    steps: list[date],
) -> list[StepBounds]:
    """Resolve bound definitions to per-timestep lb/ub values.

    Multiple bounds stack: LB takes the max of all lowers,
    UB takes the min of all uppers.
    """
    result = [StepBounds() for _ in steps]

    for b in bounds:
        if b.type == "NOB":
            continue

        for i, step in enumerate(steps):
            val = _get_bound_value(b, step)
            if val is None:
                continue

            val = float(val)
            if b.type in ("LBC", "LBM", "LBT"):
                result[i].lb = max(result[i].lb, val)
                result[i].lb_defined = True
            elif b.type in ("UBC", "UBM", "UBT"):
                if result[i].ub is None:
                    result[i].ub = val
                else:
                    result[i].ub = min(result[i].ub, val)
            elif b.type in ("EQC", "EQM", "EQT"):
                # Equality constraint: both lb and ub = value
                result[i].lb = val
                result[i].ub = val
                result[i].lb_defined = True

    return result


def _get_bound_value(b: Bound, step: date) -> float | None:
    """Extract the bound value for a specific timestep."""
    if b.is_constant:
        return b.bound
    if b.is_monthly:
        month_key = _month_name(step)
        if isinstance(b.bound, dict):
            return b.bound.get(month_key, b.bound.get(month_key.lower()))
        if isinstance(b.bound, list):
            # Monthly data stored as CSV-like table:
            # [['date', 'kaf'], ['JAN', 22.895], ['FEB', 20.679], ...]
            for row in b.bound[1:]:  # skip header
                if len(row) >= 2 and str(row[0]).upper() == month_key:
                    try:
                        return float(row[1])
                    except (ValueError, TypeError):
                        return None
        return None
    if b.is_timeseries:
        return _lookup_timeseries_value(b.bound, step)
    return None


def _lookup_timeseries_value(data: Any, step: date) -> float | None:
    """Look up a value in a timeseries (list of [date, value] rows)."""
    if not data or not isinstance(data, list):
        return None
    step_str = _date_str(step)
    for row in data[1:]:  # skip header
        if len(row) >= 2 and str(row[0]) == step_str:
            try:
                return float(row[1])
            except (ValueError, TypeError):
                return None
    return None


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

@dataclass
class CostSegment:
    """A single piecewise cost segment."""
    cost: float = 0.0
    lb: float = 0.0
    ub: float | None = None  # None = unbounded


def _resolve_costs(
    cost_def: CostDef | None,
    step_bounds: list[StepBounds],
    steps: list[date],
    max_ub: float = DEFAULT_MAX_UB,
) -> list[list[CostSegment]]:
    """Resolve cost definitions to per-timestep lists of cost segments.

    For piecewise linear penalty functions, each timestep may have
    multiple segments (different k values).
    """
    n = len(steps)
    if not cost_def or cost_def.type in ("NONE", "None"):
        return [[CostSegment(cost=0.0, lb=0.0, ub=None)] for _ in range(n)]

    if cost_def.type == "Constant":
        c = cost_def.cost or 0.0
        return [[CostSegment(cost=c, lb=0.0, ub=None)] for _ in range(n)]

    if cost_def.type == "Annual Variable":
        # Single penalty function for all months (keyed as "JAN-DEC" or first key)
        penalty_data = None
        if cost_def.costs:
            # Try standard keys
            for key in cost_def.costs:
                penalty_data = cost_def.costs[key]
                break  # just use the first (and usually only) one
        if penalty_data:
            # Must compute per-timestep: bounds vary per step and affect
            # negative-slope extension/k=0 adjustments.
            result = []
            for i in range(n):
                segments = _penalty_to_segments(penalty_data, max_ub, step_bounds[i])
                result.append(segments)
            return result
        return [[CostSegment(cost=0.0, lb=0.0, ub=None)] for _ in range(n)]

    if cost_def.type == "Monthly Variable":
        result = []
        for i, step in enumerate(steps):
            month_key = _month_name(step)
            penalty_data = None
            if cost_def.costs:
                penalty_data = cost_def.costs.get(month_key)
            if penalty_data:
                segments = _penalty_to_segments(penalty_data, max_ub, step_bounds[i])
                result.append(segments)
            else:
                result.append([CostSegment(cost=0.0, lb=0.0, ub=None)])
        return result

    return [[CostSegment(cost=0.0, lb=0.0, ub=None)] for _ in range(n)]


def _penalty_to_segments(
    data: list[list],
    max_ub: float = DEFAULT_MAX_UB,
    bounds: StepBounds | None = None,
) -> list[CostSegment]:
    """Convert a piecewise linear penalty function to cost segments.

    Input is a list of [capacity, cumulative_cost] breakpoints.
    Output is a list of segments with marginal costs.

    When *bounds* is provided, the segment list is adjusted to match the
    Node.js ``penalty_costs()`` logic in ``cost.js``:

    * **Negative-slope** links: k=0 UB is widened to the absolute capacity
      of the 2nd breakpoint (absorbing all pre-penalty capacity into the
      steepest cost segment).  An extension segment at zero cost is only
      added when the penalty region does not cover the physical UB.
    * **Positive-slope** links: the last segment UB is left as ``None``
      (unbounded) so that ``_reconcile_step_cost`` can cap it later.
    """
    if not data or len(data) < 2:
        return [CostSegment(cost=0.0, lb=0.0, ub=None)]

    # Skip header row if present
    points = []
    for row in data:
        if len(row) >= 2:
            try:
                cap = float(row[0])
                cost = float(row[1])
                points.append((cap, cost))
            except (ValueError, TypeError):
                continue  # skip header

    if len(points) < 2:
        return [CostSegment(cost=0.0, lb=0.0, ub=None)]

    # Sort by capacity
    points.sort(key=lambda p: p[0])

    segments = []
    for i in range(1, len(points)):
        cap_diff = points[i][0] - points[i - 1][0]
        if cap_diff <= 0:
            continue
        marginal = (points[i][1] - points[i - 1][1]) / cap_diff
        seg = CostSegment(
            cost=round(marginal, COST_AMP_DECIMALS),
            lb=0.0,
            ub=round(cap_diff, BOUND_DECIMALS),
        )
        segments.append(seg)

    if not segments:
        return [CostSegment(cost=0.0, lb=0.0, ub=None)]

    # Absolute capacity at the last breakpoint
    last_bound = points[-1][0]

    is_negative_slope = any(s.cost < 0 for s in segments)

    if is_negative_slope:
        # --- Match Node.js cost.js negative-slope logic (lines 56-92) ---
        # Adjust k=0 LB/UB based on physical bounds.
        # When the LB is *not* explicitly defined (the common case for
        # demand links), widen k=0 UB to the absolute capacity of the 2nd
        # breakpoint so all pre-penalty capacity is priced at the steepest
        # marginal cost.  When LB *is* defined but differs from the
        # segment default, apply the same widening.
        if bounds is not None:
            if bounds.lb_defined:
                if segments[0].lb != bounds.lb:
                    segments[0].lb = bounds.lb
                    if len(points) >= 2:
                        segments[0].ub = round(points[1][0], BOUND_DECIMALS)
            else:
                segments[0].lb = 0.0
                if len(points) >= 2:
                    segments[0].ub = round(points[1][0], BOUND_DECIMALS)

            # Conditionally add a zero-cost extension segment.
            # Node.js only adds one when the penalty region does not fully
            # cover the physical UB.
            if bounds.ub is not None:
                if last_bound < bounds.ub:
                    segments.append(CostSegment(
                        cost=0.0, lb=0.0,
                        ub=round(bounds.ub - last_bound, BOUND_DECIMALS),
                    ))
                # else: penalty already covers UB — no extension
            else:
                # No physical UB: extend to max_ub
                segments.append(CostSegment(cost=0.0, lb=0.0, ub=max_ub))
        else:
            # No bounds supplied — safe fallback (extend to unbounded)
            segments.append(CostSegment(cost=0.0, lb=0.0, ub=None))
    else:
        # Positive slope: leave last segment unbounded; _reconcile_step_cost
        # will cap it with the physical UB.
        segments[-1].ub = None

    return segments


# ---------------------------------------------------------------------------
# Step-cost reconciliation
# ---------------------------------------------------------------------------

def _reconcile_step_cost(
    cost_seg: CostSegment,
    phys_bounds: dict,
    n_segments: int,
    max_ub: float = DEFAULT_MAX_UB,
) -> tuple[float, float]:
    """Reconcile a cost segment's bounds with physical bounds.

    This matches the Node.js stepCost.js logic exactly. For multi-segment
    costs, ``phys_bounds`` is **mutated** — the consumed LB and UB are
    subtracted so subsequent k segments see the remainder.

    Parameters
    ----------
    cost_seg : CostSegment
        The cost segment for this k value.
    phys_bounds : dict
        Mutable dict with keys ``"LB"`` and ``"UB"``. Will be mutated
        for multi-segment costs.
    n_segments : int
        Total number of cost segments for this link/timestep.
    max_ub : float
        Default upper bound when unbounded.

    Returns
    -------
    tuple of (clb, cub)
    """
    seg_lb = cost_seg.lb or 0.0
    seg_ub = cost_seg.ub
    p_lb = phys_bounds["LB"] or 0.0
    p_ub = phys_bounds["UB"]

    if n_segments == 1:
        clb = seg_lb if seg_lb > p_lb else p_lb
        if seg_ub is not None and (p_ub is None or seg_ub < p_ub):
            cub = seg_ub
        else:
            cub = p_ub if p_ub is not None else (seg_ub if seg_ub is not None else max_ub)
        return (
            round(max(clb, 0.0), BOUND_DECIMALS),
            round(max(cub, 0.0), BOUND_DECIMALS),
        )

    # Multi-segment: 3-branch LB logic matching Node.js stepCost.js
    if seg_lb > p_lb:
        clb = seg_lb
    elif (seg_ub or 0.0) <= p_lb:
        clb = seg_ub or 0.0
    else:
        clb = p_lb

    phys_bounds["LB"] = p_lb - clb

    if p_ub is None:
        cub = seg_ub if seg_ub is not None else max_ub
    else:
        if seg_ub is not None and seg_ub <= p_ub:
            cub = seg_ub
        else:
            cub = p_ub
        phys_bounds["UB"] = p_ub - cub

    return (
        round(max(clb, 0.0), BOUND_DECIMALS),
        round(max(cub, 0.0), BOUND_DECIMALS),
    )


# ---------------------------------------------------------------------------
# Evaporation
# ---------------------------------------------------------------------------

def _generate_annual_steps(
    start: str = DEFAULT_START,
    stop: str = DEFAULT_STOP,
) -> list[date]:
    """Generate September 30 end-dates for each complete water year in the range.

    Parameters
    ----------
    start : str
        Start date as YYYY-MM (e.g., '1921-10').
    stop : str
        Stop date as YYYY-MM (e.g., '2003-09').

    Returns
    -------
    list[date]
        One date per water year (Sep 30 of the ending calendar year).

    Examples
    --------
    >>> _generate_annual_steps('1921-10', '2003-09')  # 82 dates
    >>> _generate_annual_steps('1921-10', '1922-09')  # [date(1922, 9, 30)]
    """
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in stop.split("-"))

    # WY ending year N starts Oct(N-1). Include if Oct(N-1) >= (sy, sm).
    # Oct(N-1) >= (sy, sm) iff sm <= 10 and N-1 >= sy, else N-1 > sy.
    first_end_year = sy + 1 if sm <= 10 else sy + 2

    # WY ending year N ends Sep(N). Include if Sep(N) <= (ey, em).
    last_end_year = ey if em >= 9 else ey - 1

    return [date(year, 9, 30) for year in range(first_end_year, last_end_year + 1)]


def _water_year_months(sep_date: date) -> list[date]:
    """Return the 12 end-of-month dates for the water year ending on sep_date.

    WY ending Sep N: Oct(N-1) through Sep(N).
    """
    year = sep_date.year
    months = []
    for m in range(10, 13):
        months.append(_end_of_month(year - 1, m))
    for m in range(1, 10):
        months.append(_end_of_month(year, m))
    return months


def _aggregate_bounds_annual(
    bounds: list[Bound],
    monthly_steps: list[date],
    max_ub: float = DEFAULT_MAX_UB,
) -> StepBounds:
    """Aggregate monthly bounds to a single annual StepBounds by summing."""
    monthly = _resolve_bounds(bounds, monthly_steps)
    annual_lb = sum(sb.lb for sb in monthly)
    annual_ub = sum(sb.ub if sb.ub is not None else max_ub for sb in monthly)
    lb_defined = any(sb.lb_defined for sb in monthly)
    return StepBounds(lb=annual_lb, ub=annual_ub, lb_defined=lb_defined)


def _resolve_costs_annual(
    cost_def: CostDef | None,
    annual_bounds: StepBounds,
    monthly_steps: list[date],
    max_ub: float = DEFAULT_MAX_UB,
) -> list[CostSegment]:
    """Return annual cost segments by aggregating monthly penalty data.

    - NONE / None: single segment, cost=0, ub=None
    - Constant: single segment, cost=cost_def.cost, ub=None
    - Annual Variable: scale all capacity breakpoints × 12, then call _penalty_to_segments()
    - Monthly Variable: sum capacity breakpoints at each k index across all months;
      keep cumulative costs from the first month that has data.
    """
    if not cost_def or cost_def.type in ("NONE", "None"):
        return [CostSegment(cost=0.0, lb=0.0, ub=None)]

    if cost_def.type == "Constant":
        c = cost_def.cost or 0.0
        return [CostSegment(cost=c, lb=0.0, ub=None)]

    if cost_def.type == "Annual Variable":
        penalty_data = None
        if cost_def.costs:
            for key in cost_def.costs:
                penalty_data = cost_def.costs[key]
                break
        if penalty_data:
            scaled_data = []
            for row in penalty_data:
                if len(row) >= 2:
                    try:
                        scaled_data.append([float(row[0]) * 12, float(row[1])])
                    except (ValueError, TypeError):
                        scaled_data.append(row)  # keep header row
            return _penalty_to_segments(scaled_data, max_ub, annual_bounds)
        return [CostSegment(cost=0.0, lb=0.0, ub=None)]

    if cost_def.type == "Monthly Variable":
        # Collect parsed penalty points for each month that has data
        month_penalties: list[list[tuple[float, float]]] = []
        for step in monthly_steps:
            month_key = _month_name(step)
            penalty_data = None
            if cost_def.costs:
                penalty_data = cost_def.costs.get(month_key)
            if penalty_data:
                points: list[tuple[float, float]] = []
                for row in penalty_data:
                    if len(row) >= 2:
                        try:
                            points.append((float(row[0]), float(row[1])))
                        except (ValueError, TypeError):
                            continue
                if points:
                    month_penalties.append(points)

        if not month_penalties:
            return [CostSegment(cost=0.0, lb=0.0, ub=None)]

        # Filter out zero-demand months before aggregating.
        # Some months have a sentinel curve like [(0,0),(1e6,0)] indicating no
        # demand; all cumulative costs are 0.  Including these distorts min_len
        # (forcing it to 2) and inflates total_cap by up to 1,000,000 TAF,
        # which collapses the derived marginal cost to nearly zero.
        active_penalties = [p for p in month_penalties if any(cost != 0.0 for _, cost in p)]
        if not active_penalties:
            return [CostSegment(cost=0.0, lb=0.0, ub=None)]

        # Sum capacity breakpoints at each index using max_len with last-value
        # padding for shorter months.  Using min_len would truncate the
        # penalty curve at the level of the shortest active month, collapsing
        # the remaining demand tiers into a zero-cost extension that the
        # optimizer ignores.  Last-value padding is correct: a month's
        # cumulative penalty reaches zero at its final breakpoint (full demand
        # served), so all higher tiers contribute zero marginal capacity and
        # zero marginal cost.
        max_len = max(len(p) for p in active_penalties)
        summed_points = []
        for k_idx in range(max_len):
            total_cap = sum(
                p[k_idx][0] if k_idx < len(p) else p[-1][0]
                for p in active_penalties
            )
            total_cost = sum(
                p[k_idx][1] if k_idx < len(p) else p[-1][1]
                for p in active_penalties
            )
            summed_points.append([total_cap, total_cost])

        if len(summed_points) < 2:
            return [CostSegment(cost=0.0, lb=0.0, ub=None)]

        return _penalty_to_segments(summed_points, max_ub, annual_bounds)

    return [CostSegment(cost=0.0, lb=0.0, ub=None)]


def _compound_amplitude(
    evaporation: list[list] | None,
    areacapfactor: float,
    monthly_steps: list[date],
) -> float:
    """Compute annual storage carryover amplitude as the product of 12 monthly amplitudes."""
    if not evaporation or areacapfactor == 0.0:
        return 1.0
    result = 1.0
    for step in monthly_steps:
        result *= _get_evap_amplitude(evaporation, areacapfactor, step)
    return round(result, COST_AMP_DECIMALS)


def _get_evap_amplitude(
    evaporation: list[list] | None,
    areacapfactor: float,
    step: date,
) -> float:
    """Compute storage carry-over amplitude accounting for evaporation.

    amplitude = 1 - areacapfactor * evap_rate
    """
    if not evaporation or areacapfactor == 0.0:
        return 1.0

    evap_val = _lookup_timeseries_value(evaporation, step)
    if evap_val is None:
        return 1.0

    amp = 1.0 - areacapfactor * evap_val
    return round(amp, COST_AMP_DECIMALS)


# ---------------------------------------------------------------------------
# Row builder helpers
# ---------------------------------------------------------------------------

def _node_id(name: str, step: date, sep: str = ".") -> str:
    """Build a time-expanded node ID: NAME.YYYY-MM-DD"""
    return f"{name}{sep}{_date_str(step)}"


MatrixRow = tuple[str, str, int, float, float, float, float]


def _make_row(
    i: str, j: str, k: int,
    cost: float, amplitude: float,
    lb: float, ub: float,
) -> MatrixRow:
    return (
        i, j, k,
        round(cost, COST_AMP_DECIMALS),
        round(amplitude, COST_AMP_DECIMALS),
        round(lb, BOUND_DECIMALS),
        round(ub, BOUND_DECIMALS),
    )


# ---------------------------------------------------------------------------
# Node processing
# ---------------------------------------------------------------------------

def _process_node(
    node: NetworkNode,
    steps: list[date],
    network: Network,
    sep: str,
    max_ub: float,
    constrain_ending: str = "none",
    lb_override: float | None = None,
) -> list[MatrixRow]:
    """Generate matrix rows for a single node."""
    rows: list[MatrixRow] = []

    # Inflow rows: SOURCE -> node
    if node.inflows:
        for inflow_name, inflow_data in node.inflows.items():
            if not inflow_data:
                continue
            for step in steps:
                val = _lookup_timeseries_value(inflow_data, step)
                if val is None:
                    val = 0.0
                rows.append(_make_row(
                    f"INFLOW{sep}{_date_str(step)}",
                    _node_id(node.prmname, step, sep),
                    0, 0.0, 1.0, val, val,
                ))

    # Sink rows: node -> SINK
    if node.sinks:
        for s_idx, sink in enumerate(node.sinks):
            sink_bounds = _resolve_bounds(sink.get("bounds", []), steps)
            sink_costs = _resolve_costs(
                sink.get("costs"), sink_bounds, steps, max_ub
            )
            is_eq = any(
                b.type.startswith("EQ")
                for b in sink.get("bounds", [])
                if isinstance(b, Bound)
            )

            for i, step in enumerate(steps):
                sb = sink_bounds[i]
                cost_segs = sink_costs[i]
                phys = {"LB": sb.lb, "UB": sb.ub}

                for k, seg in enumerate(cost_segs):
                    clb, cub = _reconcile_step_cost(
                        seg, phys, len(cost_segs), max_ub
                    )
                    if is_eq:
                        clb = cub
                    rows.append(_make_row(
                        _node_id(node.prmname, step, sep),
                        f"SINK{sep}{_date_str(step)}",
                        k, seg.cost, 1.0, clb, cub,
                    ))

    # Storage rows: node@t -> node@t+1 (with evaporation)
    if node.initialstorage is not None:
        rows.extend(_process_storage(node, steps, sep, max_ub, constrain_ending, lb_override))

    return rows


def _process_storage(
    node: NetworkNode,
    steps: list[date],
    sep: str,
    max_ub: float,
    constrain_ending: str = "none",
    lb_override: float | None = None,
) -> list[MatrixRow]:
    """Generate storage linking rows for a reservoir node."""
    rows: list[MatrixRow] = []

    # Adjust initial/ending storage based on time-range coverage of storage
    # capacity data, matching Node.js storage.js logic.
    init_val = node.initialstorage or 0.0
    end_val = node.endingstorage
    cap = node.storage
    if cap and len(cap) > 1:
        step_strs = {_date_str(s) for s in steps}
        first_idx = None
        last_idx = None
        for idx in range(1, len(cap)):  # skip header row
            row_date = str(cap[idx][0])
            if row_date in step_strs:
                if first_idx is None:
                    if idx > 1:
                        try:
                            init_val = float(cap[idx - 1][1])
                        except (ValueError, TypeError):
                            pass
                    first_idx = idx
                last_idx = idx
        if last_idx is not None and last_idx != len(cap) - 1:
            try:
                end_val = float(cap[last_idx][1])
            except (ValueError, TypeError):
                pass

    # INITIAL -> node@first_step
    rows.append(_make_row(
        "INITIAL",
        _node_id(node.prmname, steps[0], sep),
        0, 0.0, 1.0, init_val, init_val,
    ))

    # Storage bounds and costs
    storage_bounds = _resolve_bounds(node.bounds, steps)
    if lb_override is not None:
        storage_bounds = [
            StepBounds(lb=min(sb.lb, lb_override), ub=sb.ub, lb_defined=sb.lb_defined)
            for sb in storage_bounds
        ]
    storage_costs = _resolve_costs(node.costs, storage_bounds, steps, max_ub)

    # Carry-over links: node@t -> node@t+1
    for i in range(len(steps) - 1):
        step = steps[i]
        next_step = steps[i + 1]
        amplitude = _get_evap_amplitude(node.evaporation, node.areacapfactor, step)
        sb = storage_bounds[i]
        cost_segs = storage_costs[i]

        phys = {"LB": sb.lb, "UB": sb.ub}
        for k, seg in enumerate(cost_segs):
            clb, cub = _reconcile_step_cost(
                seg, phys, len(cost_segs), max_ub
            )
            rows.append(_make_row(
                _node_id(node.prmname, step, sep),
                _node_id(node.prmname, next_step, sep),
                k, seg.cost, amplitude, clb, cub,
            ))

    # node@last_step -> FINAL (using time-range-adjusted end_val from above)
    #
    # constrain_ending controls ending-storage treatment:
    #   "none"  — default: GW unconstrained (lb=0, ub=inf), SR fixed to end_val
    #   "gw"    — GW nodes lb=initial storage, ub=unconstrained (no overdraft, recovery allowed); SR unchanged
    #   "all"   — all storage nodes fixed to end_val (COSVF one-year assumption)
    is_gw = node.node_type == "Groundwater Storage"

    if constrain_ending == "all":
        final_val = end_val if end_val is not None else init_val
        rows.append(_make_row(
            _node_id(node.prmname, steps[-1], sep),
            "FINAL",
            0, 0.0, 1.0, final_val, final_val,
        ))
    elif constrain_ending == "gw" and is_gw:
        # Constrain GW ending storage >= initial storage (no net overdraft).
        # Upper bound remains unconstrained so the aquifer can recover above initial.
        rows.append(_make_row(
            _node_id(node.prmname, steps[-1], sep),
            "FINAL",
            0, 0.0, 1.0, init_val, max_ub,
        ))
    elif is_gw or end_val is None:
        rows.append(_make_row(
            _node_id(node.prmname, steps[-1], sep),
            "FINAL",
            0, 0.0, 1.0, 0.0, max_ub,
        ))
    else:
        rows.append(_make_row(
            _node_id(node.prmname, steps[-1], sep),
            "FINAL",
            0, 0.0, 1.0, end_val, end_val,
        ))

    return rows


# ---------------------------------------------------------------------------
# Link processing
# ---------------------------------------------------------------------------

def _process_link(
    link: NetworkLink,
    steps: list[date],
    network: Network,
    sep: str,
    max_ub: float,
) -> list[MatrixRow]:
    """Generate matrix rows for a single link."""
    rows: list[MatrixRow] = []

    link_bounds = _resolve_bounds(link.bounds, steps)
    link_costs = _resolve_costs(link.costs, link_bounds, steps, max_ub)
    is_eq = any(b.is_equality for b in link.bounds)

    for i, step in enumerate(steps):
        sb = link_bounds[i]
        cost_segs = link_costs[i]
        phys = {"LB": sb.lb, "UB": sb.ub}

        for k, seg in enumerate(cost_segs):
            clb, cub = _reconcile_step_cost(
                seg, phys, len(cost_segs), max_ub
            )
            if is_eq:
                clb = cub

            rows.append(_make_row(
                _node_id(link.origin, step, sep),
                _node_id(link.terminus, step, sep),
                k, seg.cost, link.amplitude,
                clb, cub,
            ))

    return rows


# ---------------------------------------------------------------------------
# Annual-step node/link/storage processing
# ---------------------------------------------------------------------------

def _process_storage_annual(
    node: NetworkNode,
    annual_steps: list[date],
    sep: str,
    max_ub: float,
    constrain_ending: str = "none",
    lb_override: float | None = None,
) -> list[MatrixRow]:
    """Generate storage linking rows for a reservoir node (annual-step version)."""
    rows: list[MatrixRow] = []

    # Adjust initial/ending storage based on time-range coverage of storage
    # capacity data — same logic as _process_storage().
    init_val = node.initialstorage or 0.0
    end_val = node.endingstorage
    cap = node.storage
    if cap and len(cap) > 1:
        step_strs = {_date_str(s) for s in annual_steps}
        first_idx = None
        last_idx = None
        for idx in range(1, len(cap)):
            row_date = str(cap[idx][0])
            if row_date in step_strs:
                if first_idx is None:
                    if idx > 1:
                        try:
                            init_val = float(cap[idx - 1][1])
                        except (ValueError, TypeError):
                            pass
                    first_idx = idx
                last_idx = idx
        if last_idx is not None and last_idx != len(cap) - 1:
            try:
                end_val = float(cap[last_idx][1])
            except (ValueError, TypeError):
                pass

    # INITIAL -> node@first_annual_step: lb=ub=init_val
    rows.append(_make_row(
        "INITIAL",
        _node_id(node.prmname, annual_steps[0], sep),
        0, 0.0, 1.0, init_val, init_val,
    ))

    # Carryover links: node@sep_n -> node@sep_{n+1}
    for i in range(len(annual_steps) - 1):
        origin_step = annual_steps[i]
        dest_step = annual_steps[i + 1]

        # Monthly steps for the WY ending at the destination
        monthly_steps = _water_year_months(dest_step)
        amplitude = _compound_amplitude(node.evaporation, node.areacapfactor, monthly_steps)

        # Storage bounds at origin (September value only)
        storage_bounds_at_origin = _resolve_bounds(node.bounds, [origin_step])
        sb = storage_bounds_at_origin[0]
        if lb_override is not None:
            sb = StepBounds(lb=min(sb.lb, lb_override), ub=sb.ub, lb_defined=sb.lb_defined)
            storage_bounds_at_origin = [sb]
        storage_costs_at_origin = _resolve_costs(node.costs, storage_bounds_at_origin, [origin_step], max_ub)
        cost_segs = storage_costs_at_origin[0]

        phys = {"LB": sb.lb, "UB": sb.ub}
        for k, seg in enumerate(cost_segs):
            clb, cub = _reconcile_step_cost(seg, phys, len(cost_segs), max_ub)
            rows.append(_make_row(
                _node_id(node.prmname, origin_step, sep),
                _node_id(node.prmname, dest_step, sep),
                k, seg.cost, amplitude, clb, cub,
            ))

    # node@last_annual_step -> FINAL
    is_gw = node.node_type == "Groundwater Storage"

    if constrain_ending == "all":
        final_val = end_val if end_val is not None else init_val
        rows.append(_make_row(
            _node_id(node.prmname, annual_steps[-1], sep),
            "FINAL", 0, 0.0, 1.0, final_val, final_val,
        ))
    elif constrain_ending == "gw" and is_gw:
        # Constrain GW ending storage >= initial storage (no net overdraft).
        # Upper bound remains unconstrained so the aquifer can recover above initial.
        rows.append(_make_row(
            _node_id(node.prmname, annual_steps[-1], sep),
            "FINAL", 0, 0.0, 1.0, init_val, max_ub,
        ))
    elif is_gw or end_val is None:
        rows.append(_make_row(
            _node_id(node.prmname, annual_steps[-1], sep),
            "FINAL", 0, 0.0, 1.0, 0.0, max_ub,
        ))
    else:
        rows.append(_make_row(
            _node_id(node.prmname, annual_steps[-1], sep),
            "FINAL", 0, 0.0, 1.0, end_val, end_val,
        ))

    return rows


def _process_node_annual(
    node: NetworkNode,
    annual_steps: list[date],
    network: Network,
    sep: str,
    max_ub: float,
    constrain_ending: str = "none",
    lb_override: float | None = None,
) -> list[MatrixRow]:
    """Generate matrix rows for a single node (annual-step version)."""
    rows: list[MatrixRow] = []

    # Inflow rows: INFLOW.{sep30} -> node.{sep30}
    if node.inflows:
        for _inflow_name, inflow_data in node.inflows.items():
            if not inflow_data:
                continue
            for annual_step in annual_steps:
                monthly_steps = _water_year_months(annual_step)
                annual_val = sum(
                    _lookup_timeseries_value(inflow_data, s) or 0.0
                    for s in monthly_steps
                )
                rows.append(_make_row(
                    f"INFLOW{sep}{_date_str(annual_step)}",
                    _node_id(node.prmname, annual_step, sep),
                    0, 0.0, 1.0, annual_val, annual_val,
                ))

    # Sink rows: node.{sep30} -> SINK.{sep30}
    if node.sinks:
        for sink in node.sinks:
            sink_bounds_raw = sink.get("bounds", [])
            is_eq = any(
                b.type.startswith("EQ")
                for b in sink_bounds_raw
                if isinstance(b, Bound)
            )
            for annual_step in annual_steps:
                monthly_steps = _water_year_months(annual_step)
                annual_bounds = _aggregate_bounds_annual(sink_bounds_raw, monthly_steps, max_ub)
                annual_costs = _resolve_costs_annual(sink.get("costs"), annual_bounds, monthly_steps, max_ub)

                phys = {"LB": annual_bounds.lb, "UB": annual_bounds.ub}
                for k, seg in enumerate(annual_costs):
                    clb, cub = _reconcile_step_cost(seg, phys, len(annual_costs), max_ub)
                    if is_eq:
                        clb = cub
                    rows.append(_make_row(
                        _node_id(node.prmname, annual_step, sep),
                        f"SINK{sep}{_date_str(annual_step)}",
                        k, seg.cost, 1.0, clb, cub,
                    ))

    # Storage rows
    if node.initialstorage is not None:
        rows.extend(_process_storage_annual(node, annual_steps, sep, max_ub, constrain_ending, lb_override))

    return rows


def _process_link_annual(
    link: NetworkLink,
    annual_steps: list[date],
    network: Network,
    sep: str,
    max_ub: float,
) -> list[MatrixRow]:
    """Generate matrix rows for a single link (annual-step version)."""
    rows: list[MatrixRow] = []
    is_eq = any(b.is_equality for b in link.bounds)

    for annual_step in annual_steps:
        monthly_steps = _water_year_months(annual_step)
        annual_bounds = _aggregate_bounds_annual(link.bounds, monthly_steps, max_ub)
        annual_costs = _resolve_costs_annual(link.costs, annual_bounds, monthly_steps, max_ub)

        phys = {"LB": annual_bounds.lb, "UB": annual_bounds.ub}
        for k, seg in enumerate(annual_costs):
            clb, cub = _reconcile_step_cost(seg, phys, len(annual_costs), max_ub)
            if is_eq:
                clb = cub
            rows.append(_make_row(
                _node_id(link.origin, annual_step, sep),
                _node_id(link.terminus, annual_step, sep),
                k, seg.cost, link.amplitude,
                clb, cub,
            ))

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_matrix(
    network: Network,
    start: str = DEFAULT_START,
    stop: str = DEFAULT_STOP,
    max_ub: float = DEFAULT_MAX_UB,
    sep: str = ".",
    add_debug: bool = False,
    debug_cost: float = 2e7,
    constrain_ending: str = "none",
    node_lb_overrides: dict | None = None,
) -> pd.DataFrame:
    """Build the time-expanded network matrix from a Network object.

    Parameters
    ----------
    network : Network
        Loaded network from ``load_network()``.
    start : str
        Start date as YYYY-MM (default: 1921-10).
    stop : str
        Stop date as YYYY-MM (default: 2003-09).
    max_ub : float
        Replace unbounded upper bounds with this value.
    sep : str
        Separator between node name and date (default: ".").
    add_debug : bool
        If True, add debug source/sink links for infeasibility diagnosis.
    debug_cost : float
        Cost assigned to debug links.
    constrain_ending : str
        Controls ending-storage constraints on node→FINAL links:

        - ``"none"`` (default) — GW nodes unconstrained; SR nodes fixed to
          their ``endingstorage`` value if defined.
        - ``"gw"`` — GW nodes: lb=initial storage, ub=unconstrained (no net
          overdraft, but aquifer recovery above initial is allowed).
          SR nodes unchanged.
        - ``"all"`` — All storage nodes (GW + SR) fixed to ending/initial
          storage (lb=ub).  Used by COSVF one-year runs.

    node_lb_overrides : dict, optional
        Mapping of node name → physical lower bound.  For each entry, the
        storage lower bound is clamped to this value before piecewise cost
        reconciliation, overriding any higher LBT-derived bound.  Used to
        allow drought drawdown to the physical dead-pool level for nodes
        whose LBT data reflects an operational target (e.g. SR_SHA, SR_CLE).

    Returns
    -------
    pd.DataFrame
        Matrix with columns: i, j, k, cost, amplitude, lower_bound, upper_bound.
    """
    if constrain_ending not in ("none", "gw", "all"):
        raise ValueError(
            f"constrain_ending must be 'none', 'gw', or 'all', got {constrain_ending!r}"
        )
    steps = _generate_steps(start, stop)
    all_rows: list[MatrixRow] = []

    # Process nodes (sorted by prmname for deterministic output)
    for prmname in sorted(network.nodes.keys()):
        node = network.nodes[prmname]
        lb_override = node_lb_overrides.get(prmname) if node_lb_overrides else None
        node_rows = _process_node(node, steps, network, sep, max_ub, constrain_ending, lb_override)
        all_rows.extend(node_rows)

    # Process links (sorted by prmname)
    for link in sorted(network.links, key=lambda l: l.prmname):
        link_rows = _process_link(link, steps, network, sep, max_ub)
        all_rows.extend(link_rows)

    # Add SOURCE -> INITIAL and FINAL -> SINK connectors
    all_rows.append(_make_row("SOURCE", "INITIAL", 0, 0.0, 1.0, 0.0, AGGREGATE_MAX_UB))
    all_rows.append(_make_row("FINAL", "SINK", 0, 0.0, 1.0, 0.0, AGGREGATE_MAX_UB))

    # Add SOURCE -> INFLOW connectors (super-source)
    # Also collect SINK.date and OUTBOUND.date nodes that need → SINK arcs
    seen_inbound: set[str] = set()
    seen_outbound: set[str] = set()

    for row in all_rows:
        i_node, j_node = row[0], row[1]
        # SOURCE -> INFLOW.date / INBOUND.date (one per unique name)
        for prefix in ("INFLOW", "INBOUND"):
            if i_node.startswith(prefix) and i_node not in seen_inbound:
                all_rows.append(_make_row("SOURCE", i_node, 0, 0.0, 1.0, 0.0, max_ub))
                seen_inbound.add(i_node)
        # SINK.date -> SINK / OUTBOUND.date -> SINK (one per unique name)
        for prefix in ("SINK", "OUTBOUND"):
            if j_node.startswith(prefix) and j_node != "SINK" and j_node not in seen_outbound:
                all_rows.append(_make_row(j_node, "SINK", 0, 0.0, 1.0, 0.0, max_ub))
                seen_outbound.add(j_node)

    # Add debug links if requested
    if add_debug:
        all_node_ids = set()
        for row in all_rows:
            all_node_ids.add(row[0])
            all_node_ids.add(row[1])

        for nid in sorted(all_node_ids):
            if nid in ("SOURCE", "SINK", "INITIAL", "FINAL"):
                continue
            # Debug source -> node
            all_rows.append(_make_row(
                "DBUGSRC", nid, 0, debug_cost, 1.0, 0.0, max_ub
            ))
            # Node -> debug sink
            all_rows.append(_make_row(
                nid, "DBUGSNK", 0, debug_cost, 1.0, 0.0, max_ub
            ))

        # Connector links so DBUGSRC/DBUGSNK satisfy flow conservation:
        # SOURCE feeds DBUGSRC; DBUGSNK drains to SINK.
        all_rows.append(_make_row(
            "SOURCE", "DBUGSRC", 0, 0.0, 1.0, 0.0, max_ub
        ))
        all_rows.append(_make_row(
            "DBUGSNK", "SINK", 0, 0.0, 1.0, 0.0, max_ub
        ))

    return pd.DataFrame(all_rows, columns=["i", "j", "k", "cost", "amplitude",
                                           "lower_bound", "upper_bound"])


def build_annual_matrix(
    network: Network,
    start: str = DEFAULT_START,
    stop: str = DEFAULT_STOP,
    max_ub: float = DEFAULT_MAX_UB,
    sep: str = ".",
    add_debug: bool = False,
    debug_cost: float = 2e7,
    constrain_ending: str = "none",
    node_lb_overrides: dict | None = None,
) -> pd.DataFrame:
    """Build the annual time-expanded network matrix from a Network object.

    Each timestep corresponds to one water year, identified by its September 30
    end-date.  Flows are aggregated totals (TAF/year); storage is tracked at
    end-of-September with compound evaporation across the 12 monthly steps.

    Parameters
    ----------
    network : Network
        Loaded network from ``load_network()``.
    start : str
        Start date as YYYY-MM (default: 1921-10 → first WY ending 1922-09-30).
    stop : str
        Stop date as YYYY-MM (default: 2003-09 → last WY ending 2003-09-30).
    max_ub : float
        Replace unbounded upper bounds with this value.
    sep : str
        Separator between node name and date (default: ".").
    add_debug : bool
        If True, add debug source/sink links for infeasibility diagnosis.
    debug_cost : float
        Cost assigned to debug links.
    constrain_ending : str
        Controls ending-storage constraints on node→FINAL links.
        Same semantics as ``build_matrix()``.
    node_lb_overrides : dict, optional
        Same semantics as ``build_matrix()``.

    Returns
    -------
    pd.DataFrame
        Matrix with columns: i, j, k, cost, amplitude, lower_bound, upper_bound.
    """
    if constrain_ending not in ("none", "gw", "all"):
        raise ValueError(
            f"constrain_ending must be 'none', 'gw', or 'all', got {constrain_ending!r}"
        )
    annual_steps = _generate_annual_steps(start, stop)
    all_rows: list[MatrixRow] = []

    # Process nodes (sorted by prmname for deterministic output)
    for prmname in sorted(network.nodes.keys()):
        node = network.nodes[prmname]
        lb_override = node_lb_overrides.get(prmname) if node_lb_overrides else None
        node_rows = _process_node_annual(node, annual_steps, network, sep, max_ub, constrain_ending, lb_override)
        all_rows.extend(node_rows)

    # Process links (sorted by prmname)
    for link in sorted(network.links, key=lambda l: l.prmname):
        link_rows = _process_link_annual(link, annual_steps, network, sep, max_ub)
        all_rows.extend(link_rows)

    # Add SOURCE -> INITIAL and FINAL -> SINK connectors
    all_rows.append(_make_row("SOURCE", "INITIAL", 0, 0.0, 1.0, 0.0, AGGREGATE_MAX_UB))
    all_rows.append(_make_row("FINAL", "SINK", 0, 0.0, 1.0, 0.0, AGGREGATE_MAX_UB))

    # Add SOURCE -> INFLOW connectors and per-step SINK.date -> SINK connectors
    seen_inbound: set[str] = set()
    seen_outbound: set[str] = set()

    for row in all_rows:
        i_node, j_node = row[0], row[1]
        for prefix in ("INFLOW", "INBOUND"):
            if i_node.startswith(prefix) and i_node not in seen_inbound:
                all_rows.append(_make_row("SOURCE", i_node, 0, 0.0, 1.0, 0.0, max_ub))
                seen_inbound.add(i_node)
        for prefix in ("SINK", "OUTBOUND"):
            if j_node.startswith(prefix) and j_node != "SINK" and j_node not in seen_outbound:
                all_rows.append(_make_row(j_node, "SINK", 0, 0.0, 1.0, 0.0, max_ub))
                seen_outbound.add(j_node)

    # Add debug links if requested
    if add_debug:
        all_node_ids: set[str] = set()
        for row in all_rows:
            all_node_ids.add(row[0])
            all_node_ids.add(row[1])

        for nid in sorted(all_node_ids):
            if nid in ("SOURCE", "SINK", "INITIAL", "FINAL"):
                continue
            all_rows.append(_make_row("DBUGSRC", nid, 0, debug_cost, 1.0, 0.0, max_ub))
            all_rows.append(_make_row(nid, "DBUGSNK", 0, debug_cost, 1.0, 0.0, max_ub))

        all_rows.append(_make_row("SOURCE", "DBUGSRC", 0, 0.0, 1.0, 0.0, max_ub))
        all_rows.append(_make_row("DBUGSNK", "SINK", 0, 0.0, 1.0, 0.0, max_ub))

    return pd.DataFrame(all_rows, columns=["i", "j", "k", "cost", "amplitude",
                                           "lower_bound", "upper_bound"])


def export_matrix(
    df: pd.DataFrame,
    output: str | Path | None = None,
    fmt: str = "csv",
) -> str | None:
    """Export a matrix DataFrame to file or string.

    Parameters
    ----------
    df : pd.DataFrame
        Matrix from ``build_matrix()``.
    output : str, Path, or None
        Output file path. If None, returns the formatted string.
    fmt : str
        Output format: "csv" or "tsv".

    Returns
    -------
    str or None
        Formatted matrix string if output is None.
    """
    if fmt == "tsv":
        sep = "\t"
    else:
        sep = ","

    if output:
        df.to_csv(output, sep=sep, index=False)
        logger.info("Wrote matrix to %s (%d rows)", output, len(df))
        return None
    else:
        return df.to_csv(sep=sep, index=False)
