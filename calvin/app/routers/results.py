"""
Results API router — serves model run outputs (flows, shortages, storage).
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from calvin.app.schemas import NodeResultSeries, StudyInfo, StudyListResponse, SummaryResponse
from calvin.app.state import AppState, Study, get_state

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_series(df: pd.DataFrame, col: str) -> list[list]:
    """Extract a column from a DataFrame as [[date_str, value], ...] pairs."""
    if col not in df.columns:
        return []
    s = df[col].dropna()
    return [[str(idx), float(val)] for idx, val in s.items()]


def _study_info(study: Study, active_name: str | None) -> StudyInfo:
    return StudyInfo(
        name=study.name,
        path=str(study.path),
        available=study.available,
        active=(study.name == active_name),
    )


def _resolve_study(state: AppState, study: str | None) -> Study:
    s = state.get_study(study)
    if s is None:
        label = f"'{study}'" if study else "active study"
        raise HTTPException(status_code=404, detail=f"Study {label} not found")
    return s


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/studies", response_model=StudyListResponse, summary="List available studies")
def list_studies(state: AppState = Depends(get_state)) -> StudyListResponse:
    """Return metadata for all loaded studies."""
    return StudyListResponse(
        studies=[_study_info(s, state.active_study) for s in state.studies.values()],
        active=state.active_study,
    )


@router.get("/summary", response_model=SummaryResponse, summary="Annual aggregate summary")
def get_summary(
    study: str | None = Query(default=None, description="Study name; defaults to active"),
    study2: str | None = Query(default=None, description="Reserved for future comparison"),
    state: AppState = Depends(get_state),
) -> SummaryResponse:
    """Return annual totals of shortage volume, shortage cost, and operation cost."""
    s = _resolve_study(state, study)

    def _annual_total(key: str) -> list[float]:
        df = s.results.get(key)
        if df is None:
            return []
        totals = df.sum(axis=1)
        annual = totals.resample("YE-SEP").sum()
        return [float(v) for v in annual]

    def _annual_years(key: str) -> list[str]:
        df = s.results.get(key)
        if df is None:
            return []
        totals = df.sum(axis=1)
        annual = totals.resample("YE-SEP").sum()
        return [str(idx.year) for idx in annual.index]

    # Use shortage_volume as the reference for year labels
    ref_key = "shortage_volume" if "shortage_volume" in s.results else next(iter(s.results), None)
    years = _annual_years(ref_key) if ref_key else []

    return SummaryResponse(
        study=s.name,
        years=years,
        total_shortage_volume=_annual_total("shortage_volume"),
        total_shortage_cost=_annual_total("shortage_cost"),
        total_operation_cost=_annual_total("operation_costs"),
    )


@router.get("/node/{prmname}", response_model=NodeResultSeries, summary="Node result timeseries")
def get_node_results(
    prmname: str,
    study: str | None = Query(default=None, description="Study name; defaults to active"),
    study2: str | None = Query(default=None, description="Reserved for future comparison"),
    state: AppState = Depends(get_state),
) -> NodeResultSeries:
    """
    Return all available result timeseries for a single node.

    - **Flow**: outflow and inflow columns from flow.csv matching this node
    - **Storage**: storage column from storage.csv
    - **Shortage volume / cost**: matching columns from shortage CSVs
    - **Operation costs**: matching columns from operation_costs.csv
    """
    s = _resolve_study(state, study)
    series: dict[str, list[list]] = {}

    # Storage — exact column match
    if "storage" in s.results:
        df = s.results["storage"]
        if prmname in df.columns:
            series["storage"] = _df_to_series(df, prmname)

    # Evaporation — exact column match
    if "evaporation" in s.results:
        df = s.results["evaporation"]
        if prmname in df.columns:
            series["evaporation"] = _df_to_series(df, prmname)

    # Flow — outflows ({prmname}-*) and inflows (*-{prmname})
    if "flow" in s.results:
        df = s.results["flow"]
        prefix = f"{prmname}-"
        suffix = f"-{prmname}"
        for col in df.columns:
            if col.startswith(prefix):
                series[f"flow_out_{col}"] = _df_to_series(df, col)
            elif col.endswith(suffix):
                series[f"flow_in_{col}"] = _df_to_series(df, col)

    # Demand shortage: column is {supply}-{demand_node}; shortage belongs to the terminus (demand node)
    for result_key in ("shortage_volume", "shortage_cost"):
        if result_key not in s.results:
            continue
        df = s.results[result_key]
        suffix = f"-{prmname}"
        for col in df.columns:
            if col.endswith(suffix):
                series[f"{result_key}_{col}"] = _df_to_series(df, col)

    # PWP shortage: column is {pwp_node}-{downstream}; shortage belongs to the origin (PWP node)
    for result_key in ("pwp_short_volume", "pwp_short_cost"):
        if result_key not in s.results:
            continue
        df = s.results[result_key]
        prefix = f"{prmname}-"
        for col in df.columns:
            if col.startswith(prefix):
                series[f"{result_key}_{col}"] = _df_to_series(df, col)

    # Operation costs — columns are link names {origin}-{terminus}
    if "operation_costs" in s.results:
        df = s.results["operation_costs"]
        prefix = f"{prmname}-"
        suffix = f"-{prmname}"
        for col in df.columns:
            if col.startswith(prefix) or col.endswith(suffix):
                series[f"op_cost_{col}"] = _df_to_series(df, col)

    # Dual node — one column per node (shadow price of mass balance constraint)
    if "dual_node" in s.results:
        df = s.results["dual_node"]
        if prmname in df.columns:
            series[f"dual_node_{prmname}"] = _df_to_series(df, prmname)

    # Dual lower / upper — storage carryover arcs use bare prmname; regular links use {origin}-{terminus}
    for dual_key in ("dual_lower", "dual_upper"):
        if dual_key not in s.results:
            continue
        df = s.results[dual_key]
        # Exact match: storage node carryover arc stored under bare prmname
        if prmname in df.columns:
            series[f"{dual_key}_{prmname}"] = _df_to_series(df, prmname)
        prefix = f"{prmname}-"
        suffix = f"-{prmname}"
        for col in df.columns:
            if col.startswith(prefix) or col.endswith(suffix):
                series[f"{dual_key}_{col}"] = _df_to_series(df, col)

    return NodeResultSeries(prmname=prmname, study=s.name, series=series)


@router.get("/shortage-nodes", summary="Per-column total shortage volume for map highlighting")
def get_shortage_nodes(
    study: str | None = Query(default=None, description="Study name; defaults to active"),
    state: AppState = Depends(get_state),
) -> dict:
    """Return total shortage per column, split by type so the frontend can apply correct attribution.

    - shortage_cols: demand shortages ({supply}-{demand_node}); match terminus to find the node
    - pwp_cols: PWP shortages ({pwp_node}-{downstream}); match origin to find the node
    """
    s = _resolve_study(state, study)

    def _col_totals(key: str) -> dict[str, float]:
        if key not in s.results:
            return {}
        df = s.results[key]
        return {str(col): float(v) for col, v in df.sum(axis=0).items() if float(v) > 0}

    return {
        "study": s.name,
        "shortage_cols": _col_totals("shortage_volume"),
        "pwp_cols": _col_totals("pwp_short_volume"),
    }


@router.get("/debug-links", summary="Debug source/sink link flows for infeasibility diagnosis")
def get_debug_links(
    study: str | None = Query(default=None, description="Study name; defaults to active"),
    state: AppState = Depends(get_state),
) -> dict:
    """Return debug source/sink links with non-zero total flow.

    Debug links (DBUGSRC→node, node→DBUGSNK) carry flow only when the optimizer
    cannot satisfy mass-balance at a node without artificial injection or extraction.

    - active_links: link prmnames (e.g. "DBUGSRC-SR_SHA") with non-zero total flow
    - active_nodes: real node prmnames at the non-debug end of those links
    """
    s = _resolve_study(state, study)

    if "flow" not in s.results:
        return {"study": s.name, "active_links": [], "active_nodes": []}

    df = s.results["flow"]
    debug_cols = [col for col in df.columns if "DBUGSRC" in col or "DBUGSNK" in col]

    active_links: list[str] = []
    active_nodes: set[str] = set()
    for col in debug_cols:
        if df[col].abs().sum() > 0:
            active_links.append(str(col))
            if col.startswith("DBUGSRC-"):
                active_nodes.add(col[len("DBUGSRC-"):])
            elif col.endswith("-DBUGSNK"):
                active_nodes.add(col[: -len("-DBUGSNK")])

    return {
        "study": s.name,
        "active_links": active_links,
        "active_nodes": sorted(active_nodes),
    }


@router.get("/cosvf/{prmname}", summary="COSVF penalty curve data for a storage node")
def get_cosvf(
    prmname: str,
    study: str | None = Query(default=None, description="Study name; defaults to active"),
    state: AppState = Depends(get_state),
) -> dict:
    """
    Return COSVF penalty curve parameters and metadata for a storage node.

    Only available for studies prepared with `prepare_cosvf` or `prepare_cosvf_astep`
    (i.e., those that include a `r-dict.json` and `results/cosvf-params.csv`).

    - **type**: 1 = surface reservoir (quadratic, pmin/pmax), 2 = groundwater (linear, p)
    - **params**: EA-optimized penalty parameters from results/cosvf-params.csv
    - **lb / ub**: Sep-30 storage bounds from r-dict.json (TAF)
    """
    s = _resolve_study(state, study)
    r_info = s.r_dict.get(prmname)
    params = s.cosvf.get(prmname)

    if not r_info and not params:
        raise HTTPException(status_code=404, detail=f"No COSVF data for '{prmname}'")

    return {
        "prmname": prmname,
        "study": s.name,
        "type": r_info.get("type") if r_info else None,
        "lb": r_info.get("lb") if r_info else None,
        "ub": r_info.get("ub") if r_info else None,
        "eop_init": r_info.get("eop_init") if r_info else None,
        "k_count": r_info.get("k_count") if r_info else None,
        "params": params or {},
    }


@router.get("/timeslice", summary="All node values at a specific date")
def get_timeslice(
    date: str = Query(..., description="Date string, e.g. 1975-09-30"),
    study: str | None = Query(default=None, description="Study name; defaults to active"),
    study2: str | None = Query(default=None, description="Reserved for future comparison"),
    state: AppState = Depends(get_state),
) -> dict:
    """
    Return all result values at a specific date across all result types.
    Used by the frontend to color the map at a given timestep.
    """
    s = _resolve_study(state, study)
    result: dict[str, dict[str, float]] = {}

    for key, df in s.results.items():
        try:
            row = df.loc[date]
            result[key] = {col: float(val) for col, val in row.items() if pd.notna(val)}
        except KeyError:
            # Fuzzy match: find closest date
            try:
                idx = df.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
                row = df.iloc[idx]
                result[key] = {col: float(val) for col, val in row.items() if pd.notna(val)}
            except Exception as e:
                logger.debug("Timeslice miss for %s at %s: %s", key, date, e)

    return {"date": date, "study": s.name, "data": result}
