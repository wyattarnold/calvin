"""
Pydantic response models for the Calvin Network App API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Network — node / link summaries
# ---------------------------------------------------------------------------

class NodeSummary(BaseModel):
    prmname: str
    description: str | None
    node_type: str | None
    disabled: bool


class LinkSummary(BaseModel):
    prmname: str
    description: str | None
    link_type: str | None
    origin: str
    terminus: str
    amplitude: float
    disabled: bool


# ---------------------------------------------------------------------------
# Network — node detail
# ---------------------------------------------------------------------------

class BoundDetail(BaseModel):
    type: str
    bound: Any  # float | dict | list[list]


class CostDefDetail(BaseModel):
    type: str
    cost: float | None = None
    costs: dict[str, Any] | None = None  # month/year -> breakpoints


class NodeDetail(BaseModel):
    prmname: str
    description: str | None
    node_type: str | None
    disabled: bool
    filepath: str
    initialstorage: float | None
    endingstorage: float | None
    areacapfactor: float
    inflows: dict[str, list[list]] | None
    sinks: list[dict] | None
    flow: list[list] | None
    storage: list[list] | None
    evaporation: list[list] | None
    bounds: list[BoundDetail]
    costs: CostDefDetail | None
    properties: dict[str, Any]


# ---------------------------------------------------------------------------
# Network — link detail
# ---------------------------------------------------------------------------

class LinkDetail(BaseModel):
    prmname: str
    description: str | None
    link_type: str | None
    origin: str
    terminus: str
    amplitude: float
    disabled: bool
    filepath: str
    flow: list[list] | None
    bounds: list[BoundDetail]
    costs: CostDefDetail | None
    properties: dict[str, Any]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class StudyInfo(BaseModel):
    name: str
    path: str
    available: list[str]  # e.g. ["flow", "storage", "shortage_volume"]
    active: bool


class StudyListResponse(BaseModel):
    studies: list[StudyInfo]
    active: str | None


class NodeResultSeries(BaseModel):
    """All result timeseries for one node across one study."""
    prmname: str
    study: str
    # Each key is a result type (e.g. "flow_out", "storage", "shortage_volume")
    # Each value is a list of [date_str, value] pairs
    series: dict[str, list[list]]


class SummaryResponse(BaseModel):
    """Annual aggregate summary for a study."""
    study: str
    years: list[str]
    total_shortage_volume: list[float]   # TAF/yr
    total_shortage_cost: list[float]     # $/yr
    total_operation_cost: list[float]    # $/yr
