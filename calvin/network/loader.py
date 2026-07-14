"""
Network data loader.

Crawls the calvin-network-data repository, reads node.geojson and link.json
files, resolves $ref pointers to CSV timeseries, and returns a structured
network representation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bound:
    """A single bound definition on a link or node."""
    type: str  # NOB, LBC, UBC, LBM, UBM, LBT, UBT, EQC, EQT, EQM
    bound: Any = None  # float, dict {month: value}, or list of [date, value]

    @property
    def is_constant(self) -> bool:
        return self.type in ("LBC", "UBC", "EQC")

    @property
    def is_monthly(self) -> bool:
        return self.type in ("LBM", "UBM", "EQM")

    @property
    def is_timeseries(self) -> bool:
        return self.type in ("LBT", "UBT", "EQT")

    @property
    def is_equality(self) -> bool:
        return self.type.startswith("EQ")


@dataclass
class CostDef:
    """Cost definition for a link or sink."""
    type: str  # "NONE", "None", "Constant", "Monthly Variable", "Annual Variable"
    cost: float | None = None
    costs: dict[str, Any] | None = None  # month -> penalty function array


@dataclass
class NetworkNode:
    """A node in the water network."""
    prmname: str
    description: str = ""
    node_type: str = ""  # Junction, Surface Storage, Groundwater Storage, Demand, etc.
    disabled: bool = False
    filepath: str = ""

    # Storage properties
    initialstorage: float | None = None
    endingstorage: float | None = None
    areacapfactor: float = 0.0

    # Timeseries data (parsed from CSV via $ref)
    inflows: dict[str, list[list]] | None = None  # name -> [[date,kaf], ...]
    sinks: list[dict] | None = None
    flow: list[list] | None = None  # [[date,kaf], ...]
    storage: list[list] | None = None  # [[date,kaf], ...]
    evaporation: list[list] | None = None  # [[date,kaf], ...]
    bounds: list[Bound] = field(default_factory=list)
    costs: CostDef | None = None
    properties: dict = field(default_factory=dict)  # raw properties


@dataclass
class NetworkLink:
    """A directed link (edge) in the water network."""
    prmname: str
    description: str = ""
    link_type: str = ""  # Diversion, Return Flow, etc.
    origin: str = ""
    terminus: str = ""
    amplitude: float = 1.0
    disabled: bool = False
    filepath: str = ""

    flow: list[list] | None = None  # [[date,kaf], ...]
    bounds: list[Bound] = field(default_factory=list)
    costs: CostDef | None = None
    properties: dict = field(default_factory=dict)  # raw properties


@dataclass
class Network:
    """The full water network."""
    nodes: dict[str, NetworkNode] = field(default_factory=dict)
    links: list[NetworkLink] = field(default_factory=list)
    regions: dict[str, list[str]] = field(default_factory=dict)  # region -> [prmnames]


# ---------------------------------------------------------------------------
# $ref resolver
# ---------------------------------------------------------------------------

def _resolve_ref(value: Any, base_dir: Path) -> Any:
    """Recursively resolve $ref pointers in JSON data.

    A $ref like {"$ref": "flow.csv"} reads the CSV file relative to
    base_dir and returns it as a list of lists.
    """
    if isinstance(value, dict):
        if "$ref" in value:
            ref_path = base_dir / value["$ref"]
            return _read_csv_data(ref_path)
        return {k: _resolve_ref(v, base_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_ref(item, base_dir) for item in value]
    return value


def _read_csv_data(filepath: Path) -> list[list]:
    """Read a CSV file and return as list of lists (preserving header)."""
    if not filepath.exists():
        logger.warning("CSV file not found: %s", filepath)
        return []
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                rows.append(row)  # header as strings
            else:
                # Try to parse numeric values
                parsed = []
                for val in row:
                    try:
                        parsed.append(float(val))
                    except ValueError:
                        parsed.append(val)
                rows.append(parsed)
    return rows


def _parse_bounds(raw_bounds: list[dict] | None, base_dir: Path) -> list[Bound]:
    """Parse bound definitions, resolving $ref on timeseries bounds."""
    if not raw_bounds:
        return []
    bounds = []
    for b in raw_bounds:
        btype = b.get("type", "NOB")
        raw_val = b.get("bound", None)
        resolved = _resolve_ref(raw_val, base_dir) if isinstance(raw_val, dict) else raw_val
        bounds.append(Bound(type=btype, bound=resolved))
    return bounds


def _parse_costs(raw_costs: dict | None, base_dir: Path) -> CostDef | None:
    """Parse cost definitions, resolving $ref on penalty function CSVs."""
    if not raw_costs:
        return None
    ctype = raw_costs.get("type", "NONE")
    if ctype in ("NONE", "None", None):
        return CostDef(type="NONE")
    if ctype == "Constant":
        return CostDef(type="Constant", cost=float(raw_costs.get("cost", 0)))
    if ctype in ("Monthly Variable", "Annual Variable"):
        raw_map = raw_costs.get("costs", {})
        resolved = {}
        for key, val in raw_map.items():
            resolved[key] = _resolve_ref(val, base_dir)
        return CostDef(type=ctype, costs=resolved)
    return CostDef(type="NONE")


def _parse_inflows(raw_inflows: Any, base_dir: Path) -> dict[str, list[list]] | None:
    """Parse inflow definitions. Can be dict of named inflows or None."""
    if not raw_inflows:
        return None
    result = {}
    if isinstance(raw_inflows, dict):
        for name, idef in raw_inflows.items():
            if isinstance(idef, dict):
                if "$ref" in idef:
                    result[name] = _resolve_ref(idef, base_dir)
                elif "inflow" in idef:
                    result[name] = _resolve_ref(idef["inflow"], base_dir)
                else:
                    # Nested structure: resolve all $refs
                    resolved = _resolve_ref(idef, base_dir)
                    if isinstance(resolved, dict) and "inflow" in resolved:
                        result[name] = resolved["inflow"]
                    else:
                        result[name] = resolved
            elif isinstance(idef, list):
                result[name] = idef
    return result if result else None


def _parse_sinks(raw_sinks: Any, base_dir: Path) -> list[dict] | None:
    """Parse sink definitions, resolving $ref and parsing bounds/costs."""
    if not raw_sinks:
        return None
    sinks = []
    if isinstance(raw_sinks, list):
        for s in raw_sinks:
            if isinstance(s, dict):
                # Each list element is {sink_name: {costs, bounds, flow, ...}}
                # matching Node.js: for (sinkName in p.sinks[i]) sink = p.sinks[i][sinkName]
                for name, sink_data in s.items():
                    if isinstance(sink_data, dict):
                        sink = {
                            "name": name,
                            "flow": _resolve_ref(sink_data.get("flow"), base_dir) if sink_data.get("flow") else None,
                            "bounds": _parse_bounds(sink_data.get("bounds"), base_dir),
                            "costs": _parse_costs(sink_data.get("costs"), base_dir),
                        }
                        sinks.append(sink)
    elif isinstance(raw_sinks, dict):
        for name, s in raw_sinks.items():
            if isinstance(s, dict):
                sink = {
                    "name": name,
                    "flow": _resolve_ref(s.get("flow"), base_dir) if s.get("flow") else None,
                    "bounds": _parse_bounds(s.get("bounds"), base_dir),
                    "costs": _parse_costs(s.get("costs"), base_dir),
                }
                sinks.append(sink)
    return sinks if sinks else None


# ---------------------------------------------------------------------------
# Directory crawler
# ---------------------------------------------------------------------------

def _load_node(dirpath: Path) -> NetworkNode | None:
    """Load a node from a directory containing node.geojson."""
    geojson_path = dirpath / "node.geojson"
    if not geojson_path.exists():
        return None

    with open(geojson_path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Some files have invalid control characters — parse leniently
        data = json.loads(text, strict=False)

    props = data.get("properties", {})
    prmname = props.get("prmname", "")
    if not prmname:
        return None

    node = NetworkNode(
        prmname=prmname,
        description=props.get("description", ""),
        node_type=props.get("type", ""),
        disabled=props.get("disabled", False),
        filepath=str(dirpath),
        initialstorage=props.get("initialstorage"),
        endingstorage=props.get("endingstorage"),
        areacapfactor=props.get("areacapfactor", 0.0) or 0.0,
        bounds=_parse_bounds(props.get("bounds"), dirpath),
        costs=_parse_costs(props.get("costs"), dirpath),
        flow=_resolve_ref(props.get("flow"), dirpath) if props.get("flow") else None,
        storage=_resolve_ref(props.get("storage"), dirpath) if props.get("storage") else None,
        evaporation=_resolve_ref(props.get("evaporation"), dirpath) if props.get("evaporation") else None,
        inflows=_parse_inflows(props.get("inflows"), dirpath),
        sinks=_parse_sinks(props.get("sinks"), dirpath),
        properties=props,
    )
    return node


def _load_link(dirpath: Path) -> NetworkLink | None:
    """Load a link from a directory containing link.json."""
    link_path = dirpath / "link.json"
    if not link_path.exists():
        return None

    with open(link_path) as f:
        data = json.load(f)

    prmname = data.get("prmname", "")
    if not prmname:
        return None

    link = NetworkLink(
        prmname=prmname,
        description=data.get("description", ""),
        link_type=data.get("type", ""),
        origin=data.get("origin", ""),
        terminus=data.get("terminus", ""),
        amplitude=float(data.get("amplitude", 1.0)),
        disabled=data.get("disabled", False),
        filepath=str(dirpath),
        bounds=_parse_bounds(data.get("bounds"), dirpath),
        costs=_parse_costs(data.get("costs"), dirpath),
        flow=_resolve_ref(data.get("flow"), dirpath) if data.get("flow") else None,
        properties=data,
    )
    return link


def _crawl_directory(data_dir: Path, network: Network, region: str | None = None):
    """Recursively crawl a data directory to discover nodes, links, and regions."""
    if not data_dir.is_dir():
        return

    # Check for region.geojson
    region_file = data_dir / "region.geojson"
    if region_file.exists():
        region_name = data_dir.name
        if region_name not in network.regions:
            network.regions[region_name] = []

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("."):
            continue

        # Try loading as node
        node = _load_node(entry)
        if node and not node.disabled:
            network.nodes[node.prmname] = node
            if region and region in network.regions:
                network.regions[region].append(node.prmname)

        # Try loading as link
        link = _load_link(entry)
        if link and not link.disabled:
            network.links.append(link)
            if region and region in network.regions:
                network.regions[region].append(link.prmname)

        # Recurse into subdirectories (for regions with nested structure)
        sub_region = data_dir / name / "region.geojson"
        if sub_region.exists():
            current_region = name
            if current_region not in network.regions:
                network.regions[current_region] = []
            _crawl_directory(entry, network, region=current_region)
        elif not (entry / "node.geojson").exists() and not (entry / "link.json").exists():
            # Not a node or link directory — might contain nested nodes/links
            _crawl_directory(entry, network, region=region)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_network(
    data_path: str | Path,
    nodes: list[str] | None = None,
    regions: list[str] | None = None,
) -> Network:
    """Load the calvin-network-data repository into a Network object.

    Parameters
    ----------
    data_path : str or Path
        Path to the ``data/`` directory inside calvin-network-data.
    nodes : list of str, optional
        If provided, only include these nodes (by prmname) and their
        connecting links. Pass ``None`` for the full network.
    regions : list of str, optional
        If provided, include all nodes within the named regions.

    Returns
    -------
    Network
        Populated network with nodes, links, and region mappings.
    """
    data_path = Path(data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    network = Network()
    _crawl_directory(data_path, network)

    logger.info(
        "Loaded %d nodes, %d links, %d regions",
        len(network.nodes), len(network.links), len(network.regions),
    )

    # Expand regions into node list
    if regions:
        if nodes is None:
            nodes = []
        for r in regions:
            region_nodes = network.regions.get(r, [])
            nodes.extend(region_nodes)

    # Filter to requested subnet
    if nodes:
        node_set = set(n.upper() for n in nodes)
        # Keep requested nodes
        network.nodes = {
            k: v for k, v in network.nodes.items() if k.upper() in node_set
        }
        # Keep links where both origin and terminus are in the node set,
        # or mark boundary links
        all_node_names = set(network.nodes.keys())
        filtered_links = []
        for link in network.links:
            origin_in = link.origin in all_node_names
            terminus_in = link.terminus in all_node_names
            if origin_in or terminus_in:
                filtered_links.append(link)
        network.links = filtered_links

    return network
