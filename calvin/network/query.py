"""
Network listing and query utilities.

Provides functions to list nodes, links, and query the network
data repository — replacing the ``cnf list`` command.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from .loader import Network

logger = logging.getLogger(__name__)


def list_items(
    data_path: str | Path,
    item_type: Literal["all", "nodes", "links"] = "all",
    filter_names: list[str] | None = None,
) -> pd.DataFrame:
    """List all nodes and/or links in the data repository.

    Parameters
    ----------
    data_path : str or Path
        Path to the ``data/`` directory.
    item_type : str
        What to list: "all", "nodes", or "links".
    filter_names : list of str, optional
        Only include items whose prmname matches one of these.

    Returns
    -------
    pd.DataFrame
        Table with columns: prmname, type, kind (node/link), path, description.
    """
    data_path = Path(data_path)
    records = []

    if item_type in ("all", "nodes"):
        for geojson_path in sorted(data_path.rglob("node.geojson")):
            try:
                with open(geojson_path) as f:
                    data = json.load(f)
                props = data.get("properties", {})
                prmname = props.get("prmname", "")
                if filter_names and prmname not in filter_names:
                    continue
                records.append({
                    "prmname": prmname,
                    "type": props.get("type", ""),
                    "kind": "node",
                    "path": str(geojson_path.parent.relative_to(data_path)),
                    "description": props.get("description", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if item_type in ("all", "links"):
        for link_path in sorted(data_path.rglob("link.json")):
            try:
                with open(link_path) as f:
                    data = json.load(f)
                prmname = data.get("prmname", "")
                if filter_names and prmname not in filter_names:
                    continue
                records.append({
                    "prmname": prmname,
                    "type": data.get("type", ""),
                    "kind": "link",
                    "path": str(link_path.parent.relative_to(data_path)),
                    "description": data.get("description", ""),
                    "origin": data.get("origin", ""),
                    "terminus": data.get("terminus", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    return pd.DataFrame(records)


def find_node(data_path: str | Path, prmname: str) -> dict | None:
    """Find and return the full data for a single node by prmname.

    Returns the raw GeoJSON properties dict, or None if not found.
    """
    data_path = Path(data_path)
    for geojson_path in data_path.rglob("node.geojson"):
        try:
            with open(geojson_path) as f:
                data = json.load(f)
            props = data.get("properties", {})
            if props.get("prmname") == prmname:
                props["_filepath"] = str(geojson_path.parent)
                return props
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def find_link(data_path: str | Path, prmname: str) -> dict | None:
    """Find and return the full data for a single link by prmname.

    Returns the raw link.json dict, or None if not found.
    """
    data_path = Path(data_path)
    for link_path in data_path.rglob("link.json"):
        try:
            with open(link_path) as f:
                data = json.load(f)
            if data.get("prmname") == prmname:
                data["_filepath"] = str(link_path.parent)
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def list_regions(data_path: str | Path) -> list[dict]:
    """List all regions in the data repository.

    Returns a list of dicts with keys: name, path.
    """
    data_path = Path(data_path)
    regions = []
    for region_path in sorted(data_path.rglob("region.geojson")):
        regions.append({
            "name": region_path.parent.name,
            "path": str(region_path.parent.relative_to(data_path)),
        })
    return regions
