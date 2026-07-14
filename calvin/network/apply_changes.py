"""
Apply external changes to the network data repository.

Supports importing data from:
- Excel workbooks (exported from the web app)
- CSV files with multi-column timeseries

Replaces the ``cnf apply-changes app`` and ``cnf apply-changes csv`` commands.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def apply_excel(
    excel_path: str | Path,
    data_path: str | Path,
    dry_run: bool = False,
) -> list[str]:
    """Import data from an Excel workbook into the data repository.

    Each sheet in the workbook represents one node/link. The first row
    contains metadata: ``prmname/path/to/file``. Subsequent rows contain
    the timeseries data to write.

    Parameters
    ----------
    excel_path : str or Path
        Path to the .xlsx file.
    data_path : str or Path
        Path to the ``data/`` directory.
    dry_run : bool
        If True, only report what would change without writing files.

    Returns
    -------
    list of str
        Paths of files that were written (or would be written).
    """
    excel_path = Path(excel_path)
    data_path = Path(data_path)
    updated_files: list[str] = []

    xls = pd.ExcelFile(excel_path)

    # Build a prmname -> directory lookup by scanning the data repo
    prmname_dirs = _build_prmname_lookup(data_path)

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        if df.empty:
            continue

        # First row: "PRMNAME/relative/path"
        header_cell = str(df.iloc[0, 0]).strip()
        parts = header_cell.split("/", 1)
        prmname = parts[0].strip()
        rel_path = parts[1].strip() if len(parts) > 1 else "flow.csv"

        if prmname not in prmname_dirs:
            logger.warning("prmname '%s' not found in data repo, skipping", prmname)
            continue

        node_dir = prmname_dirs[prmname]
        target = node_dir / rel_path

        # Data starts at row 1
        data_df = df.iloc[1:].reset_index(drop=True)

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            data_df.to_csv(target, index=False, header=False)
            logger.info("Wrote %s", target)
        else:
            logger.info("[dry-run] Would write %s", target)

        updated_files.append(str(target))

    return updated_files


def apply_csv(
    csv_path: str | Path,
    data_path: str | Path,
    prop: Literal["flow", "storage", "inflows"] = "flow",
    dry_run: bool = False,
) -> list[str]:
    """Import timeseries data from a CSV file into the data repository.

    The CSV has columns ``date, PRMNAME1, PRMNAME2, ...``. Each column's
    data is written to the corresponding node's CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV.
    data_path : str or Path
        Path to the ``data/`` directory.
    prop : str
        Which property to update: "flow", "storage", or "inflows".
    dry_run : bool
        If True, only report what would change.

    Returns
    -------
    list of str
        Paths of files written.
    """
    csv_path = Path(csv_path)
    data_path = Path(data_path)
    updated_files: list[str] = []

    df = pd.read_csv(csv_path, index_col=0)

    # Build prmname lookup
    prmname_dirs = _build_prmname_lookup(data_path)

    for col in df.columns:
        prmname = col.strip()
        if prmname not in prmname_dirs:
            logger.warning("prmname '%s' not found in data repo, skipping", prmname)
            continue

        node_dir = prmname_dirs[prmname]

        # Determine target file
        if prop == "inflows":
            target = node_dir / "inflows" / "default.csv"
        elif prop == "storage":
            target = node_dir / "storage.csv"
        else:
            target = node_dir / "flow.csv"

        # Write CSV: date, kaf
        series = df[col].dropna()
        out_df = pd.DataFrame({
            "date": series.index,
            "kaf": series.values,
        })

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(target, index=False)
            logger.info("Wrote %s", target)

            # Ensure the node's JSON references this file
            _ensure_ref(node_dir, prop, target.name if prop != "inflows" else "inflows/default.csv")
        else:
            logger.info("[dry-run] Would write %s", target)

        updated_files.append(str(target))

    return updated_files


def _build_prmname_lookup(data_path: Path) -> dict[str, Path]:
    """Build a mapping from prmname to directory path."""
    lookup = {}

    for geojson in data_path.rglob("node.geojson"):
        try:
            with open(geojson) as f:
                data = json.load(f)
            prmname = data.get("properties", {}).get("prmname")
            if prmname:
                lookup[prmname] = geojson.parent
        except (json.JSONDecodeError, KeyError):
            continue

    for link_json in data_path.rglob("link.json"):
        try:
            with open(link_json) as f:
                data = json.load(f)
            prmname = data.get("prmname")
            if prmname:
                lookup[prmname] = link_json.parent
        except (json.JSONDecodeError, KeyError):
            continue

    return lookup


def _ensure_ref(node_dir: Path, prop: str, ref_target: str):
    """Ensure a node/link's JSON file has a $ref for the given property."""
    # Try node.geojson first, then link.json
    for json_name in ("node.geojson", "link.json"):
        json_path = node_dir / json_name
        if not json_path.exists():
            continue

        with open(json_path) as f:
            data = json.load(f)

        if json_name == "node.geojson":
            props = data.get("properties", {})
        else:
            props = data

        current = props.get(prop)
        ref_obj = {"$ref": ref_target}

        if current is None or (isinstance(current, dict) and current.get("$ref") != ref_target):
            props[prop] = ref_obj
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Updated %s $ref in %s", prop, json_path)
        break
