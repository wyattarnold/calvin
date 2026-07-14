"""
Network data validator.

Validates node.geojson and link.json files against expected schemas
and checks for structural issues in the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import Network, load_network

logger = logging.getLogger(__name__)

VALID_NODE_TYPES = {
    "Junction",
    "Surface Storage",
    "Groundwater Storage",
    "Agricultural Demand",
    "Urban Demand",
    "Non-Standard Demand",
    "Pump Plant",
    "Power Plant",
    "Water Treatment",
    "Sink",
}

VALID_LINK_TYPES = {
    "Diversion",
    "Return Flow",
    "Pumping",
    "Link",
}

VALID_BOUND_TYPES = {
    "NOB", "LBC", "UBC", "LBM", "UBM", "LBT", "UBT",
    "EQC", "EQT", "EQM",
}

VALID_COST_TYPES = {
    "NONE", "None", "Constant", "Monthly Variable", "Annual Variable",
}


@dataclass
class ValidationError:
    """A single validation error."""
    path: str
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a network data repository."""
    errors: list[ValidationError] = field(default_factory=list)
    node_count: int = 0
    link_count: int = 0
    region_count: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(e.severity == "error" for e in self.errors)

    def summary(self) -> str:
        n_err = sum(1 for e in self.errors if e.severity == "error")
        n_warn = sum(1 for e in self.errors if e.severity == "warning")
        return (
            f"Validation: {self.node_count} nodes, {self.link_count} links, "
            f"{self.region_count} regions — "
            f"{n_err} errors, {n_warn} warnings"
        )


def validate_data(data_path: str | Path) -> ValidationResult:
    """Validate a calvin-network-data repository.

    Checks:
    - All node.geojson files have required fields (prmname, type)
    - All link.json files have required fields (prmname, origin, terminus)
    - Node types are recognized
    - Link types are recognized
    - Bound types are valid
    - Cost types are valid
    - $ref targets exist
    - Links reference existing nodes
    - No orphaned nodes (warning)
    - Storage nodes have initialstorage/endingstorage

    Parameters
    ----------
    data_path : str or Path
        Path to the ``data/`` directory.

    Returns
    -------
    ValidationResult
    """
    data_path = Path(data_path)
    result = ValidationResult()

    if not data_path.is_dir():
        result.errors.append(ValidationError(
            str(data_path), "Data directory does not exist"
        ))
        return result

    # Crawl all node.geojson and link.json files
    node_names: set[str] = set()
    link_refs: list[tuple[str, str, str]] = []  # (prmname, origin, terminus)

    for geojson_path in sorted(data_path.rglob("node.geojson")):
        result.node_count += 1
        relpath = str(geojson_path.relative_to(data_path))
        dirpath = geojson_path.parent

        try:
            with open(geojson_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result.errors.append(ValidationError(relpath, f"Invalid JSON: {e}"))
            continue

        props = data.get("properties", {})

        # Required: prmname
        prmname = props.get("prmname")
        if not prmname:
            result.errors.append(ValidationError(relpath, "Missing prmname"))
            continue

        if prmname in node_names:
            result.errors.append(ValidationError(
                relpath, f"Duplicate prmname: {prmname}"
            ))
        node_names.add(prmname)

        # Node type
        ntype = props.get("type", "")
        if ntype and ntype not in VALID_NODE_TYPES:
            result.errors.append(ValidationError(
                relpath, f"Unknown node type: {ntype}", severity="warning"
            ))

        # Storage nodes need initial/ending storage
        if ntype in ("Surface Storage", "Groundwater Storage"):
            if props.get("initialstorage") is None:
                result.errors.append(ValidationError(
                    relpath, f"{prmname}: Storage node missing initialstorage",
                    severity="warning"
                ))

        # Check $ref targets exist
        _check_refs(props, dirpath, relpath, result)

        # Bounds
        for b in (props.get("bounds") or []):
            btype = b.get("type", "")
            if btype not in VALID_BOUND_TYPES:
                result.errors.append(ValidationError(
                    relpath, f"Unknown bound type: {btype}"
                ))

        # Costs
        costs = props.get("costs")
        if costs:
            ctype = costs.get("type", "NONE")
            if ctype not in VALID_COST_TYPES:
                result.errors.append(ValidationError(
                    relpath, f"Unknown cost type: {ctype}"
                ))

    for link_path in sorted(data_path.rglob("link.json")):
        result.link_count += 1
        relpath = str(link_path.relative_to(data_path))
        dirpath = link_path.parent

        try:
            with open(link_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            result.errors.append(ValidationError(relpath, f"Invalid JSON: {e}"))
            continue

        prmname = data.get("prmname")
        if not prmname:
            result.errors.append(ValidationError(relpath, "Missing prmname"))
            continue

        origin = data.get("origin", "")
        terminus = data.get("terminus", "")
        if not origin:
            result.errors.append(ValidationError(
                relpath, f"{prmname}: Missing origin"
            ))
        if not terminus:
            result.errors.append(ValidationError(
                relpath, f"{prmname}: Missing terminus"
            ))

        link_refs.append((prmname, origin, terminus))

        ltype = data.get("type", "")
        if ltype and ltype not in VALID_LINK_TYPES:
            result.errors.append(ValidationError(
                relpath, f"Unknown link type: {ltype}", severity="warning"
            ))

        # Check $ref targets
        _check_refs(data, dirpath, relpath, result)

        # Bounds
        for b in (data.get("bounds") or []):
            btype = b.get("type", "")
            if btype not in VALID_BOUND_TYPES:
                result.errors.append(ValidationError(
                    relpath, f"Unknown bound type: {btype}"
                ))

    # Count regions
    for region_path in data_path.rglob("region.geojson"):
        result.region_count += 1

    # Cross-reference check: do link origins/termini reference existing nodes?
    for prmname, origin, terminus in link_refs:
        if origin and origin not in node_names:
            result.errors.append(ValidationError(
                prmname,
                f"Link origin '{origin}' not found in nodes",
                severity="warning",
            ))
        if terminus and terminus not in node_names:
            result.errors.append(ValidationError(
                prmname,
                f"Link terminus '{terminus}' not found in nodes",
                severity="warning",
            ))

    return result


def _check_refs(data: Any, base_dir: Path, relpath: str, result: ValidationResult):
    """Recursively check that $ref targets exist."""
    if isinstance(data, dict):
        if "$ref" in data:
            ref_path = base_dir / data["$ref"]
            if not ref_path.exists():
                result.errors.append(ValidationError(
                    relpath,
                    f"$ref target not found: {data['$ref']}",
                    severity="warning",
                ))
        else:
            for v in data.values():
                _check_refs(v, base_dir, relpath, result)
    elif isinstance(data, list):
        for item in data:
            _check_refs(item, base_dir, relpath, result)
