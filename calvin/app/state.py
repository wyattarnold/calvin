"""
Application state — loads network data and model results into memory at startup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from calvin.network import load_network
from calvin.network.loader import Network

logger = logging.getLogger(__name__)

_RESULT_FILES = [
    "flow",
    "storage",
    "shortage_volume",
    "shortage_cost",
    "operation_costs",
    "evaporation",
    "pwp_short_volume",
    "pwp_short_cost",
    "dual_node",
    "dual_lower",
    "dual_upper",
]


class _LazyResultDict:
    """Dict-like container that loads result CSVs on first access and caches them.

    Scanning which files exist happens at study load time (cheap).
    Reading DataFrames into memory is deferred until a route actually requests
    a specific result, keeping startup RAM usage low.
    """

    def __init__(self, paths: dict[str, Path]):
        self._paths = paths          # stem -> Path (all available, never loaded)
        self._cache: dict[str, pd.DataFrame] = {}

    def __contains__(self, key: object) -> bool:
        return key in self._paths

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            if key not in self._paths:
                raise KeyError(key)
            logger.info("Lazy-loading result: %s", self._paths[key])
            self._cache[key] = pd.read_csv(
                self._paths[key], index_col=0, parse_dates=True
            )
        return self._cache[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._paths.keys()

    def items(self):
        """Iterate (key, DataFrame) — loads all cached + uncached files."""
        for k in self._paths:
            yield k, self[k]

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)


@dataclass
class Study:
    """A single model run (PF, COSVF, etc.) with its result CSVs."""

    name: str
    path: Path
    results: _LazyResultDict = field(default_factory=lambda: _LazyResultDict({}))
    # Keys are file stems from _RESULT_FILES that were found on disk.

    # COSVF data (present only for limited-foresight studies)
    r_dict: dict = field(default_factory=dict)   # r-dict.json: {r: {type, lb, ub, k_count}}
    cosvf: dict = field(default_factory=dict)    # cosvf-params: {r: {param: value}}

    @property
    def available(self) -> list[str]:
        return list(self.results.keys())


class AppState:
    """Singleton holding all in-memory data for the web app."""

    def __init__(self):
        self.network: Network | None = None
        self.geojson: dict | None = None          # GeoJSON FeatureCollection (nodes + links)
        self._node_coords: dict[str, list] = {}   # prmname -> [lon, lat]
        self.studies: dict[str, Study] = {}
        self.active_study: str | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_prebuilt(self, tmpdir: Path) -> None:
        """Load from pre-built JSON files in a bundle tmpdir (hosted mode).

        Expects:
          tmpdir/prebuilt/network.geojson  — pre-built GeoJSON feature collection
          tmpdir/prebuilt/network.json     — serialized Network (nodes/links/regions)
          tmpdir/studies/*/                — study directories with results/

        Avoids calling load_network() (which reads ~7 000 raw files) so startup
        memory stays well within the 512 MB Render free-plan limit.
        """
        prebuilt = tmpdir / "prebuilt"

        with open(prebuilt / "network.geojson") as f:
            self.geojson = json.load(f)
        logger.info("Loaded pre-built GeoJSON: %d features", len(self.geojson["features"]))

        with open(prebuilt / "network.json") as f:
            data = json.load(f)
        self.network = _network_from_dict(data)
        logger.info(
            "Loaded pre-built network: %d nodes, %d links",
            len(self.network.nodes), len(self.network.links),
        )

        # Populate node coords from GeoJSON (used by fly-to)
        for feat in self.geojson["features"]:
            if feat["properties"].get("feature_class") == "node":
                coords = (feat.get("geometry") or {}).get("coordinates")
                if coords:
                    self._node_coords[feat["properties"]["prmname"]] = coords

        studies_dir = tmpdir / "studies"
        if studies_dir.is_dir():
            for sp in sorted(studies_dir.iterdir()):
                if sp.is_dir():
                    self._load_study(sp)

        if self.studies:
            self.active_study = next(iter(self.studies))

    def load(
        self,
        data_path: str | Path,
        study_paths: list[str | Path],
        default_study: str | None = None,
    ) -> None:
        """Load network and studies into memory. Called once at startup."""
        data_path = Path(data_path)
        logger.info("Loading network from %s …", data_path)
        self.network = load_network(str(data_path))
        logger.info(
            "Network loaded: %d nodes, %d links, %d regions",
            len(self.network.nodes),
            len(self.network.links),
            len(self.network.regions),
        )

        self._build_geojson(data_path)

        for sp in study_paths:
            self._load_study(Path(sp))

        if default_study and default_study in self.studies:
            self.active_study = default_study
        elif self.studies:
            self.active_study = next(iter(self.studies))

    def _build_geojson(self, data_path: Path) -> None:
        """Build a GeoJSON FeatureCollection from node.geojson files + link geometry."""
        features = []
        node_coords: dict[str, list] = {}

        # --- Nodes ---
        for prmname, node in self.network.nodes.items():
            geojson_path = Path(node.filepath) / "node.geojson"
            if not geojson_path.exists():
                logger.debug("No node.geojson for %s, skipping geometry", prmname)
                continue
            try:
                text = geojson_path.read_text()
                # Some node.geojson files contain literal unescaped newlines
                # inside string values (invalid JSON). Replace them with a space.
                text = text.replace("\n", " ").replace("\r", " ")
                raw = json.loads(text)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", geojson_path, e)
                continue

            coords = (raw.get("geometry") or {}).get("coordinates")
            if coords:
                node_coords[prmname] = coords

            feature = {
                "type": "Feature",
                "geometry": raw.get("geometry"),
                "properties": {
                    "prmname": node.prmname,
                    "description": node.description,
                    "node_type": node.node_type,
                    "disabled": node.disabled,
                    "feature_class": "node",
                },
            }
            features.append(feature)

        self._node_coords = node_coords

        # --- Links ---
        for link in self.network.links:
            o_coords = node_coords.get(link.origin)
            t_coords = node_coords.get(link.terminus)
            if not o_coords or not t_coords:
                # Emit a feature with null geometry so the link is still queryable
                geometry = None
            else:
                geometry = {
                    "type": "LineString",
                    "coordinates": [o_coords, t_coords],
                }
            origin_node = self.network.nodes.get(link.origin)
            terminus_node = self.network.nodes.get(link.terminus)
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "prmname": link.prmname,
                    "description": link.description,
                    "link_type": link.link_type,
                    "origin": link.origin,
                    "terminus": link.terminus,
                    "origin_type": origin_node.node_type if origin_node else None,
                    "terminus_type": terminus_node.node_type if terminus_node else None,
                    "amplitude": link.amplitude,
                    "disabled": link.disabled,
                    "feature_class": "link",
                },
            }
            features.append(feature)

        self.geojson = {"type": "FeatureCollection", "features": features}
        logger.info(
            "GeoJSON built: %d features (%d with geometry)",
            len(features),
            sum(1 for f in features if f["geometry"]),
        )

    def _load_study(self, path: Path) -> None:
        """Load result CSVs from a study directory."""
        results_dir = path / "results"
        if not results_dir.exists():
            logger.warning("No results/ directory found under %s, skipping", path)
            return

        name = path.name
        study = Study(name=name, path=path)

        paths: dict[str, Path] = {}
        for stem in _RESULT_FILES:
            csv_path = results_dir / f"{stem}.csv"
            if csv_path.exists():
                paths[stem] = csv_path
                logger.info("Registered %s/%s.csv (lazy)", name, stem)
        study.results = _LazyResultDict(paths)

        # COSVF: r-dict.json at study root
        r_dict_path = path / "r-dict.json"
        if r_dict_path.exists():
            try:
                with open(r_dict_path) as f:
                    study.r_dict = json.load(f)
                logger.info("Loaded r-dict.json for '%s': %d reservoirs", name, len(study.r_dict))
            except Exception as e:
                logger.warning("Failed to load %s: %s", r_dict_path, e)

        # COSVF: cosvf-params.csv at study root (EA output, not in results/)
        cosvf_csv = path / "cosvf-params.csv"
        if cosvf_csv.exists():
            try:
                df = pd.read_csv(cosvf_csv)
                for _, row in df.iterrows():
                    r, param, value = str(row["r"]), str(row["param"]), float(row["value"])
                    if r not in study.cosvf:
                        study.cosvf[r] = {}
                    study.cosvf[r][param] = value
                logger.info("Loaded cosvf-params.csv for '%s': %d reservoirs", name, len(study.cosvf))
            except Exception as e:
                logger.warning("Failed to load %s: %s", cosvf_csv, e)

        if study.results:
            self.studies[name] = study
            logger.info("Study '%s' loaded with: %s", name, list(study.results))
        else:
            logger.warning("Study '%s' had no readable result CSVs, skipping", name)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_study(self, name: str | None = None) -> Study | None:
        """Return a study by name, falling back to the active study."""
        key = name or self.active_study
        return self.studies.get(key)


# ---------------------------------------------------------------------------
# Network serialization helpers (used by bundle command)
# ---------------------------------------------------------------------------

def network_to_dict(network) -> dict:
    """Serialize a Network object to a JSON-serializable dict."""
    def _bound(b):
        return {"type": b.type, "bound": b.bound}

    def _cost(c):
        if c is None:
            return None
        return {"type": c.type, "cost": c.cost, "costs": c.costs}

    def _node(n):
        # Omit large solver-input timeseries (flow, storage, evaporation, inflows, sinks)
        # — the web app does not display these; excluding them keeps network.json small
        # enough to load on a 512 MB host. Results are served from the results CSVs.
        return {
            "prmname": n.prmname, "description": n.description,
            "node_type": n.node_type, "disabled": n.disabled,
            "initialstorage": n.initialstorage, "endingstorage": n.endingstorage,
            "areacapfactor": n.areacapfactor,
            "bounds": [_bound(b) for b in n.bounds],
            "costs": _cost(n.costs),
            "properties": n.properties,
        }

    def _link(l):
        return {
            "prmname": l.prmname, "description": l.description,
            "link_type": l.link_type, "origin": l.origin, "terminus": l.terminus,
            "amplitude": l.amplitude, "disabled": l.disabled,
            "bounds": [_bound(b) for b in l.bounds],
            "costs": _cost(l.costs),
            "properties": l.properties,
        }

    return {
        "nodes": {k: _node(v) for k, v in network.nodes.items()},
        "links": [_link(l) for l in network.links],
        "regions": network.regions,
    }


def _network_from_dict(data: dict):
    """Reconstruct a Network object from a serialized dict (no file I/O)."""
    from calvin.network.loader import Bound, CostDef, Network, NetworkLink, NetworkNode

    def _bound(d):
        return Bound(type=d["type"], bound=d["bound"])

    def _cost(d):
        if d is None:
            return None
        return CostDef(type=d["type"], cost=d.get("cost"), costs=d.get("costs"))

    nodes = {
        k: NetworkNode(
            prmname=nd["prmname"], description=nd.get("description", ""),
            node_type=nd.get("node_type", ""), disabled=nd.get("disabled", False),
            initialstorage=nd.get("initialstorage"), endingstorage=nd.get("endingstorage"),
            areacapfactor=nd.get("areacapfactor", 0.0),
            inflows=nd.get("inflows"), sinks=nd.get("sinks"),
            flow=nd.get("flow"), storage=nd.get("storage"), evaporation=nd.get("evaporation"),
            bounds=[_bound(b) for b in nd.get("bounds", [])],
            costs=_cost(nd.get("costs")),
            properties=nd.get("properties", {}),
        )
        for k, nd in data["nodes"].items()
    }

    links = [
        NetworkLink(
            prmname=ld["prmname"], description=ld.get("description", ""),
            link_type=ld.get("link_type", ""), origin=ld.get("origin", ""),
            terminus=ld.get("terminus", ""), amplitude=ld.get("amplitude", 1.0),
            disabled=ld.get("disabled", False), flow=ld.get("flow"),
            bounds=[_bound(b) for b in ld.get("bounds", [])],
            costs=_cost(ld.get("costs")),
            properties=ld.get("properties", {}),
        )
        for ld in data["links"]
    ]

    return Network(nodes=nodes, links=links, regions=data.get("regions", {}))


# Module-level singleton — populated by server lifespan
state = AppState()


def get_state() -> AppState:
    """FastAPI dependency that returns the global AppState."""
    return state
