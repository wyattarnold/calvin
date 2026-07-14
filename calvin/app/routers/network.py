"""
Network API router — serves network structure (nodes, links, regions).
"""

from __future__ import annotations

from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Query

from calvin.app.schemas import BoundDetail, CostDefDetail, LinkDetail, NodeDetail, NodeSummary
from calvin.app.state import AppState, get_state
from calvin.network.loader import Bound, CostDef, NetworkLink, NetworkNode

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_bound(b: Bound) -> BoundDetail:
    return BoundDetail(type=b.type, bound=b.bound)


def _serialize_cost(c: CostDef | None) -> CostDefDetail | None:
    if c is None:
        return None
    return CostDefDetail(type=c.type, cost=c.cost, costs=c.costs)


def _link_to_detail(link: NetworkLink) -> LinkDetail:
    return LinkDetail(
        prmname=link.prmname,
        description=link.description,
        link_type=link.link_type,
        origin=link.origin,
        terminus=link.terminus,
        amplitude=link.amplitude,
        disabled=link.disabled,
        filepath=link.filepath,
        flow=link.flow,
        bounds=[_serialize_bound(b) for b in link.bounds],
        costs=_serialize_cost(link.costs),
        properties=link.properties,
    )


def _node_to_detail(node: NetworkNode) -> NodeDetail:
    return NodeDetail(
        prmname=node.prmname,
        description=node.description,
        node_type=node.node_type,
        disabled=node.disabled,
        filepath=node.filepath,
        initialstorage=node.initialstorage,
        endingstorage=node.endingstorage,
        areacapfactor=node.areacapfactor,
        inflows=node.inflows,
        sinks=node.sinks,
        flow=node.flow,
        storage=node.storage,
        evaporation=node.evaporation,
        bounds=[_serialize_bound(b) for b in node.bounds],
        costs=_serialize_cost(node.costs),
        properties=node.properties,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", summary="Full network as GeoJSON FeatureCollection")
def get_network(state: AppState = Depends(get_state)) -> dict:
    """Return all nodes and links as a GeoJSON FeatureCollection."""
    return state.geojson


@router.get("/nodes", response_model=list[NodeSummary], summary="List all nodes")
def list_nodes(state: AppState = Depends(get_state)) -> list[NodeSummary]:
    """Return a summary list of all network nodes."""
    return [
        NodeSummary(
            prmname=n.prmname,
            description=n.description,
            node_type=n.node_type,
            disabled=n.disabled,
        )
        for n in state.network.nodes.values()
    ]


@router.get("/node/{prmname}", response_model=NodeDetail, summary="Node detail")
def get_node(prmname: str, state: AppState = Depends(get_state)) -> NodeDetail:
    """Return full node details including timeseries, bounds, and cost curves."""
    node = state.network.nodes.get(prmname) or state.network.nodes.get(prmname.upper())
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{prmname}' not found")
    return _node_to_detail(node)


@router.get("/link/{prmname}", response_model=LinkDetail, summary="Link detail")
def get_link(prmname: str, state: AppState = Depends(get_state)) -> LinkDetail:
    """Return full link details including flow timeseries, bounds, and cost curves."""
    upper = prmname.upper()
    link = next(
        (l for l in state.network.links if l.prmname == prmname or l.prmname == upper),
        None,
    )
    if not link:
        raise HTTPException(status_code=404, detail=f"Link '{prmname}' not found")
    return _link_to_detail(link)


@router.get("/node/{prmname}/links", response_model=list[LinkDetail], summary="Links connected to a node")
def get_node_links(prmname: str, state: AppState = Depends(get_state)) -> list[LinkDetail]:
    """Return all links where origin or terminus matches this node (case-insensitive)."""
    upper = prmname.upper()
    matches = [
        l for l in state.network.links
        if l.origin.upper() == upper or l.terminus.upper() == upper
    ]
    if not matches and prmname not in state.network.nodes and upper not in state.network.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{prmname}' not found")
    return [_link_to_detail(l) for l in matches]


@router.get("/node/{prmname}/neighborhood", summary="Neighborhood subgraph")
def get_neighborhood(
    prmname: str,
    depth: int = Query(2, ge=1, le=5),
    state: AppState = Depends(get_state),
) -> dict:
    """Return nodes and links within *depth* hops of the given node (both upstream and downstream)."""
    network = state.network
    canonical = prmname if prmname in network.nodes else prmname.upper()
    if canonical not in network.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{prmname}' not found")

    # Build adjacency maps once
    out_adj: dict[str, list] = defaultdict(list)  # node -> outgoing links
    in_adj: dict[str, list] = defaultdict(list)   # node -> incoming links
    for link in network.links:
        out_adj[link.origin].append(link)
        in_adj[link.terminus].append(link)

    # Two separate directional BFS passes to prevent combinatorial explosion.
    # A bidirectional BFS would discover node B at dist=0 if B is a downstream
    # neighbour of an upstream node, then expand B in both directions again —
    # causing exponential growth on dense junctions.  Instead:
    #   • upstream pass  : only follow in_adj  (never go forward)
    #   • downstream pass: only follow out_adj (never go backward)
    # Then merge, picking the shortest-path distance when a node appears in both.

    # Upstream-only BFS (negative distances)
    up: dict[str, int] = {canonical: 0}
    q: deque[tuple[str, int]] = deque([(canonical, 0)])
    while q:
        node, dist = q.popleft()
        if dist <= -depth:
            continue
        for link in in_adj[node]:
            if link.origin not in up:
                up[link.origin] = dist - 1
                q.append((link.origin, dist - 1))

    # Downstream-only BFS (positive distances)
    dn: dict[str, int] = {canonical: 0}
    q = deque([(canonical, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= depth:
            continue
        for link in out_adj[node]:
            if link.terminus not in dn:
                dn[link.terminus] = dist + 1
                q.append((link.terminus, dist + 1))

    # Merge: if a node appears in both passes, keep the shorter-path distance
    node_dist: dict[str, int] = {}
    for n, d in up.items():
        node_dist[n] = d
    for n, d in dn.items():
        if n not in node_dist or abs(d) < abs(node_dist[n]):
            node_dist[n] = d

    # Collect all links whose both endpoints are in the discovered subgraph
    visited_links: set[str] = set()
    for link in network.links:
        if link.origin in node_dist and link.terminus in node_dist:
            visited_links.add(link.prmname)

    nodes_out = []
    for pname, dist in node_dist.items():
        node = network.nodes.get(pname)
        if node:
            nodes_out.append({
                "prmname": pname,
                "node_type": node.node_type,
                "description": node.description,
                "distance": dist,
            })

    links_out = [
        {
            "prmname": l.prmname,
            "origin": l.origin,
            "terminus": l.terminus,
            "description": l.description,
        }
        for l in network.links
        if l.prmname in visited_links
    ]

    return {"focus": canonical, "nodes": nodes_out, "links": links_out}


@router.get("/regions", summary="Region membership")
def get_regions(state: AppState = Depends(get_state)) -> dict[str, list[str]]:
    """Return a dict mapping region names to lists of member node prmnames."""
    return state.network.regions
