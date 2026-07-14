/**
 * API client — thin wrappers around fetch for all Calvin Network App endpoints.
 */

const BASE = "/api";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

/** Full GeoJSON FeatureCollection (nodes + links). */
export const fetchNetwork = () => get("/network");

/** Summary list of all nodes. */
export const fetchNodes = () => get("/network/nodes");

/** Full node detail (timeseries, bounds, costs). */
export const fetchNode = (prmname) => get(`/network/node/${encodeURIComponent(prmname)}`);

/** Region → [prmnames] mapping. */
export const fetchRegions = () => get("/network/regions");

/** Full link detail (flow, bounds, costs). */
export const fetchLink = (prmname) => get(`/network/link/${prmname}`);

/** All links connected to a node (origin or terminus). */
export const fetchNodeLinks = (prmname) => get(`/network/node/${encodeURIComponent(prmname)}/links`);

/** Neighborhood subgraph (nodes + links within `depth` hops). */
export const fetchNeighborhood = (prmname, depth = 2) =>
  get(`/network/node/${encodeURIComponent(prmname)}/neighborhood?depth=${depth}`);

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/** List of available studies. */
export const fetchStudies = () => get("/results/studies");

/**
 * Annual summary (total shortage cost/volume) for a study.
 * @param {string|null} study - Study name; null uses the active study.
 */
export const fetchSummary = (study = null) =>
  get(`/results/summary${study ? `?study=${study}` : ""}`);

/**
 * All result timeseries for a single node.
 * @param {string} prmname
 * @param {string|null} study
 */
export const fetchNodeResults = (prmname, study = null) => {
  const params = study ? `?study=${encodeURIComponent(study)}` : "";
  return get(`/results/node/${encodeURIComponent(prmname)}${params}`);
};

/**
 * Per-column total shortage volume for map highlighting.
 * @param {string|null} study
 */
export const fetchShortageNodes = (study = null) =>
  get(`/results/shortage-nodes${study ? `?study=${study}` : ""}`);

/**
 * Debug source/sink link flows for infeasibility diagnosis.
 * @param {string|null} study
 */
export const fetchDebugLinks = (study = null) =>
  get(`/results/debug-links${study ? `?study=${study}` : ""}`);

/**
 * COSVF penalty curve data for a storage node (LF studies only).
 * Returns 404 if the study has no COSVF data or the node is not a reservoir.
 * @param {string} prmname
 * @param {string|null} study
 */
export const fetchCosvf = (prmname, study = null) => {
  const params = study ? `?study=${encodeURIComponent(study)}` : "";
  return get(`/results/cosvf/${encodeURIComponent(prmname)}${params}`);
};

/**
 * All node values at a specific date (for map coloring).
 * @param {string} date - e.g. "1975-09-30"
 * @param {string|null} study
 */
export const fetchTimeslice = (date, study = null) => {
  const params = new URLSearchParams({ date });
  if (study) params.set("study", study);
  return get(`/results/timeslice?${params}`);
};
