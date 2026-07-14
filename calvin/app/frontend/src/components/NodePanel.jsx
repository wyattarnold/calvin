import { useState, useEffect, useRef, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchNode, fetchNodeLinks, fetchNodeResults, fetchCosvf } from "../api/client.js";
import ResultsChart from "./ResultsChart.jsx";
import PenaltyCurveChart from "./PenaltyCurveChart.jsx";
import BoundsChart from "./BoundsChart.jsx";
import CosvfCurveChart from "./CosvfCurveChart.jsx";

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetaRow({ label, value }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <div className="flex gap-2 text-sm">
      <span className="text-gray-400 shrink-0 w-32">{label}</span>
      <span className="text-gray-100 break-all">{String(value)}</span>
    </div>
  );
}

function Section({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-3 border-b border-gray-700 pb-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 w-full text-left text-xs font-semibold uppercase tracking-wider text-gray-500 hover:text-gray-300 mb-2 px-4"
      >
        <span className="text-gray-600">{open ? "▾" : "▸"}</span>
        {title}
      </button>
      {open && <div className="px-4">{children}</div>}
    </div>
  );
}

// Bound types that carry real constraint information
const CONSTRAINT_TYPES = new Set(["LBT","UBT","EQT","LBM","UBM","EQM","LBC","UBC","EQC"]);

// Node types where the penalty curve on a link belongs to the terminus (demand node)
const DEMAND_NODE_TYPES = new Set(["Agricultural Demand", "Urban Demand", "Non-Standard Demand"]);
// Node types where the penalty curve on a link belongs to the origin (PWP node)
const PWP_NODE_TYPES = new Set(["Pump Plant", "Power Plant"]);

function hasMeaningfulBounds(link) {
  return (link.bounds ?? []).some(
    (b) =>
      CONSTRAINT_TYPES.has(b.type) &&
      b.bound !== null &&
      b.bound !== undefined &&
      // Skip UBC=0: means link is blocked/disabled; `disabled` flag already covers this
      !(b.type === "UBC" && b.bound === 0)
  );
}

function hasCurve(link) {
  if (!link.costs) return false;
  if (link.costs.type === "NONE" || link.costs.type === "None") return false;
  if (link.costs.type === "Constant" && (!link.costs.cost || link.costs.cost === 0)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Per-link collapsible row for the Link Constraints section
// ---------------------------------------------------------------------------

function LinkBoundRow({ link, dateRange }) {
  const [open, setOpen] = useState(false);
  const boundTypes = link.bounds
    .filter((b) => CONSTRAINT_TYPES.has(b.type) && b.bound !== null)
    .map((b) => b.type)
    .join(" · ");

  return (
    <div className="border-b border-gray-800 last:border-0 pb-2 mb-2 last:pb-0 last:mb-0">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 w-full text-left group"
      >
        <span className="text-gray-600 text-[10px]">{open ? "▾" : "▸"}</span>
        <span className="font-mono text-[11px] text-blue-400 truncate flex-1">{link.prmname}</span>
        <span className="text-[10px] text-gray-500 shrink-0 ml-1">{boundTypes}</span>
      </button>
      {link.description && (
        <p className="text-[10px] text-gray-600 ml-3.5 truncate" title={link.description}>
          {link.description}
        </p>
      )}
      {open && (
        <div className="mt-2 ml-1">
          <BoundsChart bounds={link.bounds} dateRange={dateRange} />
        </div>
      )}
    </div>
  );
}

function LinkConstraintsSection({ links, dateRange }) {
  const constrained = (links ?? []).filter(hasMeaningfulBounds);
  if (constrained.length === 0) {
    return <p className="text-gray-500 text-sm italic">No link constraints found</p>;
  }
  return (
    <div>
      {constrained.map((link) => (
        <LinkBoundRow key={link.prmname} link={link} dateRange={dateRange} />
      ))}
    </div>
  );
}

function PenaltySection({ node, links }) {
  const nodeCost = node?.costs;
  const hasNodeCurve = nodeCost && nodeCost.type !== "NONE" && nodeCost.type !== "None";
  const prmname = node?.prmname;
  const nodeType = node?.node_type;
  const linksWithCurves = (links || []).filter((link) => {
    if (!hasCurve(link)) return false;
    // Demand shortages belong to the terminus; PWP shortages to the origin.
    // Don't show a link's penalty curve on a node that isn't the "owner".
    if (DEMAND_NODE_TYPES.has(nodeType)) return link.terminus === prmname;
    if (PWP_NODE_TYPES.has(nodeType)) return link.origin === prmname;
    return false;
  });

  if (!hasNodeCurve && linksWithCurves.length === 0) {
    return <p className="text-gray-500 text-sm italic">No penalty curves found</p>;
  }

  return (
    <div className="space-y-4">
      {hasNodeCurve && (
        <PenaltyCurveChart
          costs={nodeCost.costs}
          costType={nodeCost.type}
          constantCost={nodeCost.cost}
          label="Node penalty"
        />
      )}
      {linksWithCurves.map((link) => (
        <PenaltyCurveChart
          key={link.prmname}
          costs={link.costs?.costs}
          costType={link.costs?.type}
          constantCost={link.costs?.cost}
          label={`${link.prmname} (${link.link_type || "link"})`}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Year range slider (fixed at bottom of panel, outside scroll)
// ---------------------------------------------------------------------------

const DROUGHT_PRESETS = [
  { label: "1928–37", start: "1928", end: "1937" },
  { label: "1976–77", start: "1976", end: "1977" },
  { label: "1987–92", start: "1987", end: "1992" },
];

function YearRangeSlider({ years, startIdx, endIdx, onStartChange, onEndChange }) {
  const [nearStart, setNearStart] = useState(false);

  if (years.length === 0) return null;
  const max = years.length - 1;
  const startPct = max > 0 ? (startIdx / max) * 100 : 0;
  const endPct   = max > 0 ? (endIdx   / max) * 100 : 100;

  function handleTrackPointerMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const startFrac = max > 0 ? startIdx / max : 0;
    const endFrac   = max > 0 ? endIdx   / max : 1;
    setNearStart(Math.abs(x - startFrac) <= Math.abs(x - endFrac));
  }

  function applyPreset(startYear, endYear) {
    const si = years.findIndex((y) => y >= startYear);
    let ei = -1;
    for (let i = years.length - 1; i >= 0; i--) {
      if (years[i] <= endYear) { ei = i; break; }
    }
    if (si >= 0 && ei > si) { onStartChange(si); onEndChange(ei); }
  }

  return (
    <div className="shrink-0 border-t border-gray-700 px-4 py-2 space-y-2">
      {/* Preset drought period buttons */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-gray-600 shrink-0">Dry periods:</span>
        {DROUGHT_PRESETS.map((p) => {
          const si = years.findIndex((y) => y >= p.start);
          let ei = -1;
          for (let i = years.length - 1; i >= 0; i--) {
            if (years[i] <= p.end) { ei = i; break; }
          }
          const active = si >= 0 && ei >= 0 && startIdx === si && endIdx === ei;
          return (
            <button
              key={p.label}
              onClick={() => applyPreset(p.start, p.end)}
              className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                active
                  ? "border-blue-500 text-blue-400 bg-blue-950"
                  : "border-gray-600 text-gray-400 hover:border-blue-400 hover:text-blue-400"
              }`}
            >
              {p.label}
            </button>
          );
        })}
        <button
          onClick={() => { onStartChange(0); onEndChange(max); }}
          className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ml-auto ${
            startIdx === 0 && endIdx === max
              ? "border-gray-500 text-gray-400"
              : "border-gray-700 text-gray-500 hover:border-gray-500 hover:text-gray-400"
          }`}
        >
          All
        </button>
      </div>

      {/* Single dual-handle range bar */}
      <div>
        <div className="flex justify-between text-[10px] font-mono text-gray-400 mb-1">
          <span>{years[startIdx]}</span>
          <span>{years[endIdx]}</span>
        </div>
        <div className="relative h-5" onPointerMove={handleTrackPointerMove}>
          {/* Track background + filled range */}
          <div className="absolute top-1/2 -translate-y-1/2 w-full h-1.5 rounded-full bg-gray-700 pointer-events-none">
            <div
              className="absolute h-full rounded-full bg-blue-500"
              style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}
            />
          </div>
          {/* Visual thumbs (non-interactive, sit on top for display only) */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-400 border border-gray-900 pointer-events-none"
            style={{ left: `calc(${startPct}% - 6px)`, zIndex: 10 }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-400 border border-gray-900 pointer-events-none"
            style={{ left: `calc(${endPct}% - 6px)`, zIndex: 10 }}
          />
          {/* Invisible range inputs — z-index switches based on pointer proximity */}
          <input
            type="range" min={0} max={max} value={startIdx}
            onChange={(e) => onStartChange(Math.min(Number(e.target.value), endIdx - 1))}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            style={{ zIndex: nearStart ? 5 : 3 }}
          />
          <input
            type="range" min={0} max={max} value={endIdx}
            onChange={(e) => onEndChange(Math.max(Number(e.target.value), startIdx + 1))}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            style={{ zIndex: nearStart ? 3 : 5 }}
          />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function NodePanel({ prmname, activeStudy, onClose, graphOpen, onToggleGraph }) {
  const [startIdx, setStartIdx] = useState(0);
  const [endIdx, setEndIdx] = useState(0);
  const sliderInitialized = useRef(false);

  const { data: node, isLoading: nodeLoading } = useQuery({
    queryKey: ["node", prmname],
    queryFn: () => fetchNode(prmname),
    enabled: !!prmname,
  });

  const { data: nodeLinks } = useQuery({
    queryKey: ["nodeLinks", prmname],
    queryFn: () => fetchNodeLinks(prmname),
    enabled: !!prmname,
  });

  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: ["nodeResults", prmname, activeStudy],
    queryFn: () => fetchNodeResults(prmname, activeStudy),
    enabled: !!prmname,
  });

  const isStorageNode = node?.node_type === "Surface Storage" || node?.node_type === "Groundwater Storage";
  const { data: cosvfData, isError: cosvfError } = useQuery({
    queryKey: ["cosvf", prmname, activeStudy],
    queryFn: () => fetchCosvf(prmname, activeStudy),
    enabled: !!prmname && isStorageNode,
    retry: false,         // 404 = study has no COSVF; don't hammer the server
    throwOnError: false,
  });

  // Derive sorted unique years from any result series
  const years = useMemo(() => {
    if (!results?.series) return [];
    const firstKey = Object.keys(results.series)[0];
    if (!firstKey) return [];
    const yearSet = new Set(
      results.series[firstKey]
        .map((row) => String(row[0]).slice(0, 4))
        .filter((y) => /^\d{4}$/.test(y))
    );
    return Array.from(yearSet).sort();
  }, [results]);

  // Reset the slider whenever the selected node changes, so a new node's year
  // range is shown correctly even when results come from the cache (no loading gap).
  useEffect(() => {
    sliderInitialized.current = false;
  }, [prmname]);

  // Initialise slider once results arrive for the current node.
  // Guard with a ref because switching nodes briefly clears results (loading),
  // making years go [] → [N] again and re-triggering without the guard.
  useEffect(() => {
    if (years.length > 0 && !sliderInitialized.current) {
      setStartIdx(0);
      setEndIdx(years.length - 1);
      sliderInitialized.current = true;
    }
  }, [years.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const dateRange = years.length > 0
    ? [years[Math.min(startIdx, years.length - 1)], years[Math.min(endIdx, years.length - 1)]]
    : null;

  if (!prmname) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 text-sm px-6 text-center">
        <p className="text-2xl mb-3">🗺</p>
        <p>Click a node on the map to see its details.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 shrink-0">
        <div className="min-w-0">
          <h2 className="font-mono text-blue-400 font-semibold truncate">{prmname}</h2>
          {node && <p className="text-xs text-gray-400 truncate">{node.node_type}</p>}
        </div>
        <div className="flex items-center gap-2 ml-2 shrink-0">
          {onToggleGraph && (
            <button
              onClick={onToggleGraph}
              title={graphOpen ? "Hide network graph" : "Show network graph"}
              className="text-gray-500 hover:text-gray-200 text-xs px-1.5 py-0.5 border border-gray-700 rounded transition-colors hover:border-gray-500"
            >
              {graphOpen ? "⟨" : "⟩"}
            </button>
          )}
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-200 text-xl leading-none"
            aria-label="Close panel"
          >
            ×
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto py-3">
        {nodeLoading ? (
          <p className="px-4 text-gray-500 text-sm">Loading…</p>
        ) : node ? (
          <>
            <Section title="Properties">
              <div className="space-y-1">
                <MetaRow label="Description" value={node.description} />
                <MetaRow label="Type" value={node.node_type} />
                <MetaRow label="Disabled" value={node.disabled ? "Yes" : null} />
                <MetaRow
                  label="Initial storage"
                  value={node.initialstorage != null ? `${Math.round(node.initialstorage)} TAF` : null}
                />
                <MetaRow
                  label="Ending storage"
                  value={(() => {
                    const last = results?.series?.storage?.at(-1)?.[1];
                    return last != null ? `${Math.round(last)} TAF` : null;
                  })()}
                />
              </div>
            </Section>

            {node.bounds?.some((b) =>
              CONSTRAINT_TYPES.has(b.type) && b.bound != null
            ) && (
              <Section title="Node Bounds" defaultOpen={false}>
                <BoundsChart bounds={node.bounds} dateRange={dateRange} />
              </Section>
            )}

            {nodeLinks?.some(hasMeaningfulBounds) && (
              <Section title="Link Constraints" defaultOpen={false}>
                <LinkConstraintsSection links={nodeLinks} dateRange={dateRange} />
              </Section>
            )}

            <Section title="Penalty Curves" defaultOpen={false}>
              <PenaltySection node={node} links={nodeLinks} />
            </Section>

            {isStorageNode && (cosvfData || cosvfError) && (
              <Section title="COSVF Curve" defaultOpen={false}>
                {cosvfData && Object.keys(cosvfData.params ?? {}).length > 0 ? (
                  <CosvfCurveChart
                    type={cosvfData.type}
                    lb={cosvfData.lb}
                    ub={cosvfData.ub}
                    eop_init={cosvfData.eop_init}
                    k_count={cosvfData.k_count}
                    params={cosvfData.params}
                  />
                ) : cosvfError ? (
                  <p className="text-gray-500 text-xs italic">
                    No COSVF data for the active study. Switch to a limited-foresight study.
                  </p>
                ) : (
                  <p className="text-gray-500 text-xs italic">
                    No COSVF assigned to this reservoir.
                  </p>
                )}
              </Section>
            )}
          </>
        ) : null}

        {results && Object.keys(results.series).length > 0 ? (
          <Section title="Model Results">
            <ResultsChart
              series={results.series}
              nodeType={node?.node_type}
              dateRange={dateRange}
              bounds={node?.bounds}
            />
          </Section>
        ) : resultsLoading ? (
          <Section title="Model Results">
            <p className="text-gray-500 text-sm">Loading results…</p>
          </Section>
        ) : results && Object.keys(results.series).length === 0 ? (
          <Section title="Model Results">
            <p className="text-gray-500 text-sm italic">No result data for this node.</p>
          </Section>
        ) : null}
      </div>

      {/* Fixed date range slider at panel bottom */}
      <YearRangeSlider
        years={years}
        startIdx={startIdx}
        endIdx={endIdx}
        onStartChange={setStartIdx}
        onEndChange={setEndIdx}
      />
    </div>
  );
}
