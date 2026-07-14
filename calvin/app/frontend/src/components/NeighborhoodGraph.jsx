import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchNeighborhood } from "../api/client.js";

// ---------------------------------------------------------------------------
// Color table — must match NetworkMap.jsx and ResultsChart.jsx COLORS
// ---------------------------------------------------------------------------

const NODE_COLOR = {
  "Surface Storage":     "#3b82f6",
  "Groundwater Storage": "#10b981",
  "Agricultural Demand": "#f59e0b",
  "Urban Demand":        "#f97316",
  "Water Treatment":     "#06b6d4",
  "Pump Plant":          "#8b5cf6",
  "Power Plant":         "#8b5cf6",
  "Non-Standard Demand": "#84cc16",
  "Junction":            "#6b7280",
};
const DEFAULT_NODE_COLOR = "#9ca3af";

// ---------------------------------------------------------------------------
// Layout — vertical: upstream (negative dist) at top, downstream at bottom
// ---------------------------------------------------------------------------

const NODE_R = 9;
const NODE_SPACING_X = 82;  // horizontal gap between nodes in the same row
const ROW_HEIGHT = 88;      // vertical gap between distance rows
const PAD_X = 50;
const PAD_Y = 28;
const LABEL_OFFSET = 13;    // px below circle center to baseline of label

function computeLayout(nodes) {
  if (!nodes || nodes.length === 0) return { positions: {}, svgW: 120, svgH: 100 };

  // Group nodes by their signed distance
  const byRow = new Map();
  for (const n of nodes) {
    if (!byRow.has(n.distance)) byRow.set(n.distance, []);
    byRow.get(n.distance).push(n);
  }

  // Rows sorted upstream→downstream (most negative first)
  const rows = Array.from(byRow.keys()).sort((a, b) => a - b);
  const maxInRow = Math.max(1, ...Array.from(byRow.values()).map((g) => g.length));

  const svgW = Math.max(PAD_X * 2 + (maxInRow - 1) * NODE_SPACING_X, 120);
  const svgH = PAD_Y + (rows.length - 1) * ROW_HEIGHT + NODE_R + LABEL_OFFSET + 14 + PAD_Y;

  const positions = {};
  rows.forEach((row, rowIdx) => {
    const group = byRow.get(row);
    const y = PAD_Y + NODE_R + rowIdx * ROW_HEIGHT;
    // Center the row horizontally
    const totalW = (group.length - 1) * NODE_SPACING_X;
    const startX = svgW / 2 - totalW / 2;
    group.forEach((n, colIdx) => {
      positions[n.prmname] = { x: startX + colIdx * NODE_SPACING_X, y };
    });
  });

  return { positions, svgW, svgH: Math.max(svgH, 100) };
}

// ---------------------------------------------------------------------------
// Arrow path — offset start/end to circle edges so line meets arrowhead cleanly
// ---------------------------------------------------------------------------

function arrowPath(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return null;
  const ux = dx / len;
  const uy = dy / len;
  const sx = x1 + ux * (NODE_R + 1);
  const sy = y1 + uy * (NODE_R + 1);
  const ex = x2 - ux * (NODE_R + 7); // 7 = room for arrowhead marker
  const ey = y2 - uy * (NODE_R + 7);
  return `M ${sx} ${sy} L ${ex} ${ey}`;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const MAX_NODES_WARNING = 80;

export default function NeighborhoodGraph({ prmname, onNodeClick }) {
  const [depth, setDepth] = useState(2);

  const { data, isLoading } = useQuery({
    queryKey: ["neighborhood", prmname, depth],
    queryFn: () => fetchNeighborhood(prmname, depth),
    enabled: !!prmname,
  });

  const layout = useMemo(() => {
    if (!data) return null;
    return computeLayout(data.nodes);
  }, [data]);

  if (!prmname) return null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-700 shrink-0">
        <span className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
          Network Graph
        </span>
        <div className="flex gap-1 ml-2">
          {[1, 2, 3].map((d) => (
            <button
              key={d}
              onClick={() => setDepth(d)}
              className={`text-[11px] px-2 py-0.5 rounded border transition-colors ${
                depth === d
                  ? "bg-blue-700 border-blue-500 text-white"
                  : "border-gray-600 text-gray-400 hover:border-gray-400 hover:text-gray-200"
              }`}
            >
              ±{d}
            </button>
          ))}
        </div>
        {data && (
          <span className="ml-auto text-[10px] text-gray-500 tabular-nums">
            {data.nodes.length}n · {data.links.length}e
          </span>
        )}
      </div>

      {/* Graph area — scrollable in both axes */}
      <div className="flex-1 overflow-auto bg-gray-950">
        {isLoading && <p className="text-gray-500 text-sm p-4">Loading…</p>}

        {data && data.nodes.length > MAX_NODES_WARNING && (
          <p className="text-yellow-500 text-xs px-3 py-1 shrink-0">
            Large neighborhood ({data.nodes.length} nodes). Try ±1 or ±2.
          </p>
        )}

        {data && data.nodes && data.links && layout && (
          <svg
            width={layout.svgW}
            height={layout.svgH}
            className="block"
            style={{ minWidth: "100%", minHeight: "100%" }}
          >
            <defs>
              <marker id="nbh-arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <polygon points="0 0, 7 3.5, 0 7" fill="#4b5563" />
              </marker>
              <marker id="nbh-arr-sel" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <polygon points="0 0, 7 3.5, 0 7" fill="#facc15" />
              </marker>
            </defs>

            {/* Distance-level row labels */}
            {Array.from(new Map(data.nodes.map((n) => [n.distance, n.distance])).values())
              .sort((a, b) => a - b)
              .map((dist, rowIdx) => {
                const label =
                  dist < 0 ? `↑ ${Math.abs(dist)} up` : dist > 0 ? `↓ ${dist} dn` : "focus";
                const y = PAD_Y + NODE_R + rowIdx * ROW_HEIGHT;
                return (
                  <text
                    key={dist}
                    x={6}
                    y={y}
                    fontSize={8}
                    fill={dist === 0 ? "#6b7280" : "#374151"}
                    dominantBaseline="middle"
                    fontStyle={dist === 0 ? "normal" : "italic"}
                  >
                    {label}
                  </text>
                );
              })}

            {/* Links */}
            {data.links.map((link) => {
              const from = layout.positions[link.origin];
              const to = layout.positions[link.terminus];
              if (!from || !to) return null;
              const isTouchingFocus = link.origin === prmname || link.terminus === prmname;
              const d = arrowPath(from.x, from.y, to.x, to.y);
              if (!d) return null;
              return (
                <path
                  key={link.prmname}
                  d={d}
                  fill="none"
                  stroke={isTouchingFocus ? "#facc15" : "#374151"}
                  strokeWidth={isTouchingFocus ? 1.5 : 1}
                  opacity={isTouchingFocus ? 0.9 : 0.55}
                  markerEnd={isTouchingFocus ? "url(#nbh-arr-sel)" : "url(#nbh-arr)"}
                />
              );
            })}

            {/* Nodes */}
            {data.nodes.map((n) => {
              const pos = layout.positions[n.prmname];
              if (!pos) return null;
              const isFocus = n.prmname === prmname;
              const color = NODE_COLOR[n.node_type] || DEFAULT_NODE_COLOR;
              const label = n.prmname.length > 14 ? n.prmname.slice(0, 13) + "…" : n.prmname;

              return (
                <g
                  key={n.prmname}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  onClick={() => onNodeClick(n.prmname)}
                  className="cursor-pointer"
                >
                  <title>{`${n.prmname}${n.description ? "\n" + n.description : ""}\n${n.node_type ?? ""}`}</title>
                  {isFocus && (
                    <circle r={NODE_R + 4} fill="none" stroke="#ffffff" strokeWidth={1.5} opacity={0.4} />
                  )}
                  <circle
                    r={NODE_R}
                    fill={color}
                    stroke={isFocus ? "#ffffff" : "#0f172a"}
                    strokeWidth={isFocus ? 2 : 0.8}
                    opacity={0.93}
                  />
                  <text
                    y={NODE_R + LABEL_OFFSET}
                    textAnchor="middle"
                    fontSize={9}
                    fill={isFocus ? "#e5e7eb" : "#9ca3af"}
                    fontWeight={isFocus ? "600" : "400"}
                  >
                    {label}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}
