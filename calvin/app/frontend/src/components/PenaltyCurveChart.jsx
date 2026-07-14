import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];

const MONTH_COLORS = [
  "#3b82f6","#06b6d4","#10b981","#84cc16",
  "#f59e0b","#f97316","#ef4444","#ec4899",
  "#8b5cf6","#6366f1","#0ea5e9","#14b8a6",
];

/**
 * Parse raw breakpoints and compute marginal cost ($/TAF) for each tier.
 *
 * Raw format: [[header], [cap1, cost1], [cap2, cost2], ...]
 * Convention: cumulative_cost DECREASES as cumulative_capacity increases.
 * Last breakpoint: (full_demand, 0.0)
 *
 * Marginal cost of tier i (from cap[i-1] to cap[i]):
 *   mc = (cost[i-1] - cost[i]) / (cap[i] - cap[i-1])   [$/ TAF]
 *
 * We plot one point per tier boundary at x = cap[i], y = mc.
 * The first tier uses the origin (0, cost[0]) as its left edge.
 */
function toMarginalCost(rows) {
  if (!rows || rows.length < 2) return [];

  // Skip header row if first element is a non-numeric string
  const start = typeof rows[0][0] === "string" && isNaN(rows[0][0]) ? 1 : 0;
  const pts = rows.slice(start).map(([cap, cost]) => ({
    cap: typeof cap === "number" ? cap : parseFloat(cap),
    cost: typeof cost === "number" ? cost : parseFloat(cost),
  })).filter((r) => !isNaN(r.cap) && !isNaN(r.cost));

  if (pts.length < 2) return [];

  const result = [];
  // First point: marginal cost from 0 to pts[0].cap
  // Use pts[0].cost / pts[0].cap as an approximation of the first tier
  // (The full range starts from 0 delivery at pts[0].cost total cost)
  for (let i = 1; i < pts.length; i++) {
    const dCap = pts[i].cap - pts[i - 1].cap;
    const dCost = pts[i - 1].cost - pts[i].cost; // cost decreases → dCost > 0
    if (dCap <= 0) continue;
    result.push({
      capacity: pts[i].cap,
      marginalCost: dCost / dCap,
    });
  }
  return result;
}

/**
 * Merge all months' marginal cost curves onto a shared capacity x-axis.
 */
function buildMarginalData(costs) {
  const monthData = {};

  for (const month of MONTHS) {
    const rows = costs[month];
    if (!rows) continue;
    const pts = toMarginalCost(rows);
    if (pts.length === 0) continue;
    monthData[month] = pts;
  }

  const presentMonths = Object.keys(monthData);
  if (presentMonths.length === 0) return { data: [], months: [] };

  // Collect all unique capacity breakpoints
  const capSet = new Set();
  for (const pts of Object.values(monthData)) {
    pts.forEach((p) => capSet.add(p.capacity));
  }
  const caps = Array.from(capSet).sort((a, b) => a - b);

  // For each capacity, interpolate/assign each month's marginal cost
  const data = caps.map((cap) => {
    const row = { capacity: cap };
    for (const [month, pts] of Object.entries(monthData)) {
      // Find the marginal cost tier that ends at or just after this capacity
      const exact = pts.find((p) => p.capacity === cap);
      if (exact) {
        row[month] = exact.marginalCost;
      } else {
        // Step function — use the cost of the tier whose right edge <= cap
        const lo = [...pts].reverse().find((p) => p.capacity <= cap);
        row[month] = lo ? lo.marginalCost : null;
      }
    }
    return row;
  });

  return { data, months: presentMonths };
}

/**
 * Return raw breakpoints as {capacity, totalCost} — the integrated (total $) curve.
 * Each breakpoint is (cumulative_delivery, total_penalty) in the raw data.
 */
function toTotalCost(rows) {
  if (!rows || rows.length < 2) return [];
  const start = typeof rows[0][0] === "string" && isNaN(rows[0][0]) ? 1 : 0;
  return rows.slice(start).map(([cap, cost]) => ({
    capacity: typeof cap === "number" ? cap : parseFloat(cap),
    totalCost: typeof cost === "number" ? cost : parseFloat(cost),
  })).filter((r) => !isNaN(r.capacity) && !isNaN(r.totalCost));
}

function buildTotalData(costs) {
  const monthData = {};
  for (const month of MONTHS) {
    const rows = costs[month];
    if (!rows) continue;
    const pts = toTotalCost(rows);
    if (pts.length === 0) continue;
    monthData[month] = pts;
  }

  const presentMonths = Object.keys(monthData);
  if (presentMonths.length === 0) return { data: [], months: [] };

  const capSet = new Set();
  for (const pts of Object.values(monthData)) {
    pts.forEach((p) => capSet.add(p.capacity));
  }
  const caps = Array.from(capSet).sort((a, b) => a - b);

  const data = caps.map((cap) => {
    const row = { capacity: cap };
    for (const [month, pts] of Object.entries(monthData)) {
      const exact = pts.find((p) => p.capacity === cap);
      if (exact) {
        row[month] = exact.totalCost;
      } else {
        const lo = [...pts].reverse().find((p) => p.capacity <= cap);
        row[month] = lo ? lo.totalCost : null;
      }
    }
    return row;
  });

  return { data, months: presentMonths };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function PenaltyCurveChart({ costs, costType, constantCost, label }) {
  const [hidden, setHidden] = useState(new Set());
  const [integrated, setIntegrated] = useState(false);
  const [logScale, setLogScale] = useState(false);

  if (!costType || costType === "NONE" || costType === "None") {
    return <p className="text-gray-500 text-sm italic">No penalty curve</p>;
  }

  if (costType === "Constant") {
    return (
      <div>
        {label && <p className="text-xs text-gray-400 mb-1 font-mono truncate" title={label}>{label}</p>}
        <p className="text-sm text-gray-300">
          Constant:{" "}
          <span className="text-blue-400 font-mono">${constantCost?.toFixed(2) ?? "—"}</span> / TAF
        </p>
      </div>
    );
  }

  if (!costs || Object.keys(costs).length === 0) {
    return <p className="text-gray-500 text-sm italic">No curve data</p>;
  }

  const { data: marginalData, months: marginalMonths } = buildMarginalData(costs);
  const { data: totalData,    months: totalMonths }    = buildTotalData(costs);
  const data   = integrated ? totalData   : marginalData;
  const months = integrated ? totalMonths : marginalMonths;
  if (data.length === 0) return null;

  function toggleMonth(month) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(month)) next.delete(month);
      else next.add(month);
      return next;
    });
  }

  return (
    <div>
      {/* Label + Integrate toggle */}
      <div className="flex items-center justify-between mb-1">
        {label && (
          <p className="text-xs text-gray-400 font-mono truncate" title={label}>{label}</p>
        )}
        <div className="flex gap-1 shrink-0 ml-2">
          <button
            onClick={() => setIntegrated((v) => !v)}
            className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
              integrated
                ? "border-blue-500 text-blue-400 bg-blue-950"
                : "border-gray-600 text-gray-400 hover:border-blue-400 hover:text-blue-400"
            }`}
          >
            Integrate
          </button>
          <button
            onClick={() => setLogScale((v) => !v)}
            className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
              logScale
                ? "border-blue-500 text-blue-400 bg-blue-950"
                : "border-gray-600 text-gray-400 hover:border-blue-400 hover:text-blue-400"
            }`}
          >
            Log
          </button>
        </div>
      </div>

      {/* Toggleable month legend */}
      <div className="flex flex-wrap gap-1 mb-2">
        {months.map((month) => {
          const idx = MONTHS.indexOf(month);
          const isHidden = hidden.has(month);
          return (
            <button
              key={month}
              onClick={() => toggleMonth(month)}
              title={isHidden ? "Show" : "Hide"}
              className={`text-[10px] px-1.5 py-0.5 rounded border transition-opacity ${
                isHidden ? "opacity-25 border-gray-700" : "border-gray-600"
              }`}
              style={{ color: MONTH_COLORS[idx >= 0 ? idx : 0] }}
            >
              {month}
            </button>
          );
        })}
      </div>

      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="capacity"
            tick={{ fontSize: 9, fill: "#9ca3af" }}
            tickFormatter={(v) => v.toFixed(1)}
            label={{
              value: "Delivery (TAF)",
              position: "insideBottom",
              offset: -4,
              style: { fontSize: 9, fill: "#6b7280" },
            }}
          />
          <YAxis
            scale={logScale ? "log" : "auto"}
            domain={logScale ? [1, "auto"] : undefined}
            allowDataOverflow={logScale}
            tick={{ fontSize: 9, fill: "#9ca3af" }}
            tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}
            label={{
              value: integrated ? "$" : "$/TAF",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 9, fill: "#6b7280" },
            }}
          />
          <Tooltip
            contentStyle={{ background: "rgba(17,24,39,0.72)", border: "1px solid #374151", fontSize: 11, backdropFilter: "blur(4px)" }}
            labelStyle={{ color: "#9ca3af" }}
            labelFormatter={(v) => `Delivery: ${Number(v).toFixed(2)} TAF`}
            formatter={(v, name) => [
              integrated ? `$${Number(v).toFixed(0)}` : `$${Number(v).toFixed(0)}/TAF`,
              name,
            ]}
          />
          {months.map((month) => {
            const idx = MONTHS.indexOf(month);
            return (
              <Line
                key={month}
                type={integrated ? "linear" : "stepAfter"}
                dataKey={month}
                stroke={MONTH_COLORS[idx >= 0 ? idx : 0]}
                hide={hidden.has(month)}
                dot={false}
                strokeWidth={1.5}
                connectNulls={false}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
