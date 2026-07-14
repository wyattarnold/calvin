import { useState } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Bound types we render as charts
const TIMESERIES_TYPES = new Set(["LBT", "UBT", "EQT"]);
const MONTHLY_TYPES    = new Set(["LBM", "UBM", "EQM"]);
const CONSTANT_TYPES   = new Set(["LBC", "UBC", "EQC"]);

const BOUND_COLORS = {
  LBT: "#10b981",  // green — lower bound
  UBT: "#ef4444",  // red — upper bound
  LBM: "#10b981",
  UBM: "#ef4444",
  EQT: "#f59e0b",
  EQM: "#f59e0b",
  LBC: "#10b981",
  UBC: "#ef4444",
  EQC: "#f59e0b",
};

const MONTH_ORDER = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Parse [[header], [date, val], ...] into [{date, val}, ...].
 * Skips the header row (first row where first element is a non-numeric string).
 */
function parseRows(rows) {
  if (!rows || rows.length < 2) return [];
  const start = typeof rows[0][0] === "string" && isNaN(rows[0][0]) ? 1 : 0;
  return rows.slice(start).map(([x, y]) => ({
    x: x,
    y: typeof y === "number" ? y : parseFloat(y),
  })).filter((r) => !isNaN(r.y));
}

// ---------------------------------------------------------------------------
// Monthly bounds chart (UBM / LBM) — 12 bars, one per calendar month
// ---------------------------------------------------------------------------

function MonthlyBoundsChart({ bounds }) {
  // Collect all monthly-type bounds, keyed by type
  const series = {};
  for (const b of bounds) {
    if (!MONTHLY_TYPES.has(b.type) || !b.bound) continue;
    const pts = parseRows(b.bound);
    if (pts.length === 0) continue;
    // Sort by calendar month order
    const sorted = MONTH_ORDER.map((m) => {
      const found = pts.find((p) => p.x === m);
      return { month: m, [b.type]: found ? found.y : null };
    });
    series[b.type] = sorted;
  }

  const types = Object.keys(series);
  if (types.length === 0) return null;

  // Merge all types onto the same month axis
  const data = MONTH_ORDER.map((month) => {
    const row = { month };
    for (const [type, pts] of Object.entries(series)) {
      const found = pts.find((p) => p.month === month);
      row[type] = found ? found[type] : null;
    }
    return row;
  });

  return (
    <div className="mb-4">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
        Monthly Bounds (TAF)
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 9, fill: "#6b7280" }}
          />
          <YAxis
            tick={{ fontSize: 9, fill: "#6b7280" }}
            tickFormatter={(v) => Math.abs(v) >= 1000 ? `${(v/1000).toFixed(0)}k` : v}
            label={{ value: "TAF", angle: -90, position: "insideLeft", style: { fontSize: 9, fill: "#4b5563" } }}
          />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 11 }}
            formatter={(v, name) => [`${Number(v).toFixed(1)} TAF`, name]}
          />
          <Legend
            formatter={(v) => <span style={{ fontSize: 10, color: "#9ca3af" }}>{v}</span>}
          />
          {types.map((type) => (
            <Bar key={type} dataKey={type} fill={BOUND_COLORS[type] ?? "#6b7280"} opacity={0.75} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Timeseries bounds chart (LBT / UBT) — date × value
// ---------------------------------------------------------------------------

function TimeseriesBoundsChart({ bounds, dateRange }) {
  const [hidden, setHidden] = useState(new Set());

  const series = {};
  for (const b of bounds) {
    if (!TIMESERIES_TYPES.has(b.type) || !b.bound) continue;
    const pts = parseRows(b.bound);
    if (pts.length === 0) continue;
    series[b.type] = pts;
  }

  const types = Object.keys(series);
  if (types.length === 0) return null;

  // Merge all types on shared date axis, filtered by dateRange
  const [startYear, endYear] = dateRange ?? [null, null];
  const dateMap = new Map();
  for (const [type, pts] of Object.entries(series)) {
    for (const { x: date, y: val } of pts) {
      const dateStr = String(date).slice(0, 10);
      const year = dateStr.slice(0, 4);
      if (startYear && year < startYear) continue;
      if (endYear   && year > endYear)   continue;
      if (!dateMap.has(dateStr)) dateMap.set(dateStr, { date: dateStr });
      dateMap.get(dateStr)[type] = val;
    }
  }
  const data = Array.from(dateMap.values()).sort((a, b) => a.date.localeCompare(b.date));
  if (data.length === 0) return null;

  function toggleType(type) {
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(type) ? next.delete(type) : next.add(type);
      return next;
    });
  }

  return (
    <div className="mb-4">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
        Timeseries Bounds (TAF)
      </p>

      {/* Toggleable legend */}
      <div className="flex gap-1 mb-2 flex-wrap">
        {types.map((type) => (
          <button
            key={type}
            onClick={() => toggleType(type)}
            className={`text-[10px] px-1.5 py-0.5 rounded border transition-opacity ${
              hidden.has(type) ? "opacity-25 border-gray-700" : "border-gray-600"
            }`}
            style={{ color: BOUND_COLORS[type] ?? "#9ca3af" }}
          >
            {type}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 8, fill: "#6b7280" }}
            tickFormatter={(v) => v.slice(0, 4)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 8, fill: "#6b7280" }}
            tickFormatter={(v) => Math.abs(v) >= 1000 ? `${(v/1000).toFixed(0)}k` : v}
            label={{ value: "TAF", angle: -90, position: "insideLeft", style: { fontSize: 8, fill: "#4b5563" } }}
          />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 11 }}
            formatter={(v, name) => [`${Number(v).toFixed(1)} TAF`, name]}
          />
          {types.map((type) => (
            <Line
              key={type}
              type="monotone"
              dataKey={type}
              stroke={BOUND_COLORS[type] ?? "#6b7280"}
              hide={hidden.has(type)}
              dot={false}
              strokeWidth={1.5}
              connectNulls
              activeDot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Constant bounds — just text
// ---------------------------------------------------------------------------

function ConstantBounds({ bounds }) {
  const items = bounds.filter(
    (b) => CONSTANT_TYPES.has(b.type) && b.bound != null
  );
  if (items.length === 0) return null;
  return (
    <div className="mb-3 space-y-1">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
        Constant Bounds (TAF)
      </p>
      {items.map((b, i) => (
        <div key={i} className="flex gap-2 text-sm">
          <span className="text-gray-400 w-10">{b.type}</span>
          <span style={{ color: BOUND_COLORS[b.type] ?? "#9ca3af" }} className="font-mono">
            {typeof b.bound === "number" ? b.bound.toFixed(1) : String(b.bound)} TAF
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export — renders all bound types for a node
// ---------------------------------------------------------------------------

export default function BoundsChart({ bounds, dateRange }) {
  if (!bounds || bounds.length === 0) return null;

  const hasTimeseries = bounds.some((b) => TIMESERIES_TYPES.has(b.type) && b.bound);
  const hasMonthly    = bounds.some((b) => MONTHLY_TYPES.has(b.type)    && b.bound);
  const hasConstant   = bounds.some((b) => CONSTANT_TYPES.has(b.type)   && b.bound != null);

  if (!hasTimeseries && !hasMonthly && !hasConstant) {
    return <p className="text-gray-500 text-sm italic">No bound data to chart</p>;
  }

  return (
    <div>
      <ConstantBounds bounds={bounds} />
      <MonthlyBoundsChart bounds={bounds} />
      <TimeseriesBoundsChart bounds={bounds} dateRange={dateRange} />
    </div>
  );
}
