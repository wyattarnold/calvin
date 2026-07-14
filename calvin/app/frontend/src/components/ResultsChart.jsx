import { useState, useMemo } from "react";
import {
  ComposedChart,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const COLORS = [
  "#3b82f6","#10b981","#f59e0b","#ef4444",
  "#8b5cf6","#06b6d4","#f97316","#84cc16",
];

// Colors for constant bound reference lines (LBC/UBC/EQC)
const REF_COLORS = { LBC: "#10b981", UBC: "#ef4444", EQC: "#f59e0b" };

// Colors for time-varying / monthly bound series injected into storage chart
const BOUND_SERIES_COLORS = {
  bound_LBT: "#10b981", bound_LBM: "#10b981",
  bound_UBT: "#ef4444", bound_UBM: "#ef4444",
  bound_EQT: "#f59e0b", bound_EQM: "#f59e0b",
};

function seriesColor(key, index) {
  return BOUND_SERIES_COLORS[key] ?? COLORS[index % COLORS.length];
}

function shortKey(key) {
  if (key === "bound_LBT" || key === "bound_LBM") return "Lower Bound";
  if (key === "bound_UBT" || key === "bound_UBM") return "Upper Bound";
  if (key === "bound_EQT" || key === "bound_EQM") return "Eq Bound";
  if (key.startsWith("dual_node_")) return "Node λ";
  if (key.startsWith("dual_lower_")) return "(lb) " + key.slice(11);
  if (key.startsWith("dual_upper_")) return "(ub) " + key.slice(11);
  if (key === "evaporation") return "(out) evaporation";
  if (key.startsWith("flow_out_")) return "(out) " + key.slice(9);
  if (key.startsWith("flow_in_"))  return "(in) "  + key.slice(8);
  if (key.startsWith("shortage_volume_")) return "ShortVol " + key.slice(16);
  if (key.startsWith("shortage_cost_")) return "ShortCost " + key.slice(14);
  if (key.startsWith("op_cost_")) return "OpCost " + key.slice(8);
  if (key.startsWith("pwp_short_volume_")) return "PWP Vol " + key.slice(17);
  if (key.startsWith("pwp_short_cost_")) return "PWP Cost " + key.slice(15);
  return key;
}

/**
 * Build merged Recharts data from a map of series, filtered to a date range.
 * @param {Record<string, [string|number, number][]>} seriesMap
 * @param {[string, string]|null} dateRange  - [startYear, endYear] inclusive
 */
function buildChartData(seriesMap, dateRange, negateKeys = null) {
  const [startYear, endYear] = dateRange ?? [null, null];
  const dateMap = new Map();

  for (const [key, rows] of Object.entries(seriesMap)) {
    const start = typeof rows[0]?.[0] === "string" && isNaN(rows[0][0]) && rows[0][1] === undefined ? 1 : 0;
    const sign = negateKeys?.has(key) ? -1 : 1;
    for (const row of rows.slice(start)) {
      if (!Array.isArray(row) || row.length < 2) continue;
      const [dateRaw, value] = row;
      const dateKey = String(dateRaw).slice(0, 10);
      const year = dateKey.slice(0, 4);
      if (startYear && year < startYear) continue;
      if (endYear && year > endYear) continue;
      if (!dateMap.has(dateKey)) dateMap.set(dateKey, { date: dateKey });
      const num = typeof value === "number" ? value : Number(value);
      dateMap.get(dateKey)[key] = sign * num;
    }
  }

  return Array.from(dateMap.values()).sort((a, b) => a.date.localeCompare(b.date));
}

const STORAGE_NODE_TYPES = new Set(["Surface Storage", "Groundwater Storage"]);

// Month name → 2-digit month number string for date matching
const MONTH_TO_NUM = {
  JAN: "01", FEB: "02", MAR: "03", APR: "04", MAY: "05", JUN: "06",
  JUL: "07", AUG: "08", SEP: "09", OCT: "10", NOV: "11", DEC: "12",
};

/**
 * Convert time-varying and monthly bounds into synthetic series rows
 * so they can be overlaid on the storage results chart.
 * @param {Array} bounds - node.bounds from API
 * @param {Array} resultDates - sorted date strings ("YYYY-MM-DD") from storage series
 */
function buildStorageBoundSeries(bounds, resultDates) {
  const result = {};
  for (const b of bounds) {
    if (!b.bound || !Array.isArray(b.bound)) continue;

    if (b.type === "LBT" || b.type === "UBT" || b.type === "EQT") {
      // Timeseries bound — add rows directly; buildChartData will align on date
      result[`bound_${b.type}`] = b.bound;
    } else if (b.type === "LBM" || b.type === "UBM" || b.type === "EQM") {
      // Monthly bound — expand to one row per result date using the month lookup
      const monthVals = {};
      for (const [month, val] of b.bound) {
        const num = MONTH_TO_NUM[month]; // ignore header rows that don't match
        if (num) monthVals[num] = typeof val === "number" ? val : Number(val);
      }
      if (Object.keys(monthVals).length === 0) continue;
      const rows = resultDates
        .map((d) => [d, monthVals[d.slice(5, 7)] ?? null])
        .filter(([, v]) => v !== null && !isNaN(v));
      if (rows.length > 0) result[`bound_${b.type}`] = rows;
    }
  }
  return result;
}

function groupSeries(series, nodeType) {
  const isStorageNode = STORAGE_NODE_TYPES.has(nodeType);
  const groups = { storage: {}, shortage: {}, flow: {}, cost: {}, dual_node: {}, dual_lower: {}, dual_upper: {}, other: {} };
  for (const [key, data] of Object.entries(series)) {
    if (key === "storage") groups.storage[key] = data;
    else if (key === "evaporation") {
      if (isStorageNode) groups.flow[key] = data;
      else groups.storage[key] = data;
    }
    else if (key.startsWith("shortage_volume") || key.startsWith("pwp_short_volume")) groups.shortage[key] = data;
    else if (key.startsWith("flow_in") || key.startsWith("flow_out")) groups.flow[key] = data;
    else if (key.startsWith("shortage_cost") || key.startsWith("op_cost") || key.startsWith("pwp_short_cost")) groups.cost[key] = data;
    else if (key.startsWith("dual_node_")) groups.dual_node[key] = data;
    else if (key.startsWith("dual_lower_")) groups.dual_lower[key] = data;
    else if (key.startsWith("dual_upper_")) groups.dual_upper[key] = data;
    else groups.other[key] = data;
  }
  return groups;
}

// ---------------------------------------------------------------------------
// Toggleable legend
// ---------------------------------------------------------------------------

function ToggleLegend({ keys, hidden, onToggle }) {
  if (keys.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-1 mb-2">
      {keys.map((key, i) => {
        const isHidden = hidden.has(key);
        const isBound = key.startsWith("bound_");
        return (
          <button
            key={key}
            onClick={() => onToggle(key)}
            title={isHidden ? "Click to show" : "Click to hide"}
            className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border transition-opacity ${
              isHidden
                ? "opacity-30 border-gray-700 text-gray-600"
                : "opacity-100 border-gray-600 text-gray-300"
            }`}
          >
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 2,
                background: seriesColor(key, i),
                borderRadius: 1,
                backgroundImage: isBound ? `repeating-linear-gradient(90deg, ${seriesColor(key, i)} 0 4px, transparent 4px 7px)` : undefined,
              }}
            />
            {shortKey(key)}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Custom tooltip — filters _top_ stacking keys, semi-transparent background
// ---------------------------------------------------------------------------

function ChartTooltip({ active, payload, label, yUnit = "" }) {
  if (!active || !payload?.length) return null;
  const items = payload.filter((p) => !String(p.dataKey).startsWith("_top_"));
  if (items.length === 0) return null;
  return (
    <div style={{
      background: "rgba(17, 24, 39, 0.72)",
      border: "1px solid #374151",
      borderRadius: 4,
      fontSize: 11,
      padding: "6px 10px",
      backdropFilter: "blur(4px)",
    }}>
      <p style={{ color: "#9ca3af", marginBottom: 4 }}>{label}</p>
      {items.map((item) => (
        <p key={item.dataKey} style={{ color: item.color, margin: "1px 0" }}>
          {shortKey(String(item.dataKey))}: {typeof item.value === "number" ? item.value.toFixed(2) : item.value}{yUnit}
        </p>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single chart
// ---------------------------------------------------------------------------

function Chart({ title, series, yLabel, yUnit = "", dateRange, refBounds = [], stacked = false, negateKeys = null }) {
  const [hidden, setHidden] = useState(new Set());
  const [collapsed, setCollapsed] = useState(false);

  // All hooks must run unconditionally before any early returns.
  const keys = Object.keys(series);
  const data = buildChartData(series, dateRange, negateKeys);
  const visibleKeys = keys.filter((k) => !hidden.has(k));

  const allVals = data.flatMap((row) =>
    visibleKeys.map((k) => row[k]).filter((v) => v != null && !isNaN(v))
  );
  const meanVal = allVals.length ? allVals.reduce((a, b) => a + b, 0) / allVals.length : 0;
  const yDomain = useMemo(() => {
    if (allVals.length === 0) return ["auto", "auto"];
    const mn = Math.min(...allVals);
    const mx = Math.max(...allVals);
    const pad = (mx - mn) * 0.05 || Math.abs(mx) * 0.05 || 1;
    return [mn - pad, mx + pad];
  }, [hidden, data]); // eslint-disable-line react-hooks/exhaustive-deps

  // Pre-compute stacked top-edge positions for each series so we can render
  // outline Lines in reverse order (lowest block's line on top in SVG z-order).
  const stackedLineData = useMemo(() => {
    if (!stacked || data.length === 0) return data;
    const inKeys = keys.filter((k) => !negateKeys?.has(k));
    const outKeys = keys.filter((k) => negateKeys?.has(k));
    return data.map((row) => {
      const extra = {};
      let acc = 0;
      for (const k of inKeys) {
        acc += hidden.has(k) ? 0 : (row[k] ?? 0);
        extra[`_top_${k}`] = acc;
      }
      acc = 0;
      for (const k of outKeys) {
        acc += hidden.has(k) ? 0 : (row[k] ?? 0);
        extra[`_top_${k}`] = acc;
      }
      return { ...row, ...extra };
    });
  }, [stacked, data, keys, negateKeys, hidden]); // eslint-disable-line react-hooks/exhaustive-deps

  // One tick position per unique year (first data point of that year).
  const yearTicks = useMemo(() => {
    const seen = new Set();
    return data
      .filter((row) => { const y = row.date.slice(0, 4); if (seen.has(y)) return false; seen.add(y); return true; })
      .map((row) => row.date);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

  if (keys.length === 0) return null;
  if (data.length === 0) return null;

  function toggleSeries(key) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div className="mb-5">
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex items-center gap-1 w-full text-left text-xs font-semibold uppercase tracking-wider text-gray-500 hover:text-gray-300 mb-1"
      >
        <span className="text-gray-600">{collapsed ? "▸" : "▾"}</span>
        {title}
      </button>

      {!collapsed && <ToggleLegend
        keys={keys}
        hidden={hidden}
        onToggle={toggleSeries}
      />}

      {!collapsed && <ResponsiveContainer width="100%" height={180}>
        {stacked ? (
          <ComposedChart data={stackedLineData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="date"
              ticks={yearTicks}
              tick={{ fontSize: 8, fill: "#6b7280" }}
              tickFormatter={(v) => v.slice(0, 4)}
            />
            <YAxis
              domain={yDomain}
              tick={{ fontSize: 8, fill: "#6b7280" }}
              tickFormatter={(v) =>
                Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)
              }
              label={
                yLabel
                  ? { value: yLabel, angle: -90, position: "insideLeft", style: { fontSize: 8, fill: "#4b5563" } }
                  : undefined
              }
            />
            <Tooltip content={<ChartTooltip yUnit={yUnit} />} />
            <ReferenceLine y={0} stroke="#4b5563" strokeWidth={1} />
            {data
              .filter((row) => row.date.slice(5, 7) === "09")
              .map((row) => (
                <ReferenceLine key={`oct-${row.date}`} x={row.date} stroke="#374151" strokeWidth={0.5} opacity={0.7} />
              ))}
            {/* Pass 1: fills only (no stroke), in stack order */}
            {keys.map((key, i) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke="none"
                fill={seriesColor(key, i)}
                fillOpacity={0.25}
                stackId={negateKeys?.has(key) ? "out" : "in"}
                hide={hidden.has(key)}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            ))}
            {/* Pass 2: outlines only (no fill), in REVERSE order so lower blocks' lines are on top */}
            {[...keys].reverse().map((key) => {
              const i = keys.indexOf(key);
              return (
                <Line
                  key={`_line_${key}`}
                  type="monotone"
                  dataKey={`_top_${key}`}
                  stroke={seriesColor(key, i)}
                  strokeWidth={1}
                  dot={false}
                  hide={hidden.has(key)}
                  connectNulls
                  activeDot={false}
                  isAnimationActive={false}
                />
              );
            })}
          </ComposedChart>
        ) : (
          <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="date"
              ticks={yearTicks}
              tick={{ fontSize: 8, fill: "#6b7280" }}
              tickFormatter={(v) => v.slice(0, 4)}
            />
            <YAxis
              domain={yDomain}
              tick={{ fontSize: 8, fill: "#6b7280" }}
              tickFormatter={(v) =>
                Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)
              }
              label={
                yLabel
                  ? { value: yLabel, angle: -90, position: "insideLeft", style: { fontSize: 8, fill: "#4b5563" } }
                  : undefined
              }
            />
            <Tooltip content={<ChartTooltip yUnit={yUnit} />} />
            {/* Light vertical lines at each October (water-year boundary) */}
            {data
              .filter((row) => row.date.slice(5, 7) === "09")
              .map((row) => (
                <ReferenceLine key={`oct-${row.date}`} x={row.date} stroke="#374151" strokeWidth={0.5} opacity={0.7} />
              ))}
            {meanVal > 0 && (
              <ReferenceLine
                y={meanVal}
                stroke="#374151"
                strokeDasharray="4 2"
                label={{ value: "avg", position: "right", fontSize: 8, fill: "#4b5563" }}
              />
            )}
            {keys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={seriesColor(key, i)}
                hide={hidden.has(key)}
                dot={false}
                strokeWidth={key.startsWith("bound_") ? 1 : 1.5}
                strokeDasharray={key.startsWith("bound_") ? "5 3" : undefined}
                connectNulls
                activeDot={{ r: 3 }}
              />
            ))}
            {refBounds.map((b) => (
              <ReferenceLine
                key={b.type}
                y={b.bound}
                stroke={REF_COLORS[b.type] ?? "#6b7280"}
                strokeDasharray="6 3"
                strokeWidth={1.5}
                label={{
                  value: `${b.type} ${b.bound}`,
                  position: "insideTopRight",
                  fontSize: 8,
                  fill: REF_COLORS[b.type] ?? "#6b7280",
                }}
              />
            ))}
          </LineChart>
        )}
      </ResponsiveContainer>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export default function ResultsChart({ series, nodeType, dateRange, bounds }) {
  const groups = groupSeries(series, nodeType);

  // For storage nodes, inject time-varying and monthly bounds as dashed overlay series
  if (STORAGE_NODE_TYPES.has(nodeType) && bounds && groups.storage["storage"]) {
    const storageRows = groups.storage["storage"];
    const startIdx = typeof storageRows[0]?.[0] === "string" && isNaN(storageRows[0][0]) && storageRows[0][1] === undefined ? 1 : 0;
    const resultDates = storageRows.slice(startIdx).map(([d]) => String(d).slice(0, 10));
    Object.assign(groups.storage, buildStorageBoundSeries(bounds, resultDates));
  }

  // Extract constant storage bounds (LBC/UBC/EQC with numeric values) for reference lines
  const storageBounds = (bounds ?? []).filter(
    (b) => Object.prototype.hasOwnProperty.call(REF_COLORS, b.type) && typeof b.bound === "number"
  );

  // Outflow keys get negated so they plot below zero in the flow chart
  const flowNegateKeys = new Set(
    Object.keys(groups.flow).filter((k) => k.startsWith("flow_out_") || k === "evaporation")
  );

  return (
    <div>
      <Chart title="Storage" series={groups.storage} yLabel="TAF" yUnit=" TAF" dateRange={dateRange} refBounds={storageBounds} />
      <Chart title="Shortage Volume" series={groups.shortage} yLabel="TAF" yUnit=" TAF" dateRange={dateRange} />
      <Chart title="Flow" series={groups.flow} yLabel="TAF" yUnit=" TAF" dateRange={dateRange} stacked negateKeys={flowNegateKeys} />
      <Chart title="Cost" series={groups.cost} yLabel="$" yUnit=" $" dateRange={dateRange} />
      <Chart title="Dual — Node" series={groups.dual_node} yLabel="$/TAF" yUnit=" $/TAF" dateRange={dateRange} />
      <Chart title="Dual — Lower Bound" series={groups.dual_lower} yLabel="$/TAF" yUnit=" $/TAF" dateRange={dateRange} />
      <Chart title="Dual — Upper Bound" series={groups.dual_upper} yLabel="$/TAF" yUnit=" $/TAF" dateRange={dateRange} />
      <Chart title="Other" series={groups.other} dateRange={dateRange} />
    </div>
  );
}
