import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

// ---------------------------------------------------------------------------
// Curve reconstruction
// ---------------------------------------------------------------------------

/**
 * Build the actual LP breakpoints for the COSVF penalty curve.
 *
 * Type 1 — Surface reservoir (quadratic, k_count segments):
 *   cost(s) = pmin + (pmax − pmin) × ((ub − s) / (ub − lb))²
 *   k_count+1 breakpoints evenly spaced in [lb, ub] — these are the exact
 *   storage levels used in the piecewise-linear LP approximation.
 *
 * Type 2 — Groundwater (linear, k_count=2):
 *   Flat penalty p at every storage level. GW ub is unbounded (1e12);
 *   use eop_init as the display upper bound so the chart is readable.
 */
function buildBreakpoints(type, lb, ub, params, eop_init, k_count) {
  if (type === 2) {
    const p = -(params.p ?? 0);
    const displayUb = (eop_init != null && eop_init > lb)
      ? eop_init * 1.2
      : lb + 1000;
    return [
      { storage: lb,        cost: p, isBreakpoint: true },
      { storage: displayUb, cost: p, isBreakpoint: true },
    ];
  }

  if (type === 1) {
    const { pmin = 0, pmax = 0 } = params;
    const range = ub - lb;
    if (range <= 0) return [];
    const n = k_count ?? 15;
    return Array.from({ length: n + 1 }, (_, k) => {
      const s = lb + (k / n) * range;
      const frac = (ub - s) / range;
      const cost = -(pmin + (pmax - pmin) * frac * frac);
      return { storage: s, cost, isBreakpoint: true };
    });
  }

  return [];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CosvfCurveChart({ type, lb, ub, eop_init, k_count, params }) {
  if (!params || Object.keys(params).length === 0) return null;
  if (lb == null || ub == null) return null;

  const data = buildBreakpoints(type, lb, ub, params, eop_init, k_count);
  if (data.length === 0) return null;

  const isLinear = type === 2;
  const typeLabel = isLinear ? "Groundwater (linear)" : "Surface Reservoir (quadratic)";
  const displayUb = data[data.length - 1].storage;

  const paramRows = Object.entries(params).map(([k, v]) => ({ k, v }));

  return (
    <div>
      {/* Parameter table */}
      <div className="flex flex-wrap gap-x-4 gap-y-0.5 mb-3 text-sm">
        {paramRows.map(({ k, v }) => (
          <div key={k} className="flex gap-1.5">
            <span className="text-gray-400 font-mono">{k}</span>
            <span className="text-blue-300 font-mono">{Math.abs(v).toFixed(1)}</span>
            <span className="text-gray-500 text-xs">$/TAF</span>
          </div>
        ))}
        <div className="flex gap-1.5">
          <span className="text-gray-400 font-mono">range</span>
          <span className="text-gray-300 font-mono">
            {lb.toFixed(0)}–{displayUb.toFixed(0)}
          </span>
          <span className="text-gray-500 text-xs">TAF</span>
        </div>
        {k_count != null && (
          <div className="flex gap-1.5">
            <span className="text-gray-400 font-mono">segments</span>
            <span className="text-gray-300 font-mono">{k_count}</span>
          </div>
        )}
      </div>

      {/* Penalty curve — marginal cost ($/TAF) vs. end-of-period storage */}
      <p className="text-[10px] text-gray-500 mb-1 uppercase tracking-wider">
        Marginal penalty vs. EOP Storage — {typeLabel}
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="storage"
            tick={{ fontSize: 9, fill: "#9ca3af" }}
            tickFormatter={(v) => Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}
            label={{
              value: "EOP Storage (TAF)",
              position: "insideBottom",
              offset: -4,
              style: { fontSize: 9, fill: "#6b7280" },
            }}
          />
          <YAxis
            tick={{ fontSize: 9, fill: "#9ca3af" }}
            tickFormatter={(v) => v.toFixed(0)}
            label={{
              value: "$/TAF",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 9, fill: "#6b7280" },
            }}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(17,24,39,0.85)",
              border: "1px solid #374151",
              fontSize: 11,
              backdropFilter: "blur(4px)",
            }}
            labelStyle={{ color: "#9ca3af" }}
            labelFormatter={(v) => `Storage: ${Number(v).toFixed(1)} TAF`}
            formatter={(v) => [`${Number(v).toFixed(1)} $/TAF`, "Penalty"]}
          />
          <Line
            type={isLinear ? "linear" : "monotone"}
            dataKey="cost"
            stroke="#60a5fa"
            dot={isLinear ? false : { r: 3, fill: "#60a5fa", stroke: "#1e3a5f", strokeWidth: 1 }}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      {!isLinear && (
        <p className="text-[10px] text-gray-600 mt-1">
          Dots = LP breakpoints ({k_count ?? 15} segments)
        </p>
      )}
    </div>
  );
}
