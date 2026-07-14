import { useEffect, useRef, useMemo } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import { useQuery } from "@tanstack/react-query";
import L from "leaflet";
import { fetchNetwork } from "../api/client.js";

// Fix Leaflet default icon paths broken by Vite bundling
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// ---------------------------------------------------------------------------
// Visual style tables
// ---------------------------------------------------------------------------

// Node styles: color, radius (r), stroke weight (w), fillOpacity (fo)
const NODE_STYLE = {
  "Surface Storage":     { color: "#3b82f6", r: 9,  w: 1.5, fo: 0.9 },
  "Groundwater Storage": { color: "#10b981", r: 8,  w: 1.5, fo: 0.9 },
  "Agricultural Demand": { color: "#f59e0b", r: 6,  w: 1.2, fo: 0.85 },
  "Urban Demand":        { color: "#f97316", r: 6,  w: 1.2, fo: 0.85 },
  "Water Treatment":     { color: "#22d3ee", r: 5,  w: 1,   fo: 0.8 },
  "Pump Plant":          { color: "#eab308", r: 5,  w: 1.5, fo: 0.85 },
  "Power Plant":         { color: "#a855f7", r: 5,  w: 1,   fo: 0.85 },
  "Non-Standard Demand": { color: "#84cc16", r: 5,  w: 1,   fo: 0.8 },
  "Junction":            { color: "#6b7280", r: 3,  w: 0.6, fo: 0.65 },
};

const DEFAULT_NODE_STYLE = { color: "#9ca3af", r: 3, w: 0.5, fo: 0.6 };

// Link colors by terminus node type — same palette as ResultsChart.jsx COLORS array
// so map links and chart series share a consistent visual language.
const LINK_COLOR_BY_TERMINUS = {
  "Junction":            "#3b82f6",   // blue  — transit (index 0)
  "Groundwater Storage": "#10b981",   // green — GW recharge (index 1)
  "Agricultural Demand": "#f59e0b",   // amber — irrigation (index 2)
  "Surface Storage":     "#ef4444",   // red   — filling a reservoir (index 3)
  "Pump Plant":          "#8b5cf6",   // violet— pump intake (index 4)
  "Power Plant":         "#8b5cf6",   // violet— hydro (index 4)
  "Water Treatment":     "#06b6d4",   // cyan  — to WTP (index 5)
  "Urban Demand":        "#f97316",   // orange— urban supply (index 6)
  "Non-Standard Demand": "#84cc16",   // lime  — (index 7)
};

const DEFAULT_LINK_COLOR = "#3b82f6"; // blue fallback

function linkColor(terminusType, originType) {
  return LINK_COLOR_BY_TERMINUS[terminusType] ?? LINK_COLOR_BY_TERMINUS[originType] ?? DEFAULT_LINK_COLOR;
}

// ---------------------------------------------------------------------------
// Style functions
// ---------------------------------------------------------------------------

function nodeFeatureStyle(feature, selectedNode, shortageMap = null, debugNodeSet = null) {
  const { prmname, node_type, disabled } = feature.properties;
  const isSelected = prmname === selectedNode;
  const ns = NODE_STYLE[node_type] || DEFAULT_NODE_STYLE;

  if (shortageMap !== null) {
    const hasShortage = (shortageMap[prmname] ?? 0) > 0;
    return {
      radius: isSelected ? ns.r + 4 : ns.r,
      fillColor: hasShortage ? "#ef4444" : "#374151",
      color: isSelected ? "#ffffff" : "#111827",
      weight: isSelected ? 2.5 : ns.w,
      opacity: 1,
      fillOpacity: hasShortage ? 0.9 : 0.2,
    };
  }

  if (debugNodeSet !== null) {
    const isDebug = debugNodeSet.has(prmname);
    return {
      radius: isSelected ? ns.r + 4 : isDebug ? ns.r + 2 : ns.r,
      fillColor: isDebug ? "#ec4899" : "#374151",
      color: isSelected ? "#ffffff" : isDebug ? "#ec4899" : "#111827",
      weight: isSelected ? 2.5 : isDebug ? 2 : ns.w,
      opacity: 1,
      fillOpacity: isDebug ? 0.9 : 0.2,
    };
  }

  return {
    radius: isSelected ? ns.r + 4 : ns.r,
    fillColor: ns.color,
    color: isSelected ? "#ffffff" : "#111827",
    weight: isSelected ? 2.5 : ns.w,
    opacity: disabled ? 0.3 : 1,
    fillOpacity: disabled ? 0.15 : isSelected ? 1 : ns.fo,
  };
}

function linkFeatureStyle(feature, selectedNode, debugLinkSet = null) {
  const { prmname, origin, terminus, origin_type, terminus_type, disabled } = feature.properties;
  const isSelected = origin === selectedNode || terminus === selectedNode;
  const isDebug = debugLinkSet !== null && debugLinkSet.has(prmname);
  const color = isSelected ? "#facc15" : isDebug ? "#ec4899" : linkColor(terminus_type, origin_type);
  return {
    color,
    weight: isSelected ? 2.5 : isDebug ? 2.5 : 1,
    opacity: disabled ? 0.1 : isSelected ? 0.95 : isDebug ? 0.9 : 0.4,
    dashArray: isDebug ? "6 4" : null,
  };
}

function pointToLayer(feature, latlng) {
  return L.circleMarker(latlng);
}

// ---------------------------------------------------------------------------
// Keep Leaflet's internal size in sync with the container's actual CSS size.
// Without this, flyTo centres on a stale (wrong) viewport size when panels
// open/close and the map div shrinks or grows.
// ---------------------------------------------------------------------------

function MapResizeHandler() {
  const map = useMap();
  useEffect(() => {
    const container = map.getContainer();
    const observer = new ResizeObserver(() => {
      map.invalidateSize({ animate: false });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [map]);
  return null;
}

// ---------------------------------------------------------------------------
// Fly-to helper — pans the map when a node is selected from outside the map
// ---------------------------------------------------------------------------

function MapFlyTo({ prmname, geojson }) {
  const map = useMap();

  useEffect(() => {
    if (!prmname || !geojson) return;
    const feature = geojson.features.find(
      (f) => f.properties.feature_class === "node" && f.properties.prmname === prmname
    );
    if (!feature || feature.geometry?.type !== "Point") return;
    const [lng, lat] = feature.geometry.coordinates;

    // Wait two animation frames so the panel layout finishes and the
    // ResizeObserver fires before we compute the map centre.
    let raf1, raf2;
    raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        map.invalidateSize({ animate: false });
        map.flyTo([lat, lng], Math.max(map.getZoom(), 9), { duration: 0.8 });
      });
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [prmname]); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
}

// ---------------------------------------------------------------------------
// Inner layer — mounted once, styles updated via useEffect
// ---------------------------------------------------------------------------

function GeoJSONLayer({ geojson, selectedNode, onNodeClick, onLinkClick, shortageMap, debugLinkSet, debugNodeSet }) {
  const layerRef = useRef(null);
  const map = useMap();

  // Fit California bounds on first load
  useEffect(() => {
    if (layerRef.current) {
      try {
        const bounds = layerRef.current.getBounds();
        if (bounds.isValid()) map.fitBounds(bounds, { padding: [20, 20] });
      } catch {}
    }
  }, [geojson]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reactively update styles without remounting when selection or mode changes
  useEffect(() => {
    if (!layerRef.current) return;
    layerRef.current.eachLayer((layer) => {
      if (!layer.feature) return;
      const fc = layer.feature.properties.feature_class;
      if (fc === "node" && layer.setStyle) {
        layer.setStyle(nodeFeatureStyle(layer.feature, selectedNode, shortageMap, debugNodeSet));
        if (layer.feature.properties.prmname === selectedNode && layer.bringToFront) {
          layer.bringToFront();
        }
      } else if (fc === "link" && layer.setStyle) {
        layer.setStyle(linkFeatureStyle(layer.feature, selectedNode, debugLinkSet));
      }
    });
  }, [selectedNode, shortageMap, debugLinkSet, debugNodeSet]);

  if (!geojson) return null;

  return (
    <GeoJSON
      ref={layerRef}
      data={geojson}
      style={(feature) => {
        const fc = feature.properties.feature_class;
        if (fc === "node") return nodeFeatureStyle(feature, selectedNode, shortageMap, debugNodeSet);
        return linkFeatureStyle(feature, selectedNode, debugLinkSet);
      }}
      pointToLayer={pointToLayer}
      onEachFeature={(feature, layer) => {
        const { prmname, description, node_type, link_type, feature_class } = feature.properties;
        const typeLabel = node_type || link_type || "";
        layer.bindTooltip(
          `<strong>${prmname}</strong>${description ? `<br/>${description}` : ""}${
            typeLabel ? `<br/><em style="color:#9ca3af">${typeLabel}</em>` : ""
          }`,
          { sticky: true, className: "calvin-tooltip" }
        );
        if (feature_class === "node") {
          layer.on("click", () => onNodeClick(prmname));
        } else if (feature_class === "link") {
          layer.on("click", () => onLinkClick(prmname));
        }
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Map legend
// ---------------------------------------------------------------------------

function Legend() {
  return (
    <div
      className="absolute bottom-8 left-3 z-[1000] bg-gray-900/90 border border-gray-700 rounded p-2 text-xs text-gray-300 pointer-events-none"
      style={{ backdropFilter: "blur(4px)" }}
    >
      <p className="font-semibold text-gray-400 mb-1.5">Nodes</p>
      {Object.entries(NODE_STYLE).map(([type, { color, r }]) => (
        <div key={type} className="flex items-center gap-1.5 mb-0.5">
          <span
            style={{
              display: "inline-block",
              width: Math.max(r * 2, 8),
              height: Math.max(r * 2, 8),
              borderRadius: "50%",
              background: color,
              flexShrink: 0,
            }}
          />
          <span className="text-gray-400 text-[10px]">{type}</span>
        </div>
      ))}
      <p className="font-semibold text-gray-400 mt-2 mb-1">Links (by destination)</p>
      {[
        ["Transit (junction)",  "#3b82f6"],
        ["To Groundwater",      "#10b981"],
        ["To Ag Demand",        "#f59e0b"],
        ["To Surface Storage",  "#ef4444"],
        ["To Pump / Power",     "#8b5cf6"],
        ["To Water Treatment",  "#06b6d4"],
        ["To Urban Demand",     "#f97316"],
        ["To Non-Std Demand",   "#84cc16"],
      ].map(([label, color]) => (
        <div key={label} className="flex items-center gap-1.5 mb-0.5">
          <span style={{ display: "inline-block", width: 16, height: 2, background: color, flexShrink: 0, borderRadius: 1 }} />
          <span className="text-gray-400 text-[10px]">{label}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main map component
// ---------------------------------------------------------------------------

export default function NetworkMap({ selectedNode, flyToNode, onNodeClick, onLinkClick, shortageData, debugData }) {
  const { data: geojson, isLoading } = useQuery({
    queryKey: ["network"],
    queryFn: fetchNetwork,
  });

  // Build prmname → total shortage map from raw column totals.
  // Shortage columns are link names "{origin}-{terminus}"; match each node's prmname
  // using the same prefix/suffix logic as the results API.
  const shortageMap = useMemo(() => {
    if (!shortageData || !geojson) return null;
    const { shortage_cols = {}, pwp_cols = {} } = shortageData;
    const map = {};
    for (const feature of geojson.features) {
      if (feature.properties.feature_class !== "node") continue;
      const prmname = feature.properties.prmname;
      const prefix = `${prmname}-`;
      const suffix = `-${prmname}`;
      let total = 0;
      for (const [col, val] of Object.entries(shortage_cols)) {
        if (col.endsWith(suffix)) total += val;
      }
      for (const [col, val] of Object.entries(pwp_cols)) {
        if (col.startsWith(prefix)) total += val;
      }
      if (total > 0) map[prmname] = total;
    }
    return map;
  }, [shortageData, geojson]);

  // Build Sets of debug link/node prmnames for O(1) lookup in style functions.
  const debugLinkSet = useMemo(
    () => (debugData ? new Set(debugData.active_links ?? []) : null),
    [debugData]
  );
  const debugNodeSet = useMemo(
    () => (debugData ? new Set(debugData.active_nodes ?? []) : null),
    [debugData]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        Loading network…
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[37.5, -119.5]}
        zoom={6}
        className="h-full w-full"
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          opacity={0.35}
        />
        {geojson && (
          <>
            <GeoJSONLayer
              geojson={geojson}
              selectedNode={selectedNode}
              onNodeClick={onNodeClick}
              onLinkClick={onLinkClick}
              shortageMap={shortageMap}
              debugLinkSet={debugLinkSet}
              debugNodeSet={debugNodeSet}
            />
            <MapResizeHandler />
            <MapFlyTo prmname={flyToNode} geojson={geojson} />
          </>
        )}
      </MapContainer>
      <Legend />
    </div>
  );
}
