import { useRef, useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import NetworkMap from "./components/NetworkMap.jsx";
import NodePanel from "./components/NodePanel.jsx";
import NeighborhoodGraph from "./components/NeighborhoodGraph.jsx";
import StudySelector from "./components/StudySelector.jsx";
import NodeSearch from "./components/NodeSearch.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import { fetchStudies, fetchShortageNodes, fetchDebugLinks } from "./api/client.js";

const MIN_PANEL_WIDTH = 260;
const MAX_PANEL_WIDTH = 800;
const DEFAULT_RIGHT_WIDTH = 500;
const DEFAULT_LEFT_WIDTH = 280;

export default function App() {
  const [selectedNode, setSelectedNode] = useState(null);
  const [flyToNode, setFlyToNode] = useState(null);
  const [activeStudy, setActiveStudy] = useState(null);
  const [panelOpen, setPanelOpen] = useState(true);
  const [graphOpen, setGraphOpen] = useState(true);
  const [rightWidth, setRightWidth] = useState(DEFAULT_RIGHT_WIDTH);
  const [leftWidth, setLeftWidth] = useState(DEFAULT_LEFT_WIDTH);
  const [viewMode, setViewMode] = useState("default"); // "default" | "shortage" | "debug"

  // Track which drag handle is active: "left" | "middle" | null
  const dragging = useRef(null);

  // Refs so the resize handler always sees current values without re-registering.
  const leftWidthRef = useRef(DEFAULT_LEFT_WIDTH);
  const showGraphRef = useRef(false);

  const { data: studiesData } = useQuery({
    queryKey: ["studies"],
    queryFn: fetchStudies,
  });

  const _study = activeStudy ?? studiesData?.active;

  const { data: shortageData } = useQuery({
    queryKey: ["shortage-nodes", _study],
    queryFn: () => fetchShortageNodes(_study),
    enabled: viewMode === "shortage",
  });

  const { data: debugData } = useQuery({
    queryKey: ["debug-links", _study],
    queryFn: () => fetchDebugLinks(_study),
    enabled: viewMode === "debug",
  });

  // ---------------------------------------------------------------------------
  // Resize logic
  // ---------------------------------------------------------------------------
  const handleLeftDragStart = useCallback((e) => {
    e.preventDefault();
    dragging.current = "left";
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const handleMiddleDragStart = useCallback((e) => {
    e.preventDefault();
    dragging.current = "middle";
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      if (dragging.current === "left") {
        const w = Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, clientX));
        setLeftWidth(w);
        leftWidthRef.current = w;
      } else if (dragging.current === "middle") {
        // NodePanel is immediately right of the graph (if open) + its 4px divider.
        const graphOffset = showGraphRef.current ? leftWidthRef.current + 4 : 0;
        setRightWidth(Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, clientX - graphOffset)));
      }
    };
    const onUp = () => {
      if (!dragging.current) return;
      dragging.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Node / link selection
  // ---------------------------------------------------------------------------
  function handleNodeClick(prmname) {
    setSelectedNode(prmname);
    setPanelOpen(true);
    // Map click — don't fly, user already sees the node
  }

  function handleNeighborhoodNodeClick(prmname) {
    setSelectedNode(prmname);
    setPanelOpen(true);
    setFlyToNode(prmname); // Pan map to the clicked node
  }

  function handleLinkClick(prmname) {
    setSelectedNode(prmname);
    setPanelOpen(true);
  }

  function handlePanelClose() {
    setPanelOpen(false);
    setSelectedNode(null);
  }

  const showPanels = panelOpen && !!selectedNode;
  const showGraph = showPanels && graphOpen;

  // Keep refs in sync with current render values.
  showGraphRef.current = showGraph;
  leftWidthRef.current = leftWidth;

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100 overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-3 px-4 py-2 bg-gray-800 border-b border-gray-700 shrink-0 z-10">
        <h1 className="text-lg font-semibold text-blue-400 whitespace-nowrap">Calvin Network</h1>
        <NodeSearch onSelect={handleNeighborhoodNodeClick} />
        <StudySelector
          studies={studiesData?.studies ?? []}
          active={_study}
          onChange={setActiveStudy}
        />
        <div className="flex-1" />
        <button
          onClick={() => setViewMode("default")}
          className={`text-xs px-2.5 py-1 border rounded transition-colors ${
            viewMode === "default"
              ? "border-gray-500 text-gray-200 bg-gray-700"
              : "border-gray-700 text-gray-500 hover:border-gray-500 hover:text-gray-400"
          }`}
        >
          Default
        </button>
        <button
          onClick={() => setViewMode("shortage")}
          className={`text-xs px-2.5 py-1 border rounded transition-colors ${
            viewMode === "shortage"
              ? "border-orange-500 text-orange-300 bg-orange-950"
              : "border-gray-700 text-gray-500 hover:border-orange-600 hover:text-orange-400"
          }`}
        >
          Shortages
        </button>
        <button
          onClick={() => setViewMode("debug")}
          className={`text-xs px-2.5 py-1 border rounded transition-colors ${
            viewMode === "debug"
              ? "border-pink-500 text-pink-300 bg-pink-950"
              : "border-gray-700 text-gray-500 hover:border-pink-600 hover:text-pink-400"
          }`}
        >
          Debug
        </button>
      </header>

      {/* Layout: [NeighborhoodGraph] [NodePanel] [Map] */}
      <div className="flex flex-1 overflow-hidden">

        {/* Leftmost panel — neighborhood graph */}
        {showGraph && (
          <>
            <aside
              style={{ width: leftWidth, backgroundColor: "#0a0f1a" }}
              className="flex flex-col border-r border-gray-700 overflow-hidden shrink-0"
            >
              <ErrorBoundary>
                <NeighborhoodGraph
                  prmname={selectedNode}
                  onNodeClick={handleNeighborhoodNodeClick}
                />
              </ErrorBoundary>
            </aside>
            <div
              onMouseDown={handleLeftDragStart}
              onTouchStart={handleLeftDragStart}
              className="w-1 bg-gray-700 hover:bg-blue-500 cursor-col-resize shrink-0 transition-colors"
            />
          </>
        )}

        {/* Node detail panel */}
        {showPanels && (
          <>
            <aside
              style={{ width: rightWidth, backgroundColor: "#1f2937" }}
              className="flex flex-col border-r border-gray-700 overflow-hidden shrink-0"
            >
              <ErrorBoundary>
                <NodePanel
                  prmname={selectedNode}
                  activeStudy={activeStudy}
                  onClose={handlePanelClose}
                  graphOpen={graphOpen}
                  onToggleGraph={() => setGraphOpen((g) => !g)}
                />
              </ErrorBoundary>
            </aside>
            <div
              onMouseDown={handleMiddleDragStart}
              onTouchStart={handleMiddleDragStart}
              className="w-1 bg-gray-700 hover:bg-blue-500 cursor-col-resize shrink-0 transition-colors"
            />
          </>
        )}

        {/* Map — fills remaining space */}
        <div className="flex-1 relative overflow-hidden">
          <NetworkMap
            selectedNode={selectedNode}
            flyToNode={flyToNode}
            onNodeClick={handleNodeClick}
            onLinkClick={handleLinkClick}
            shortageData={viewMode === "shortage" ? shortageData : null}
            debugData={viewMode === "debug" ? debugData : null}
          />
        </div>
      </div>
    </div>
  );
}
