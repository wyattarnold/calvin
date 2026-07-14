import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchNodes } from "../api/client.js";

export default function NodeSearch({ onSelect }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  const { data: nodes = [] } = useQuery({
    queryKey: ["nodes"],
    queryFn: fetchNodes,
  });

  const q = query.trim().toLowerCase();
  const results = q.length === 0 ? [] : nodes
    .filter((n) =>
      n.prmname.toLowerCase().includes(q) ||
      (n.description || "").toLowerCase().includes(q)
    )
    .slice(0, 10);

  // Close dropdown on outside click
  useEffect(() => {
    function onMouseDown(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  function handleSelect(prmname) {
    setQuery("");
    setOpen(false);
    onSelect(prmname);
  }

  return (
    <div ref={containerRef} className="relative">
      <input
        ref={inputRef}
        type="text"
        placeholder="Search nodes…"
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => { if (query.trim()) setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "Escape") { setOpen(false); setQuery(""); inputRef.current?.blur(); }
          if (e.key === "Enter" && results.length === 1) handleSelect(results[0].prmname);
        }}
        className="w-44 text-xs bg-gray-700 border border-gray-600 rounded px-2.5 py-1 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
      />
      {open && results.length > 0 && (
        <div className="absolute top-full left-0 mt-1 w-64 bg-gray-800 border border-gray-700 rounded shadow-xl z-[2000] max-h-72 overflow-y-auto">
          {results.map((n) => (
            <button
              key={n.prmname}
              onMouseDown={(e) => { e.preventDefault(); handleSelect(n.prmname); }}
              className="flex flex-col w-full text-left px-3 py-2 hover:bg-gray-700 border-b border-gray-700 last:border-0"
            >
              <span className="font-mono text-xs text-blue-400">{n.prmname}</span>
              {n.description && (
                <span className="text-[10px] text-gray-400 truncate">{n.description}</span>
              )}
              {n.node_type && (
                <span className="text-[10px] text-gray-600">{n.node_type}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
