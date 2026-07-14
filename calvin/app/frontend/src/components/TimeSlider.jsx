import { useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchSummary } from "../api/client.js";

/**
 * TimeSlider — controls the active date driving map node coloring.
 *
 * The slider steps through annual water-year endpoints (Sep 30 of each year).
 * Moving the slider updates the active date → the map re-colors nodes
 * by shortage magnitude at that date, and the footer shows annual summary stats.
 */
export default function TimeSlider({ activeStudy, onDateChange }) {
  const [sliderIdx, setSliderIdx] = useState(null);

  const { data: summary } = useQuery({
    queryKey: ["summary", activeStudy],
    queryFn: () => fetchSummary(activeStudy),
    enabled: !!activeStudy,
  });

  const years = summary?.years ?? [];
  const maxIdx = Math.max(0, years.length - 1);

  // Initialise to the last year on first load
  useEffect(() => {
    if (years.length > 0 && sliderIdx === null) {
      setSliderIdx(maxIdx);
    }
  }, [years.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // Emit date whenever the index changes
  useEffect(() => {
    if (sliderIdx !== null && years[sliderIdx]) {
      onDateChange(`${years[sliderIdx]}-09-30`);
    }
  }, [sliderIdx, years]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeYear = sliderIdx != null ? years[sliderIdx] : null;
  const shortageVol = summary?.total_shortage_volume?.[sliderIdx ?? maxIdx] ?? null;
  const shortageCost = summary?.total_shortage_cost?.[sliderIdx ?? maxIdx] ?? null;

  // Colour the label by shortage severity
  const shortageColor = useMemo(() => {
    if (shortageVol == null) return "text-gray-400";
    const maxShortage = summary ? Math.max(...summary.total_shortage_volume) : 1;
    const ratio = shortageVol / maxShortage;
    if (ratio > 0.7) return "text-red-400";
    if (ratio > 0.4) return "text-amber-400";
    return "text-green-400";
  }, [shortageVol, summary]);

  if (years.length === 0) {
    return (
      <div className="flex items-center gap-3 text-xs text-gray-500">
        <span>No study loaded — start the server with</span>
        <code className="text-gray-400 bg-gray-700 px-1 rounded">--study my-models/calvin-pf</code>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 min-w-0">
      {/* Label */}
      <span className="text-xs text-gray-500 whitespace-nowrap">Map date</span>

      {/* Start year */}
      <span className="text-xs text-gray-600 whitespace-nowrap">{years[0]}</span>

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={maxIdx}
        value={sliderIdx ?? maxIdx}
        onChange={(e) => setSliderIdx(Number(e.target.value))}
        className="flex-1 accent-blue-500 min-w-0"
        title="Drag to change the active water year shown on the map"
      />

      {/* End year */}
      <span className="text-xs text-gray-600 whitespace-nowrap">{years[maxIdx]}</span>

      {/* Active year + stats */}
      <div className="flex items-center gap-3 shrink-0">
        <span className="text-sm font-mono font-semibold text-blue-400 w-12 text-right">
          {activeYear ?? "—"}
        </span>
        {shortageVol !== null && (
          <span className={`text-xs whitespace-nowrap ${shortageColor}`}>
            ▼ {shortageVol.toFixed(0)} TAF
          </span>
        )}
        {shortageCost !== null && (
          <span className="text-xs text-gray-500 whitespace-nowrap">
            ${(shortageCost / 1e6).toFixed(0)}M
          </span>
        )}
      </div>
    </div>
  );
}
