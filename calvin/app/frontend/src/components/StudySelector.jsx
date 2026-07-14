/**
 * StudySelector — dropdown to switch between loaded studies.
 * Designed to eventually support selecting a second study for comparison.
 */
export default function StudySelector({ studies, active, onChange }) {
  if (!studies || studies.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-gray-400 whitespace-nowrap" htmlFor="study-select">
        Study
      </label>
      <select
        id="study-select"
        value={active ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="bg-gray-700 text-gray-100 text-sm rounded px-2 py-1 border border-gray-600
                   focus:outline-none focus:ring-1 focus:ring-blue-500"
      >
        {studies.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name}
          </option>
        ))}
      </select>
    </div>
  );
}
