"use client";

/** Shows a confidence value AND the method that produced it. Clicking opens the evidence. */
export function ConfidenceBadge({ value, method, onClick }: { value: number; method: string; onClick?: () => void }) {
  const tone = value >= 0.75 ? "bg-good-soft text-good" : value >= 0.5 ? "bg-warn-soft text-warn" : "bg-bad-soft text-bad";
  const body = (
    <>
      <span className="font-mono">{value.toFixed(2)}</span>
      <span className="opacity-80">· {method.replace(/_/g, " ")}</span>
    </>
  );
  if (!onClick) {
    return <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ${tone}`}>{body}</span>;
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title="Show the evidence behind this confidence"
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold underline-offset-2 hover:underline ${tone}`}
    >
      {body}
    </button>
  );
}
