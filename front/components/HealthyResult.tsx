"use client";

interface Props {
  probability: number;
  threshold: number;
  onReset: () => void;
  previewUrl: string;
}

export default function HealthyResult({ probability, threshold, onReset, previewUrl }: Props) {
  const confidence = ((1 - probability) * 100).toFixed(1);

  return (
    <div className="w-full max-w-2xl mx-auto animate-fade-up">
      {/* Status banner */}
      <div className="glow-green rounded-2xl border border-healthy/30 bg-healthy/5 p-6 mb-6">
        <div className="flex items-center gap-4">
          {/* Icon */}
          <div className="flex-shrink-0 w-14 h-14 rounded-full border-2 border-healthy/50 bg-healthy/10
                          flex items-center justify-center animate-pulse-ring">
            <svg className="w-7 h-7 text-healthy" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </div>

          <div>
            <p className="font-mono text-xs text-healthy/70 tracking-widest uppercase mb-1">
              Classification result
            </p>
            <h2 className="font-display text-3xl font-bold text-healthy">
              No Tumour Detected
            </h2>
          </div>
        </div>
      </div>

      {/* Two-column: image + metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Preview */}
        <div className="rounded-xl overflow-hidden border border-border bg-panel">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl} alt="MRI scan" className="w-full object-cover max-h-56" />
          <p className="font-mono text-xs text-text-dim text-center py-2">
            Original MRI scan
          </p>
        </div>

        {/* Metrics */}
        <div className="rounded-xl border border-border bg-panel p-5 flex flex-col justify-between gap-4">
          <Metric label="Tumour probability" value={`${(probability * 100).toFixed(2)}%`} accent="text-text-secondary" />
          <Metric label="Healthy confidence"  value={`${confidence}%`}   accent="text-healthy" large />
          <Metric label="Decision threshold"  value={`${(threshold * 100).toFixed(0)}%`} accent="text-text-secondary" />

          {/* Confidence bar */}
          <div>
            <div className="flex justify-between font-mono text-xs text-text-dim mb-1">
              <span>Healthy confidence</span>
              <span>{confidence}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-healthy transition-all duration-1000"
                style={{ width: `${confidence}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Reset */}
      <button
        onClick={onReset}
        className="mt-6 w-full py-3 rounded-xl border border-border bg-panel hover:bg-surface
                   font-display font-semibold text-text-secondary hover:text-text-primary
                   transition-all duration-200 tracking-wide"
      >
        Analyse Another Scan
      </button>
    </div>
  );
}

function Metric({
  label, value, accent, large,
}: {
  label: string; value: string; accent: string; large?: boolean;
}) {
  return (
    <div>
      <p className="font-mono text-xs text-text-dim uppercase tracking-wider mb-0.5">{label}</p>
      <p className={`font-mono font-semibold ${large ? "text-2xl" : "text-lg"} ${accent}`}>
        {value}
      </p>
    </div>
  );
}
