"use client";

interface Props {
  previewUrl: string;
}

export default function ScanningState({ previewUrl }: Props) {
  return (
    <div className="w-full max-w-xl mx-auto animate-fade-in">
      {/* Label */}
      <div className="flex items-center gap-2 mb-3">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-scan opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-scan" />
        </span>
        <span className="font-mono text-xs text-scan tracking-widest uppercase">
          Analysing scan
        </span>
      </div>

      {/* Image with scan line */}
      <div className="relative rounded-2xl overflow-hidden border border-border scan-overlay">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={previewUrl}
          alt="MRI being analysed"
          className="w-full object-cover max-h-80 opacity-60"
        />
        {/* Blue tint overlay */}
        <div className="absolute inset-0 bg-scan/10" />
      </div>

      {/* Progress steps */}
      <div className="mt-5 space-y-2">
        {[
          { label: "Stage 1 — ResNet50 classification", delay: "0ms" },
          { label: "Stage 2 — UNet segmentation (if needed)", delay: "300ms" },
          { label: "Rendering results", delay: "600ms" },
        ].map(({ label, delay }) => (
          <div key={label} className="flex items-center gap-3">
            <div
              className="h-1 flex-1 rounded-full shimmer-bar"
              style={{ animationDelay: delay }}
            />
            <span
              className="font-mono text-xs text-text-dim w-72 text-right"
              style={{ animationDelay: delay }}
            >
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
