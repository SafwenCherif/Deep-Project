"use client";

import { useState } from "react";

interface Props {
  probability: number;
  threshold: number;
  coveragePct: number;
  overlayB64: string;
  heatmapB64: string;
  onReset: () => void;
  previewUrl: string;
}

type ActiveView = "overlay" | "heatmap" | "original";

export default function TumourResult({
  probability,
  threshold,
  coveragePct,
  overlayB64,
  heatmapB64,
  onReset,
  previewUrl,
}: Props) {
  const [activeView, setActiveView] = useState<ActiveView>("overlay");

  const views: { id: ActiveView; label: string }[] = [
    { id: "overlay", label: "Tumour Mask" },
    { id: "heatmap", label: "AI Heatmap" },
    { id: "original", label: "Original" },
  ];

  const activeImage =
    activeView === "overlay"
      ? `data:image/png;base64,${overlayB64}`
      : activeView === "heatmap"
      ? `data:image/png;base64,${heatmapB64}`
      : previewUrl;

  const activeCaption =
    activeView === "overlay"
      ? "MRI + red tumour mask overlay"
      : activeView === "heatmap"
      ? "Segmentor confidence · blue=0% → red=100%"
      : "Original uploaded MRI scan";

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-up space-y-5">

      {/* Status banner */}
      <div className="glow-red rounded-2xl border border-tumour/30 bg-tumour/5 p-5">
        <div className="flex items-center gap-4">
          <div className="flex-shrink-0 w-14 h-14 rounded-full border-2 border-tumour/50 bg-tumour/10
                          flex items-center justify-center">
            <svg className="w-7 h-7 text-tumour" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            </svg>
          </div>
          <div>
            <p className="font-mono text-xs text-tumour/70 tracking-widest uppercase mb-1">
              Classification result
            </p>
            <h2 className="font-display text-3xl font-bold text-tumour">
              Tumour Detected
            </h2>
          </div>

          {/* Coverage badge */}
          <div className="ml-auto text-right hidden sm:block">
            <p className="font-mono text-xs text-text-dim uppercase tracking-wider">Coverage</p>
            <p className="font-mono text-2xl font-semibold text-amber-400">
              {coveragePct.toFixed(2)}%
            </p>
            <p className="font-mono text-xs text-text-dim">of image pixels</p>
          </div>
        </div>
      </div>

      {/* Main grid: image viewer + metrics */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">

        {/* ── Image viewer ─────────────────────────────────────────── */}
        <div className="rounded-2xl border border-border bg-panel overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-border">
            {views.map((v) => (
              <button
                key={v.id}
                onClick={() => setActiveView(v.id)}
                className={[
                  "flex-1 py-2.5 font-mono text-xs tracking-wider uppercase transition-all duration-200",
                  activeView === v.id
                    ? "text-scan border-b-2 border-scan bg-scan/5"
                    : "text-text-dim hover:text-text-secondary",
                ].join(" ")}
              >
                {v.label}
              </button>
            ))}
          </div>

          {/* Image */}
          <div className="relative bg-void">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={activeView}
              src={activeImage}
              alt={activeCaption}
              className="w-full object-contain max-h-80 animate-fade-in"
            />

            {/* Heatmap legend */}
            {activeView === "heatmap" && (
              <div className="absolute bottom-3 left-3 right-3 flex items-center gap-2">
                <span className="font-mono text-xs text-white/60">0%</span>
                <div className="flex-1 h-1.5 rounded-full"
                     style={{ background: "linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)" }} />
                <span className="font-mono text-xs text-white/60">100%</span>
              </div>
            )}
          </div>

          <p className="font-mono text-xs text-text-dim text-center py-2 px-3">
            {activeCaption}
          </p>
        </div>

        {/* ── Metrics panel ─────────────────────────────────────────── */}
        <div className="rounded-2xl border border-border bg-panel p-5 flex flex-col gap-5">
          <MetricBlock
            label="Tumour probability"
            value={`${(probability * 100).toFixed(2)}%`}
            accent="text-tumour"
            bar={{ value: probability, color: "bg-tumour" }}
            large
          />
          <MetricBlock
            label="Decision threshold"
            value={`${(threshold * 100).toFixed(0)}%`}
            accent="text-text-secondary"
          />
          <MetricBlock
            label="Tumour coverage"
            value={`${coveragePct.toFixed(2)}%`}
            accent="text-amber-400"
            bar={{ value: coveragePct / 100, color: "bg-amber-400" }}
          />

          {/* Disclaimer */}
          <div className="mt-auto pt-4 border-t border-border">
            <p className="font-mono text-xs text-text-dim leading-relaxed">
              ⚠ For research use only. Not a clinical diagnostic tool. Always consult a qualified radiologist.
            </p>
          </div>
        </div>
      </div>

      {/* Reset */}
      <button
        onClick={onReset}
        className="w-full py-3 rounded-xl border border-border bg-panel hover:bg-surface
                   font-display font-semibold text-text-secondary hover:text-text-primary
                   transition-all duration-200 tracking-wide"
      >
        Analyse Another Scan
      </button>
    </div>
  );
}

function MetricBlock({
  label, value, accent, bar, large,
}: {
  label: string;
  value: string;
  accent: string;
  bar?: { value: number; color: string };
  large?: boolean;
}) {
  return (
    <div>
      <p className="font-mono text-xs text-text-dim uppercase tracking-wider mb-1">{label}</p>
      <p className={`font-mono font-semibold ${large ? "text-3xl" : "text-xl"} ${accent}`}>
        {value}
      </p>
      {bar && (
        <div className="mt-2 h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-1000 ${bar.color}`}
            style={{ width: `${Math.min(bar.value * 100, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}
