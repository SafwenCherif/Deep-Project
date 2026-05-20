"use client";

import { useState, useCallback } from "react";
import UploadZone from "@/components/UploadZone";
import ScanningState from "@/components/ScanningState";
import HealthyResult from "@/components/HealthyResult";
import TumourResult from "@/components/TumourResult";
import { predict, type PredictionResult } from "@/lib/api";

type AppState = "idle" | "scanning" | "result" | "error";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    // Generate local preview URL for displaying during scan and in result
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setState("scanning");
    setResult(null);
    setErrorMsg(null);

    try {
      const data = await predict(file);
      setResult(data);
      setState("result");
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Unexpected error");
      setState("error");
    }
  }, []);

  const reset = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
    setState("idle");
  }, [previewUrl]);

  return (
    <div className="relative min-h-screen bg-void bg-grid bg-vignette">
      {/* Ambient glow blobs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-scan/10 blur-3xl" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 rounded-full bg-scan/5 blur-3xl" />
      </div>

      <div className="relative z-10 flex flex-col min-h-screen">

        {/* ── Header ──────────────────────────────────────────────────── */}
        <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-20">
          <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Logo mark */}
              <div className="w-8 h-8 rounded-lg border border-scan/40 bg-scan/10
                              flex items-center justify-center">
                <svg className="w-4 h-4 text-scan" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <circle cx="12" cy="12" r="3" />
                  <path strokeLinecap="round" d="M12 2v3M12 19v3M2 12h3M19 12h3
                    M5.22 5.22l2.12 2.12M16.66 16.66l2.12 2.12
                    M5.22 18.78l2.12-2.12M16.66 7.34l2.12-2.12" />
                </svg>
              </div>
              <span className="font-display font-bold text-lg text-text-primary tracking-wide">
                NeuroScan<span className="text-scan">AI</span>
              </span>
            </div>

            {/* Pipeline badge */}
            <div className="hidden sm:flex items-center gap-2 font-mono text-xs text-text-dim">
              <span className="px-2 py-0.5 rounded border border-border bg-panel">ResNet50</span>
              <span className="text-text-dim">→</span>
              <span className="px-2 py-0.5 rounded border border-border bg-panel">UNet</span>
            </div>
          </div>
        </header>

        {/* ── Main content ─────────────────────────────────────────────── */}
        <main className="flex-1 flex flex-col items-center justify-center px-6 py-12">

          {/* Hero (idle only) */}
          {state === "idle" && (
            <div className="text-center mb-10 animate-fade-up">
              <p className="font-mono text-xs text-scan tracking-widest uppercase mb-4">
                Two-stage AI pipeline
              </p>
              <h1 className="font-display text-5xl sm:text-6xl font-bold text-text-primary leading-tight mb-4">
                Brain Tumour<br />
                <span className="text-scan">Detection</span>
              </h1>
              <p className="text-text-secondary text-base max-w-md mx-auto leading-relaxed">
                Upload an MRI scan. The classifier screens for tumours; if detected,
                the segmentor pinpoints the exact region.
              </p>

              {/* Pipeline diagram */}
              <div className="mt-8 flex items-center justify-center gap-2 flex-wrap">
                {[
                  { label: "MRI Upload", icon: "↑" },
                  { label: "ResNet50 Classify", icon: "⬡" },
                  { label: "UNet Segment", icon: "◈" },
                  { label: "Overlay Result", icon: "◉" },
                ].map((step, i) => (
                  <div key={step.label} className="flex items-center gap-2">
                    <div className="flex flex-col items-center gap-1">
                      <div className="w-9 h-9 rounded-lg border border-border bg-panel
                                      flex items-center justify-center text-scan font-mono text-sm">
                        {step.icon}
                      </div>
                      <span className="font-mono text-xs text-text-dim">{step.label}</span>
                    </div>
                    {i < 3 && <span className="text-border mb-4">──</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* State machine */}
          <div className="w-full max-w-3xl mx-auto flex flex-col items-center">
            {state === "idle" && (
              <div className="w-full animate-fade-up delay-200">
                <UploadZone onFile={handleFile} />
              </div>
            )}

            {state === "scanning" && previewUrl && (
              <ScanningState previewUrl={previewUrl} />
            )}

            {state === "result" && result && previewUrl && (
              result.status === "healthy" ? (
                <HealthyResult
                  probability={result.tumour_probability}
                  threshold={result.classification_threshold}
                  previewUrl={previewUrl}
                  onReset={reset}
                />
              ) : (
                <TumourResult
                  probability={result.tumour_probability}
                  threshold={result.classification_threshold}
                  coveragePct={result.segmentation_coverage_pct!}
                  overlayB64={result.segmentation_overlay!}
                  heatmapB64={result.confidence_heatmap!}
                  previewUrl={previewUrl}
                  onReset={reset}
                />
              )
            )}

            {state === "error" && (
              <div className="w-full max-w-xl animate-fade-up">
                <div className="rounded-2xl border border-tumour/30 bg-tumour/5 p-8 text-center">
                  <div className="w-14 h-14 rounded-full border-2 border-tumour/40 bg-tumour/10
                                  flex items-center justify-center mx-auto mb-4">
                    <svg className="w-7 h-7 text-tumour" fill="none" viewBox="0 0 24 24"
                         stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </div>
                  <h3 className="font-display text-xl font-bold text-tumour mb-2">
                    Pipeline Error
                  </h3>
                  <p className="font-mono text-sm text-text-secondary mb-6 break-words">
                    {errorMsg}
                  </p>
                  <button
                    onClick={reset}
                    className="px-6 py-2.5 rounded-xl border border-border bg-panel
                               hover:bg-surface font-display font-semibold
                               text-text-primary transition-all duration-200"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* ── Footer ──────────────────────────────────────────────────── */}
        <footer className="border-t border-border/40 py-5 px-6">
          <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center
                          justify-between gap-2 font-mono text-xs text-text-dim">
            <span>
              ResNet50 · acc 84.07% · recall 95.59% · ROC-AUC 0.9564
            </span>
            <span className="text-border">|</span>
            <span>
              UNet · Focal Tversky Loss · Dice 51.5%
            </span>
            <span className="text-border">|</span>
            <span className="text-tumour/60">
              ⚠ Research only — not for clinical use
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
