// lib/api.ts
// Typed wrapper around the FastAPI /predict endpoint.

export interface PredictionResult {
  status: "healthy" | "tumour_detected";
  tumour_probability: number;
  classification_threshold: number;
  segmentation_overlay: string | null;
  confidence_heatmap: string | null;
  segmentation_coverage_pct: number | null;
}

/**
 * Send an MRI image to the FastAPI backend and return the prediction.
 * Uses the Next.js rewrite so the request goes to /api/predict → localhost:8000/predict.
 */
export async function predict(file: File): Promise<PredictionResult> {
  const body = new FormData();
  body.append("file", file);

  const res = await fetch("/api/predict", {
    method: "POST",
    body,
  });

  if (!res.ok) {
    let detail = `Server error ${res.status}`;
    try {
      const json = await res.json();
      detail = json.detail ?? detail;
    } catch {
      // ignore JSON parse failure
    }
    throw new Error(detail);
  }

  return res.json() as Promise<PredictionResult>;
}
