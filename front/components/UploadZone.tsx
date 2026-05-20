"use client";

import { useRef, useState, useCallback, DragEvent, ChangeEvent } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED = [".jpg", ".jpeg", ".png", ".tif", ".tiff"];
const ACCEPTED_MIME = ["image/jpeg", "image/png", "image/tiff", "image/tif"];

export default function UploadZone({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((file: File): boolean => {
    const ok =
      ACCEPTED_MIME.includes(file.type) ||
      ACCEPTED.some((ext) => file.name.toLowerCase().endsWith(ext));
    if (!ok) {
      setError(`Unsupported format. Send a JPEG, PNG, or TIFF MRI image.`);
      return false;
    }
    setError(null);
    return true;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      if (validate(file)) onFile(file);
    },
    [onFile, validate]
  );

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      // Reset so the same file can be re-selected
      e.target.value = "";
    },
    [handleFile]
  );

  return (
    <div className="w-full flex flex-col items-center gap-4">
      {/* Drop target */}
      <div
        onDragOver={(e) => { e.preventDefault(); !disabled && setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={disabled ? undefined : onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={[
          "relative w-full max-w-xl h-64 rounded-2xl border-2 border-dashed",
          "flex flex-col items-center justify-center gap-4",
          "transition-all duration-300 cursor-pointer select-none",
          dragging
            ? "border-scan bg-scan/10 glow-blue scale-[1.01]"
            : "border-border bg-panel hover:border-muted hover:bg-surface",
          disabled ? "opacity-40 cursor-not-allowed pointer-events-none" : "",
        ].join(" ")}
      >
        {/* Corner accents */}
        {["top-3 left-3", "top-3 right-3", "bottom-3 left-3", "bottom-3 right-3"].map((pos) => (
          <span
            key={pos}
            className={`absolute ${pos} w-4 h-4 border-scan
              ${pos.includes("top") && pos.includes("left")    ? "border-t-2 border-l-2" : ""}
              ${pos.includes("top") && pos.includes("right")   ? "border-t-2 border-r-2" : ""}
              ${pos.includes("bottom") && pos.includes("left") ? "border-b-2 border-l-2" : ""}
              ${pos.includes("bottom") && pos.includes("right") ? "border-b-2 border-r-2" : ""}
            `}
          />
        ))}

        {/* Icon */}
        <div className={`transition-transform duration-300 ${dragging ? "scale-110" : ""}`}>
          <svg
            className={`w-12 h-12 ${dragging ? "text-scan" : "text-text-dim"} transition-colors duration-300`}
            fill="none" viewBox="0 0 48 48" stroke="currentColor" strokeWidth={1.2}
          >
            <rect x="6" y="6" width="36" height="36" rx="4" strokeDasharray="4 2" />
            <path d="M24 18v12M18 24h12" strokeLinecap="round" />
            <circle cx="24" cy="24" r="3" fill="currentColor" className="opacity-40" />
          </svg>
        </div>

        {/* Text */}
        <div className="text-center px-6">
          <p className="font-display font-semibold text-text-primary text-base tracking-wide">
            {dragging ? "Drop to analyse" : "Drop MRI scan here"}
          </p>
          <p className="text-text-secondary text-sm font-mono mt-1">
            or click to browse · JPEG · PNG · TIFF
          </p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="hidden"
          onChange={onInputChange}
        />
      </div>

      {/* Validation error */}
      {error && (
        <p className="font-mono text-tumour text-xs animate-fade-in">
          ⚠ {error}
        </p>
      )}
    </div>
  );
}
