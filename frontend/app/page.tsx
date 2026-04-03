"use client";

import axios from "axios";
import Image from "next/image";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

type PupilAssessment = {
  isTooLarge: boolean;
  isTooSmall: boolean;
  isIrregular: boolean;
};

type Point2D = { x: number; y: number };

type FittedCircle = {
  centerX: number;
  centerY: number;
  radius: number;
  rmsDeviationPx: number;
  maxDeviationPx: number;
};

type ProportionalityAssessment = {
  pupilToIrisRatio: number | null;
  isProportional: boolean | null;
  note: string;
};

type AnalysisResponse = {
  pupilDiameterPixels: number | null;
  circularity: number | null;
  contourArea: number | null;
  warnings: string[];
  assessment: PupilAssessment;
  processingNotes?: string;
  pupilPoints?: Point2D[] | null;
  irisPoints?: Point2D[] | null;
  fittedCircle?: FittedCircle | null;
  deviationFromPerfectCircle?: string | null;
  proportionality?: ProportionalityAssessment | null;
  imageWidth?: number | null;
  imageHeight?: number | null;
  processedImageBase64?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

function getIrisRadiusFromPoints(centerX: number, centerY: number, points: Point2D[]): number {
  if (!points.length) return 0;
  const p = points[0];
  return Math.hypot(p.x - centerX, p.y - centerY);
}

function escapeCsvCell(value: string): string {
  const s = String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function rowsToCsv(dataRows: Array<[string, string]>): string {
  const header = "field,value";
  const body = dataRows.map(([k, v]) => `${escapeCsvCell(k)},${escapeCsvCell(v)}`).join("\n");
  return `${header}\n${body}`;
}

function triggerCsvDownload(filename: string, csvContent: string) {
  const bom = "\uFEFF";
  const blob = new Blob([bom + csvContent], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [adjustedIrisRadius, setAdjustedIrisRadius] = useState<number | null>(null);
  const [adjustedIrisCenter, setAdjustedIrisCenter] = useState<Point2D | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragModeRef = useRef<"radius" | "center" | null>(null);
  const dragPointerIdRef = useRef<number | null>(null);
  const windowDragListenersRef = useRef<{
    move: (e: PointerEvent) => void;
    up: (e: PointerEvent) => void;
  } | null>(null);
  const dragStateRef = useRef({
    pupilCenter: null as { x: number; y: number } | null,
    irisCenter: null as { x: number; y: number } | null,
    irisRadius: 0,
    pupilRadius: 0,
  });

  const hasResult = !!result;

  const pupilCenter = result?.fittedCircle ? { x: result.fittedCircle.centerX, y: result.fittedCircle.centerY } : null;
  const pupilRadius = result?.fittedCircle?.radius ?? 0;
  const initialIrisRadius = useMemo(() => {
    if (!result?.irisPoints?.length || !pupilCenter) return 0;
    return getIrisRadiusFromPoints(pupilCenter.x, pupilCenter.y, result.irisPoints);
  }, [result?.irisPoints, pupilCenter]);
  const irisRadius = adjustedIrisRadius ?? initialIrisRadius;
  const irisCenter = adjustedIrisCenter ?? pupilCenter;

  const irisPointsForDisplay = useMemo(() => {
    if (!irisCenter || irisRadius <= 0) return result?.irisPoints ?? [];
    const pts: Point2D[] = [];
    for (let i = 0; i < 64; i++) {
      const t = (2 * Math.PI * i) / 64;
      pts.push({ x: irisCenter.x + irisRadius * Math.cos(t), y: irisCenter.y + irisRadius * Math.sin(t) });
    }
    return pts;
  }, [irisCenter, irisRadius, result?.irisPoints]);

  const manualProportion = useMemo(() => {
    if (!pupilCenter || pupilRadius <= 0 || irisRadius <= 0) return null;
    const ratio = pupilRadius / irisRadius;
    const typicalMin = 0.15, typicalMax = 0.55;
    const isProportional = typicalMin <= ratio && ratio <= typicalMax;
    return { ratio, isProportional };
  }, [pupilCenter, pupilRadius, irisRadius]);

  const centering = useMemo(() => {
    if (!pupilCenter || !irisCenter || irisRadius <= 0) return null;
    const dx = pupilCenter.x - irisCenter.x;
    const dy = pupilCenter.y - irisCenter.y;
    const offsetPx = Math.hypot(dx, dy);
    const normalized = offsetPx / irisRadius;
    let label: string;
    if (normalized < 0.15) label = "well centered";
    else if (normalized < 0.30) label = "mildly decentered";
    else label = "markedly decentered";
    return { offsetPx, normalized, label };
  }, [pupilCenter, irisCenter, irisRadius]);

  /** Merges API assessment with live pupil/iris ratio and centering (manual iris edits). */
  const effectiveAssessment = useMemo((): PupilAssessment | null => {
    if (!result) return null;
    const typicalMin = 0.15;
    const typicalMax = 0.55;
    let isTooLarge = result.assessment.isTooLarge;
    // Small pill: pupil/iris ratio only (matches backend; ignores absolute pixel diameter).
    let isTooSmall = manualProportion
      ? manualProportion.ratio < typicalMin
      : result.assessment.isTooSmall;
    if (manualProportion && manualProportion.ratio > typicalMax) {
      isTooLarge = true;
    }
    // Irregular = decentering only (API + live centering vs adjusted iris).
    let isIrregular = result.assessment.isIrregular;
    if (centering && centering.normalized >= 0.15) {
      isIrregular = true;
    }
    return { isTooLarge, isTooSmall, isIrregular };
  }, [result, manualProportion, centering]);

  dragStateRef.current = {
    pupilCenter,
    irisCenter,
    irisRadius,
    pupilRadius,
  };

  const clientToSvg = useCallback((clientX: number, clientY: number): { x: number; y: number } | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    return svgPt ? { x: svgPt.x, y: svgPt.y } : null;
  }, []);

  const applyIrisPointerMove = useCallback((clientX: number, clientY: number) => {
    const mode = dragModeRef.current;
    const { pupilCenter: pc, irisCenter: ic, pupilRadius: pr } = dragStateRef.current;
    if (!mode || !pc) return;
    const pt = clientToSvg(clientX, clientY);
    if (!pt) return;
    if (mode === "radius" && ic) {
      const r = Math.hypot(pt.x - ic.x, pt.y - ic.y);
      if (r > pr * 1.1 && r < 500) setAdjustedIrisRadius(r);
    } else if (mode === "center") {
      setAdjustedIrisCenter({ x: pt.x, y: pt.y });
    }
  }, [clientToSvg]);

  const startIrisDrag = useCallback(
    (e: React.PointerEvent, mode: "radius" | "center") => {
      e.preventDefault();
      e.stopPropagation();
      if (dragPointerIdRef.current != null) return;
      const el = e.currentTarget as HTMLElement;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* already captured or unsupported */
      }
      dragPointerIdRef.current = e.pointerId;
      dragModeRef.current = mode;

      const onMove = (ev: PointerEvent) => {
        if (ev.pointerId !== dragPointerIdRef.current) return;
        applyIrisPointerMove(ev.clientX, ev.clientY);
      };
      const onUp = (ev: PointerEvent) => {
        if (ev.pointerId !== dragPointerIdRef.current) return;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);
        windowDragListenersRef.current = null;
        try {
          el.releasePointerCapture(ev.pointerId);
        } catch {
          /* */
        }
        dragModeRef.current = null;
        dragPointerIdRef.current = null;
      };
      windowDragListenersRef.current = { move: onMove, up: onUp };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
    },
    [applyIrisPointerMove]
  );

  useEffect(() => {
    return () => {
      const h = windowDragListenersRef.current;
      if (h) {
        window.removeEventListener("pointermove", h.move);
        window.removeEventListener("pointerup", h.up);
        window.removeEventListener("pointercancel", h.up);
        windowDragListenersRef.current = null;
      }
      dragModeRef.current = null;
      dragPointerIdRef.current = null;
    };
  }, []);

  const summary = useMemo(() => {
    if (!result) return "No analysis yet.";
    let text =
      result.warnings.length === 0
        ? "Pupil appears within expected range based on this heuristic check (non-medical)."
        : result.warnings.join(" ");
    const manual = adjustedIrisRadius != null || adjustedIrisCenter != null;
    if (manual && manualProportion && !manualProportion.isProportional) {
      text += " With your iris adjustment, pupil/iris proportion reads as atypical.";
    }
    if (manual && centering && centering.normalized >= 0.15) {
      text += " With your iris adjustment, pupil centering vs iris suggests decentration.";
    }
    return text;
  }, [result, manualProportion, centering, adjustedIrisRadius, adjustedIrisCenter]);

  const handleDownloadResultsCsv = useCallback(() => {
    if (!result) return;

    const ratio = manualProportion?.ratio ?? result.proportionality?.pupilToIrisRatio ?? null;
    const propTypical =
      manualProportion?.isProportional ?? result.proportionality?.isProportional ?? null;
    const manualIris = adjustedIrisRadius != null || adjustedIrisCenter != null;
    const assess = effectiveAssessment;

    const rows: Array<[string, string]> = [
      ["exported_at_utc", new Date().toISOString()],
      ["source_image_filename", file?.name ?? ""],
      ["image_width_px", result.imageWidth != null ? String(result.imageWidth) : ""],
      ["image_height_px", result.imageHeight != null ? String(result.imageHeight) : ""],
      ["pupil_diameter_px", result.pupilDiameterPixels != null ? result.pupilDiameterPixels.toFixed(4) : ""],
      ["circularity", result.circularity != null ? result.circularity.toFixed(6) : ""],
      ["contour_area_px2", result.contourArea != null ? result.contourArea.toFixed(4) : ""],
    ];

    if (result.fittedCircle) {
      rows.push(
        ["fitted_circle_center_x_px", result.fittedCircle.centerX.toFixed(4)],
        ["fitted_circle_center_y_px", result.fittedCircle.centerY.toFixed(4)],
        ["fitted_circle_radius_px", result.fittedCircle.radius.toFixed(4)],
        ["fitted_circle_rms_deviation_px", result.fittedCircle.rmsDeviationPx.toFixed(4)],
        ["fitted_circle_max_deviation_px", result.fittedCircle.maxDeviationPx.toFixed(4)]
      );
    }

    rows.push(
      ["pupil_iris_ratio", ratio != null ? ratio.toFixed(6) : ""],
      ["pupil_iris_ratio_percent", ratio != null ? (ratio * 100).toFixed(2) : ""],
      ["proportion_typical", propTypical === null ? "" : propTypical ? "yes" : "no"],
      ["iris_manually_adjusted", manualIris ? "yes" : "no"],
      ["adjusted_iris_center_x_px", irisCenter ? irisCenter.x.toFixed(4) : ""],
      ["adjusted_iris_center_y_px", irisCenter ? irisCenter.y.toFixed(4) : ""],
      ["adjusted_iris_radius_px", irisRadius > 0 ? irisRadius.toFixed(4) : ""]
    );

    if (centering) {
      rows.push(
        ["pupil_iris_center_offset_px", centering.offsetPx.toFixed(4)],
        ["pupil_iris_center_offset_iris_radii", centering.normalized.toFixed(6)],
        ["centering_label", centering.label]
      );
    }

    if (assess) {
      rows.push(
        ["heuristic_large", assess.isTooLarge ? "yes" : "no"],
        ["heuristic_small", assess.isTooSmall ? "yes" : "no"],
        ["heuristic_irregular_decentered", assess.isIrregular ? "yes" : "no"]
      );
    }

    rows.push(
      ["summary_text", summary],
      ["api_warnings_pipe_separated", result.warnings.join(" | ")],
      ["processing_notes", result.processingNotes ?? ""],
      ["proportion_note", result.proportionality?.note ?? ""]
    );

    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    triggerCsvDownload(`eye-metrics-${stamp}.csv`, rowsToCsv(rows));
  }, [
    result,
    file,
    manualProportion,
    centering,
    effectiveAssessment,
    summary,
    adjustedIrisRadius,
    adjustedIrisCenter,
    irisCenter,
    irisRadius,
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);

    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please choose an eye photo first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const { data } = await axios.post<AnalysisResponse>(`${API_BASE}/analyze-eye`, form, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setResult(data);
      setAdjustedIrisRadius(null);
      setAdjustedIrisCenter(null);
    } catch (err: any) {
      console.error(err);
      setError(
        err?.response?.data?.detail ??
          "Unable to analyze this image. Please try a clear close-up of one eye."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
    <main className="min-h-screen flex items-center justify-center px-4 py-10 bg-transparent">
      <div className="max-w-5xl w-full">
        <div className="relative rounded-3xl gradient-border p-[1px] shadow-soft">
          <div className="ui-shell-gradient rounded-3xl border border-white/60 px-6 py-8 md:px-10 md:py-10 shadow-card backdrop-blur-md">
            <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
              <div>
                <h1 className="text-2xl md:text-3xl font-bold tracking-tight bg-gradient-to-r from-slate-800 via-sky-600 to-violet-600 bg-clip-text text-transparent">
                  Eye Metrics Analyzer
                </h1>
                <p className="mt-2 text-sm md:text-base text-ink-muted max-w-xl leading-relaxed">
                  Upload a clear photo of a single eye. We&apos;ll normalize it, estimate pupil size
                  and shape, and flag potential concerns. This is a heuristic tool only, not
                  medical advice.
                </p>
              </div>
            </header>

            <div className="grid gap-8 md:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] items-start">
              <section className="space-y-5">
                <form
                  onSubmit={handleSubmit}
                  className="ui-card-gradient space-y-5 rounded-2xl border border-white/80 p-5 md:p-6 shadow-card ring-1 ring-sky-500/10"
                >
                  <div className="flex flex-col gap-2">
                    <label className="text-sm font-medium text-ink">
                      Eye photo
                      <span className="ml-1 text-xs font-normal text-fuchsia-600">(required)</span>
                    </label>
                    <p className="text-xs text-ink-muted">
                      Use a sharp, well-lit close-up of one eye. Avoid heavy reflections or
                      sunglasses.
                    </p>
                    <label className="mt-2 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-sky-200/80 bg-gradient-to-br from-white to-sky-50/60 px-4 py-6 text-center transition-all hover:border-sky-400 hover:from-sky-50 hover:to-cyan-50 hover:shadow-md hover:shadow-sky-200/40">
                      <span className="text-sm font-medium text-ink">
                        {file ? file.name : "Click to choose a file"}
                      </span>
                      <span className="text-xs text-ink-faint">
                        PNG, JPG or HEIC up to ~10 MB
                      </span>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-cyan-400 via-sky-500 to-indigo-500 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-sky-500/35 transition-all hover:from-cyan-300 hover:via-sky-400 hover:to-indigo-400 hover:shadow-lg hover:shadow-indigo-500/25 disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:from-cyan-400 disabled:hover:via-sky-500 disabled:hover:to-indigo-500"
                  >
                    {loading ? "Analyzing eye..." : "Analyze pupil"}
                  </button>

                  {error && (
                    <p className="text-xs text-red-900 border border-rose-300/90 bg-gradient-to-r from-rose-100 to-orange-50 rounded-lg px-3 py-2 shadow-sm">
                      {error}
                    </p>
                  )}

                  <p className="text-[11px] text-ink-faint leading-relaxed">
                    This tool performs basic computer-vision heuristics on a 2D image without any
                    calibration reference. Measurements are in pixels and should not be interpreted
                    as clinical guidance. Always consult an eye-care professional for any
                    concerns.
                  </p>
                </form>
              </section>

              <section className="space-y-4">
                <div className="ui-card-gradient-alt rounded-2xl border border-white/80 p-4 md:p-5 shadow-card ring-1 ring-violet-500/10">
                  <h2 className="text-sm font-semibold bg-gradient-to-r from-slate-800 to-violet-600 bg-clip-text text-transparent mb-3">
                    Preview
                  </h2>
                  <div className="aspect-video w-full overflow-hidden rounded-xl border border-violet-100 bg-gradient-to-br from-slate-50 via-white to-violet-50/50 flex items-center justify-center relative shadow-inner">
                    {hasResult && result?.processedImageBase64 && result?.imageWidth && result?.imageHeight ? (
                      <div className="relative w-full h-full flex items-center justify-center">
                        <img
                          src={result.processedImageBase64}
                          alt="Processed eye"
                          className="max-w-full max-h-full object-contain"
                          style={{ aspectRatio: `${result.imageWidth} / ${result.imageHeight}` }}
                        />
                        <svg
                          ref={svgRef}
                          viewBox={`0 0 ${result.imageWidth} ${result.imageHeight}`}
                          className="absolute inset-0 w-full h-full"
                          preserveAspectRatio="xMidYMid meet"
                        >
                          {result.fittedCircle && (
                            <circle
                              cx={result.fittedCircle.centerX}
                              cy={result.fittedCircle.centerY}
                              r={result.fittedCircle.radius}
                              fill="none"
                              stroke="rgb(56 189 248)"
                              strokeWidth={1.5}
                              strokeDasharray="4 2"
                              opacity={0.9}
                              pointerEvents="none"
                            />
                          )}
                          {irisPointsForDisplay.map((p, i) => (
                            <circle key={`iris-${i}`} cx={p.x} cy={p.y} r={2.5} fill="rgb(251 191 36)" fillOpacity={0.9} pointerEvents="none" />
                          ))}
                          {result.pupilPoints?.map((p, i) => (
                            <circle key={`pupil-${i}`} cx={p.x} cy={p.y} r={2} fill="rgb(34 211 238)" fillOpacity={1} pointerEvents="none" />
                          ))}
                          {irisCenter && irisRadius > 0 && (
                            <>
                              <circle
                                cx={irisCenter.x}
                                cy={irisCenter.y}
                                r={8}
                                fill="rgb(251 191 36)"
                                stroke="white"
                                strokeWidth={1.5}
                                opacity={0.9}
                                cursor="grab"
                                style={{ touchAction: "none" }}
                                onPointerDown={(e) => startIrisDrag(e, "center")}
                                aria-label="Drag to move iris circle"
                              />
                              <circle
                                cx={irisCenter.x + irisRadius}
                                cy={irisCenter.y}
                                r={8}
                                fill="rgb(251 191 36)"
                                stroke="white"
                                strokeWidth={1.5}
                                opacity={0.95}
                                cursor="grab"
                                style={{ touchAction: "none" }}
                                onPointerDown={(e) => startIrisDrag(e, "radius")}
                                aria-label="Drag to resize iris circle"
                              />
                            </>
                          )}
                        </svg>
                        <div className="absolute bottom-2 left-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-ink-muted bg-white/95 backdrop-blur-sm rounded-md px-2 py-1 border border-line shadow-sm">
                          <span><span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-1" />Pupil</span>
                          <span><span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />Iris</span>
                          <span className="text-sky-400">— Fitted circle</span>
                          {result?.irisPoints?.length ? (
                            <span className="text-amber-600/90">· Drag center to move iris, edge dot to resize</span>
                          ) : null}
                        </div>
                      </div>
                    ) : preview ? (
                      <Image
                        src={preview}
                        alt="Eye preview"
                        width={640}
                        height={360}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <p className="text-xs text-ink-muted">
                        Choose a photo to see it here before analysis.
                      </p>
                    )}
                  </div>
                </div>

                <div className="ui-card-gradient rounded-2xl border border-white/80 p-4 md:p-5 space-y-3 shadow-card ring-1 ring-emerald-500/10">
                  <h2 className="text-sm font-semibold bg-gradient-to-r from-slate-800 to-teal-600 bg-clip-text text-transparent">
                    Pupil metrics
                  </h2>
                  <p className="text-xs text-ink-muted">Heuristic measurements (pixels, not mm).</p>

                  <div className="grid grid-cols-3 gap-3 text-center text-xs">
                    <div className="ui-metric-tile rounded-xl border border-sky-100 px-2 py-2.5 shadow-sm ring-1 ring-sky-500/5">
                      <p className="text-[11px] text-ink-muted mb-1">Diameter</p>
                      <p className="text-sm font-semibold text-ink">
                        {hasResult && result?.pupilDiameterPixels
                          ? `${result.pupilDiameterPixels.toFixed(1)} px`
                          : "–"}
                      </p>
                    </div>
                    <div className="ui-metric-tile rounded-xl border border-indigo-100 px-2 py-2.5 shadow-sm ring-1 ring-indigo-500/5">
                      <p className="text-[11px] text-ink-muted mb-1">Circularity</p>
                      <p className="text-sm font-semibold text-ink">
                        {hasResult && result?.circularity
                          ? result.circularity.toFixed(3)
                          : "–"}
                      </p>
                    </div>
                    <div className="ui-metric-tile rounded-xl border border-violet-100 px-2 py-2.5 shadow-sm ring-1 ring-violet-500/5">
                      <p className="text-[11px] text-ink-muted mb-1">Area</p>
                      <p className="text-sm font-semibold text-ink">
                        {hasResult && result?.contourArea
                          ? `${result.contourArea.toFixed(0)} px²`
                          : "–"}
                      </p>
                    </div>
                  </div>

                  {hasResult && result?.fittedCircle && (
                    <div className="mt-3 rounded-xl border border-cyan-100 bg-gradient-to-br from-white to-cyan-50/40 px-3 py-3 text-xs space-y-1 shadow-sm">
                      <p className="font-medium text-ink">Circle fit (vs perfect circle)</p>
                      <p className="text-ink-muted">
                        RMS deviation: {result.fittedCircle.rmsDeviationPx.toFixed(2)} px · Max: {result.fittedCircle.maxDeviationPx.toFixed(2)} px
                      </p>
                      {result.deviationFromPerfectCircle && (
                        <p className="text-[11px] text-ink-faint mt-0.5">{result.deviationFromPerfectCircle}</p>
                      )}
                    </div>
                  )}

                  {hasResult && (result?.proportionality || manualProportion) && (
                    <div className={`mt-3 rounded-xl border-2 px-3 py-3 text-xs space-y-1 shadow-sm ${
                      (manualProportion?.isProportional ?? result?.proportionality?.isProportional)
                        ? "border-emerald-300 bg-gradient-to-br from-emerald-50 via-teal-50/80 to-cyan-50/50 ring-2 ring-emerald-200/50"
                        : "border-rose-300 bg-gradient-to-br from-rose-50 via-orange-50/70 to-amber-50/40 ring-2 ring-rose-200/50"
                    }`}>
                      <p className="font-medium text-ink">Pupil / Iris proportion</p>
                      <p className="text-ink-muted">
                        Ratio: {(manualProportion?.ratio ?? result?.proportionality?.pupilToIrisRatio) != null
                          ? ((manualProportion?.ratio ?? result?.proportionality?.pupilToIrisRatio)! * 100).toFixed(1)
                          : "–"}%
                        {(manualProportion ?? result?.proportionality) && (
                          <span className={`ml-1 font-medium ${
                            (manualProportion?.isProportional ?? result?.proportionality?.isProportional) ? "text-emerald-800" : "text-rose-700"
                          }`}>
                            ({(manualProportion?.isProportional ?? result?.proportionality?.isProportional) ? "typical" : "atypical"})
                          </span>
                        )}
                      </p>
                      <p className="text-[11px] text-ink-faint mt-0.5">
                        {adjustedIrisRadius != null || adjustedIrisCenter != null
                          ? "Iris adjusted manually. Ratio reflects your circle."
                          : result?.proportionality?.note}
                      </p>
                    </div>
                  )}

                  {hasResult && centering && (
                    <div className={`mt-3 rounded-xl border-2 px-3 py-3 text-xs space-y-1 shadow-sm ${
                      centering.normalized < 0.15
                        ? "border-emerald-300 bg-gradient-to-br from-emerald-50 via-teal-50/80 to-cyan-50/50 ring-2 ring-emerald-200/50"
                        : centering.normalized < 0.30
                          ? "border-slate-200 bg-gradient-to-br from-white to-slate-50/90 ring-1 ring-slate-200/60"
                          : "border-rose-300 bg-gradient-to-br from-rose-50 via-orange-50/70 to-amber-50/40 ring-2 ring-rose-200/50"
                    }`}>
                      <p className="font-medium text-ink">Pupil centering (vs iris)</p>
                      <p className="text-ink-muted">
                        Offset: {centering.offsetPx.toFixed(1)} px ({centering.normalized.toFixed(2)}× iris radius)
                        <span className={`ml-1 font-medium ${
                          centering.normalized < 0.15 ? "text-emerald-800" : centering.normalized < 0.30 ? "text-ink-muted" : "text-rose-700"
                        }`}>
                          — {centering.label}
                        </span>
                      </p>
                      <p className="text-[11px] text-ink-faint mt-0.5">
                        How far the pupil center is from the iris center. Adjust the iris circle to match the eye, then read this.
                      </p>
                    </div>
                  )}

                  <div className="ui-heuristic-panel mt-3 rounded-xl border border-violet-200/80 px-3 py-3 text-xs space-y-1.5 shadow-sm ring-1 ring-violet-300/30">
                    <p className="font-semibold bg-gradient-to-r from-violet-700 to-fuchsia-600 bg-clip-text text-transparent">
                      Heuristic assessment
                    </p>
                    <p className="text-ink-muted leading-relaxed">{summary}</p>
                    {result?.processingNotes && (
                      <p className="text-[11px] text-ink-faint mt-1">
                        Notes: {result.processingNotes}
                      </p>
                    )}
                    {hasResult && effectiveAssessment && (
                      <ul className="mt-1 grid grid-cols-3 gap-1.5 text-[11px]">
                        <li
                          className={`rounded-full border px-2 py-1 text-center font-medium transition-shadow ${
                            effectiveAssessment.isTooLarge
                              ? "border-2 border-orange-400 bg-gradient-to-r from-orange-400 to-rose-400 text-white shadow-md shadow-orange-400/35 ring-2 ring-orange-200 ring-offset-2 ring-offset-white"
                              : "border border-slate-200/90 bg-gradient-to-b from-white to-slate-50 text-slate-400"
                          }`}
                        >
                          Large
                        </li>
                        <li
                          className={`rounded-full border px-2 py-1 text-center font-medium transition-shadow ${
                            effectiveAssessment.isTooSmall
                              ? "border-2 border-amber-400 bg-gradient-to-r from-amber-300 to-yellow-300 text-amber-950 shadow-md shadow-amber-400/35 ring-2 ring-amber-200 ring-offset-2 ring-offset-white"
                              : "border border-slate-200/90 bg-gradient-to-b from-white to-slate-50 text-slate-400"
                          }`}
                        >
                          Small
                        </li>
                        <li
                          className={`rounded-full border px-2 py-1 text-center font-medium transition-shadow ${
                            effectiveAssessment.isIrregular
                              ? "border-2 border-violet-400 bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white shadow-md shadow-violet-400/35 ring-2 ring-violet-200 ring-offset-2 ring-offset-white"
                              : "border border-slate-200/90 bg-gradient-to-b from-white to-slate-50 text-slate-400"
                          }`}
                        >
                          Irregular
                        </li>
                      </ul>
                    )}
                    {hasResult && (
                      <button
                        type="button"
                        onClick={handleDownloadResultsCsv}
                        className="mt-3 w-full rounded-xl border border-violet-200/90 bg-gradient-to-r from-white via-violet-50/80 to-fuchsia-50/60 px-3 py-2.5 text-xs font-semibold text-violet-900 shadow-sm transition-all hover:border-violet-300 hover:shadow-md hover:shadow-violet-200/40"
                      >
                        Download results as CSV
                      </button>
                    )}
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </main>
    </>
  );
}

