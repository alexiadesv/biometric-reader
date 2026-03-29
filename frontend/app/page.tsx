"use client";

import axios from "axios";
import Image from "next/image";
import React, { useCallback, useMemo, useRef, useState } from "react";

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

  const clientToSvg = useCallback((clientX: number, clientY: number): { x: number; y: number } | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    return svgPt ? { x: svgPt.x, y: svgPt.y } : null;
  }, []);

  const handleRadiusHandlePointerDown = useCallback((e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragModeRef.current = "radius";
  }, []);

  const handleCenterHandlePointerDown = useCallback((e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragModeRef.current = "center";
  }, []);

  const handleIrisPointerMove = useCallback((e: React.PointerEvent) => {
    const mode = dragModeRef.current;
    if (!mode || !pupilCenter) return;
    const pt = clientToSvg(e.clientX, e.clientY);
    if (!pt) return;
    if (mode === "radius" && irisCenter) {
      const r = Math.hypot(pt.x - irisCenter.x, pt.y - irisCenter.y);
      if (r > pupilRadius * 1.1 && r < 500) setAdjustedIrisRadius(r);
    } else if (mode === "center") {
      setAdjustedIrisCenter({ x: pt.x, y: pt.y });
    }
  }, [pupilCenter, pupilRadius, irisCenter, clientToSvg]);

  const handleIrisPointerUp = useCallback(() => {
    dragModeRef.current = null;
  }, []);

  const summary = useMemo(() => {
    if (!result) return "No analysis yet.";
    if (result.warnings.length === 0) {
      return "Pupil appears within expected range based on this heuristic check (non-medical).";
    }
    return result.warnings.join(" ");
  }, [result]);

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
    <main className="min-h-screen flex items-center justify-center px-4">
      <div className="max-w-5xl w-full">
        <div className="relative rounded-3xl gradient-border p-[1px] shadow-soft">
          <div className="rounded-3xl bg-slate-950/80 border border-slate-800/60 backdrop-blur-xl px-6 py-8 md:px-10 md:py-10">
            <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
              <div>
                <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-slate-50">
                  Eye Metrics Analyzer
                </h1>
                <p className="mt-2 text-sm md:text-base text-slate-400 max-w-xl">
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
                  className="space-y-5 rounded-2xl border border-slate-800 bg-slate-950/60 p-5 md:p-6"
                >
                  <div className="flex flex-col gap-2">
                    <label className="text-sm font-medium text-slate-200">
                      Eye photo
                      <span className="ml-1 text-xs font-normal text-sky-400">(required)</span>
                    </label>
                    <p className="text-xs text-slate-500">
                      Use a sharp, well-lit close-up of one eye. Avoid heavy reflections or
                      sunglasses.
                    </p>
                    <label className="mt-2 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-slate-700 bg-slate-900/60 px-4 py-6 text-center hover:border-sky-500/70 hover:bg-slate-900/80 transition-colors">
                      <span className="text-sm font-medium text-slate-100">
                        {file ? file.name : "Click to choose a file"}
                      </span>
                      <span className="text-xs text-slate-500">
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
                    className="inline-flex items-center justify-center rounded-full bg-sky-500 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-sky-500/30 hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-60 transition-colors"
                  >
                    {loading ? "Analyzing eye..." : "Analyze pupil"}
                  </button>

                  {error && (
                    <p className="text-xs text-rose-400 border border-rose-900/50 bg-rose-950/40 rounded-lg px-3 py-2">
                      {error}
                    </p>
                  )}

                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    This tool performs basic computer-vision heuristics on a 2D image without any
                    calibration reference. Measurements are in pixels and should not be interpreted
                    as clinical guidance. Always consult an eye-care professional for any
                    concerns.
                  </p>
                </form>
              </section>

              <section className="space-y-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4 md:p-5">
                  <h2 className="text-sm font-semibold text-slate-100 mb-3">Preview</h2>
                  <div className="aspect-video w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-900/80 flex items-center justify-center relative">
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
                          onPointerMove={handleIrisPointerMove}
                          onPointerUp={handleIrisPointerUp}
                          onPointerLeave={handleIrisPointerUp}
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
                                onPointerDown={handleCenterHandlePointerDown}
                                onPointerMove={handleIrisPointerMove}
                                onPointerUp={handleIrisPointerUp}
                                onPointerCancel={handleIrisPointerUp}
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
                                onPointerDown={handleRadiusHandlePointerDown}
                                onPointerMove={handleIrisPointerMove}
                                onPointerUp={handleIrisPointerUp}
                                onPointerCancel={handleIrisPointerUp}
                                aria-label="Drag to resize iris circle"
                              />
                            </>
                          )}
                        </svg>
                        <div className="absolute bottom-2 left-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-300">
                          <span><span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-1" />Pupil</span>
                          <span><span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />Iris</span>
                          <span className="text-sky-400">— Fitted circle</span>
                          {result?.irisPoints?.length ? (
                            <span className="text-amber-300/90">· Drag center to move iris, edge dot to resize</span>
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
                      <p className="text-xs text-slate-500">
                        Choose a photo to see it here before analysis.
                      </p>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4 md:p-5 space-y-3">
                  <h2 className="text-sm font-semibold text-slate-100">Pupil metrics</h2>
                  <p className="text-xs text-slate-400">Heuristic measurements (pixels, not mm).</p>

                  <div className="grid grid-cols-3 gap-3 text-center text-xs">
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 px-2 py-2.5">
                      <p className="text-[11px] text-slate-400 mb-1">Diameter</p>
                      <p className="text-sm font-semibold text-slate-50">
                        {hasResult && result?.pupilDiameterPixels
                          ? `${result.pupilDiameterPixels.toFixed(1)} px`
                          : "–"}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 px-2 py-2.5">
                      <p className="text-[11px] text-slate-400 mb-1">Circularity</p>
                      <p className="text-sm font-semibold text-slate-50">
                        {hasResult && result?.circularity
                          ? result.circularity.toFixed(3)
                          : "–"}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 px-2 py-2.5">
                      <p className="text-[11px] text-slate-400 mb-1">Area</p>
                      <p className="text-sm font-semibold text-slate-50">
                        {hasResult && result?.contourArea
                          ? `${result.contourArea.toFixed(0)} px²`
                          : "–"}
                      </p>
                    </div>
                  </div>

                  {hasResult && result?.fittedCircle && (
                    <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-3 text-xs space-y-1">
                      <p className="font-medium text-slate-100">Circle fit (vs perfect circle)</p>
                      <p className="text-slate-400">
                        RMS deviation: {result.fittedCircle.rmsDeviationPx.toFixed(2)} px · Max: {result.fittedCircle.maxDeviationPx.toFixed(2)} px
                      </p>
                      {result.deviationFromPerfectCircle && (
                        <p className="text-[11px] text-slate-500 mt-0.5">{result.deviationFromPerfectCircle}</p>
                      )}
                    </div>
                  )}

                  {hasResult && (result?.proportionality || manualProportion) && (
                    <div className={`mt-3 rounded-xl border px-3 py-3 text-xs space-y-1 ${
                      (manualProportion?.isProportional ?? result?.proportionality?.isProportional)
                        ? "border-emerald-700/50 bg-emerald-950/30"
                        : "border-amber-700/50 bg-amber-950/30"
                    }`}>
                      <p className="font-medium text-slate-100">Pupil / Iris proportion</p>
                      <p className="text-slate-300">
                        Ratio: {(manualProportion?.ratio ?? result?.proportionality?.pupilToIrisRatio) != null
                          ? ((manualProportion?.ratio ?? result?.proportionality?.pupilToIrisRatio)! * 100).toFixed(1)
                          : "–"}%
                        {(manualProportion ?? result?.proportionality) && (
                          <span className={`ml-1 font-medium ${
                            (manualProportion?.isProportional ?? result?.proportionality?.isProportional) ? "text-emerald-400" : "text-amber-400"
                          }`}>
                            ({(manualProportion?.isProportional ?? result?.proportionality?.isProportional) ? "typical" : "atypical"})
                          </span>
                        )}
                      </p>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        {adjustedIrisRadius != null || adjustedIrisCenter != null
                          ? "Iris adjusted manually. Ratio reflects your circle."
                          : result?.proportionality?.note}
                      </p>
                    </div>
                  )}

                  {hasResult && centering && (
                    <div className={`mt-3 rounded-xl border px-3 py-3 text-xs space-y-1 ${
                      centering.normalized < 0.15
                        ? "border-emerald-700/50 bg-emerald-950/30"
                        : centering.normalized < 0.30
                          ? "border-slate-700 bg-slate-900/70"
                          : "border-amber-700/50 bg-amber-950/30"
                    }`}>
                      <p className="font-medium text-slate-100">Pupil centering (vs iris)</p>
                      <p className="text-slate-300">
                        Offset: {centering.offsetPx.toFixed(1)} px ({centering.normalized.toFixed(2)}× iris radius)
                        <span className={`ml-1 font-medium ${
                          centering.normalized < 0.15 ? "text-emerald-400" : centering.normalized < 0.30 ? "text-slate-400" : "text-amber-400"
                        }`}>
                          — {centering.label}
                        </span>
                      </p>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        How far the pupil center is from the iris center. Adjust the iris circle to match the eye, then read this.
                      </p>
                    </div>
                  )}

                  <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-3 text-xs space-y-1.5">
                    <p className="font-medium text-slate-100">Heuristic assessment</p>
                    <p className="text-slate-300 leading-relaxed">{summary}</p>
                    {result?.processingNotes && (
                      <p className="text-[11px] text-slate-500 mt-1">
                        Notes: {result.processingNotes}
                      </p>
                    )}
                    {hasResult && (
                      <ul className="mt-1 grid grid-cols-3 gap-1.5 text-[11px]">
                        <li
                          className={`rounded-full border px-2 py-1 text-center ${
                            result.assessment.isTooLarge
                              ? "border-amber-400/70 bg-amber-500/10 text-amber-200"
                              : "border-slate-700 bg-slate-900/80 text-slate-400"
                          }`}
                        >
                          Large
                        </li>
                        <li
                          className={`rounded-full border px-2 py-1 text-center ${
                            result.assessment.isTooSmall
                              ? "border-amber-400/70 bg-amber-500/10 text-amber-200"
                              : "border-slate-700 bg-slate-900/80 text-slate-400"
                          }`}
                        >
                          Small
                        </li>
                        <li
                          className={`rounded-full border px-2 py-1 text-center ${
                            result.assessment.isIrregular
                              ? "border-rose-400/70 bg-rose-500/10 text-rose-200"
                              : "border-slate-700 bg-slate-900/80 text-slate-400"
                          }`}
                        >
                          Irregular
                        </li>
                      </ul>
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

