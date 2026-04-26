"use client";

import axios from "axios";
import Image from "next/image";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

type PupilAssessment = {
  isTooLarge: boolean;
  isTooSmall: boolean;
  isIrregular: boolean;
  isOccludedByLid: boolean;
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
  upperLidPoints?: Point2D[] | null;
  lowerLidPoints?: Point2D[] | null;
  pupilVisiblePercent?: number | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/** Centroid + mean radial distance — matches detected iris geometry (backend uses min-enclosing circle center). */
function getIrisCenterAndRadiusFromPoints(points: Point2D[]): { center: Point2D; radius: number } | null {
  if (!points.length) return null;
  const n = points.length;
  const cx = points.reduce((s, p) => s + p.x, 0) / n;
  const cy = points.reduce((s, p) => s + p.y, 0) / n;
  let sumR = 0;
  for (const p of points) {
    sumR += Math.hypot(p.x - cx, p.y - cy);
  }
  const radius = sumR / n;
  return { center: { x: cx, y: cy }, radius };
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

function computePupilVisiblePercent(
  pupilCx: number,
  pupilCy: number,
  pupilR: number,
  upperPts: Point2D[],
  lowerPts: Point2D[],
): number {
  if (pupilR <= 0) return 0;
  const N = 100;
  const xVals: number[] = [];
  for (let i = 0; i <= N; i++) xVals.push(pupilCx - pupilR + (2 * pupilR * i) / N);

  function interpY(pts: Point2D[], x: number): number | null {
    if (pts.length < 2) return null;
    if (x < pts[0].x || x > pts[pts.length - 1].x) return null;
    let lo = 0,
      hi = pts.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (pts[mid].x <= x) lo = mid;
      else hi = mid;
    }
    const span = pts[hi].x - pts[lo].x;
    const frac = span ? (x - pts[lo].x) / span : 0;
    return pts[lo].y + frac * (pts[hi].y - pts[lo].y);
  }

  let visArea = 0,
    totArea = 0;
  for (let i = 0; i < N; i++) {
    const xMid = (xVals[i] + xVals[i + 1]) / 2;
    const dx = xVals[i + 1] - xVals[i];
    const off = xMid - pupilCx;
    if (Math.abs(off) >= pupilR) continue;
    const halfH = Math.sqrt(pupilR * pupilR - off * off);
    const pTop = pupilCy - halfH;
    const pBot = pupilCy + halfH;
    let vTop = pTop,
      vBot = pBot;
    const uy = interpY(upperPts, xMid);
    if (uy !== null) vTop = Math.max(vTop, uy);
    const ly = interpY(lowerPts, xMid);
    if (ly !== null) vBot = Math.min(vBot, ly);
    visArea += Math.max(0, vBot - vTop) * dx;
    totArea += 2 * halfH * dx;
  }
  if (totArea <= 0) return 0;
  return Math.max(0, Math.min(100, (100 * visArea) / totArea));
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
  const [upperLidOffset, setUpperLidOffset] = useState(0);
  const [upperLidOffsetX, setUpperLidOffsetX] = useState(0);
  const [lowerLidOffset, setLowerLidOffset] = useState(0);
  const [lowerLidOffsetX, setLowerLidOffsetX] = useState(0);
  const [upperLidAngle, setUpperLidAngle] = useState(0);
  const [lowerLidAngle, setLowerLidAngle] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragModeRef = useRef<"radius" | "center" | "upper-lid" | "lower-lid" | "upper-lid-rotate" | "lower-lid-rotate" | null>(null);
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
  const initialIrisGeometry = useMemo(() => {
    if (!result?.irisPoints?.length) return null;
    return getIrisCenterAndRadiusFromPoints(result.irisPoints);
  }, [result?.irisPoints]);
  const initialIrisRadius = initialIrisGeometry?.radius ?? 0;
  const irisRadius = adjustedIrisRadius ?? initialIrisRadius;
  const irisCenter = adjustedIrisCenter ?? initialIrisGeometry?.center ?? pupilCenter;

  const irisPointsForDisplay = useMemo(() => {
    if (!irisCenter || irisRadius <= 0) return result?.irisPoints ?? [];
    const pts: Point2D[] = [];
    for (let i = 0; i < 64; i++) {
      const t = (2 * Math.PI * i) / 64;
      pts.push({ x: irisCenter.x + irisRadius * Math.cos(t), y: irisCenter.y + irisRadius * Math.sin(t) });
    }
    return pts;
  }, [irisCenter, irisRadius, result?.irisPoints]);

  const upperLidDisplay = useMemo(() => {
    if (!result?.upperLidPoints?.length) return [];
    const pts = result.upperLidPoints;
    const midIdx = Math.floor(pts.length / 2);
    const cx = pts[midIdx].x + upperLidOffsetX;
    const cy = pts[midIdx].y + upperLidOffset;
    const cos = Math.cos(upperLidAngle);
    const sin = Math.sin(upperLidAngle);
    return pts.map((p) => {
      const dx = p.x + upperLidOffsetX - cx;
      const dy = p.y + upperLidOffset - cy;
      return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
    });
  }, [result?.upperLidPoints, upperLidOffset, upperLidOffsetX, upperLidAngle]);

  const lowerLidDisplay = useMemo(() => {
    if (!result?.lowerLidPoints?.length) return [];
    const pts = result.lowerLidPoints;
    const midIdx = Math.floor(pts.length / 2);
    const cx = pts[midIdx].x + lowerLidOffsetX;
    const cy = pts[midIdx].y + lowerLidOffset;
    const cos = Math.cos(lowerLidAngle);
    const sin = Math.sin(lowerLidAngle);
    return pts.map((p) => {
      const dx = p.x + lowerLidOffsetX - cx;
      const dy = p.y + lowerLidOffset - cy;
      return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
    });
  }, [result?.lowerLidPoints, lowerLidOffset, lowerLidOffsetX, lowerLidAngle]);

  const pupilVisiblePercent = useMemo(() => {
    if (!pupilCenter || pupilRadius <= 0) return result?.pupilVisiblePercent ?? null;
    const hasLidAdjustment = upperLidOffset !== 0 || upperLidOffsetX !== 0 || lowerLidOffset !== 0 || lowerLidOffsetX !== 0 || upperLidAngle !== 0 || lowerLidAngle !== 0;
    if (!hasLidAdjustment) return result?.pupilVisiblePercent ?? null;
    if (!upperLidDisplay.length && !lowerLidDisplay.length) return result?.pupilVisiblePercent ?? null;
    return Math.round(
      computePupilVisiblePercent(pupilCenter.x, pupilCenter.y, pupilRadius, upperLidDisplay, lowerLidDisplay) * 10,
    ) / 10;
  }, [pupilCenter, pupilRadius, upperLidDisplay, lowerLidDisplay, upperLidOffset, upperLidOffsetX, lowerLidOffset, lowerLidOffsetX, upperLidAngle, lowerLidAngle, result?.pupilVisiblePercent]);

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
    if (normalized < 0.25) label = "well centered";
    else if (normalized < 0.40) label = "mildly decentered";
    else label = "markedly decentered";
    return { offsetPx, normalized, label };
  }, [pupilCenter, irisCenter, irisRadius]);

  const effectiveAssessment = useMemo((): PupilAssessment | null => {
    if (!result) return null;
    const typicalMin = 0.15;
    const typicalMax = 0.55;
    let isTooLarge = result.assessment.isTooLarge;
    let isTooSmall = manualProportion
      ? manualProportion.ratio < typicalMin
      : result.assessment.isTooSmall;
    if (manualProportion && manualProportion.ratio > typicalMax) {
      isTooLarge = true;
    }
    const isIrregular = !!(centering && centering.normalized >= 0.25);
    const isOccludedByLid = pupilVisiblePercent != null && pupilVisiblePercent < 98;
    return { isTooLarge, isTooSmall, isIrregular, isOccludedByLid };
  }, [result, manualProportion, centering, pupilVisiblePercent]);

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

  const lidDragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const lidDragStartOffsetRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const lidRotatePivotRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const lidRotateStartAngleRef = useRef<number>(0);
  const lidRotateBaseAngleRef = useRef<number>(0);

  const applyPointerMove = useCallback((clientX: number, clientY: number) => {
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
    } else if (mode === "upper-lid") {
      const dx = pt.x - lidDragStartRef.current.x;
      const dy = pt.y - lidDragStartRef.current.y;
      setUpperLidOffsetX(lidDragStartOffsetRef.current.x + dx);
      setUpperLidOffset(lidDragStartOffsetRef.current.y + dy);
    } else if (mode === "lower-lid") {
      const dx = pt.x - lidDragStartRef.current.x;
      const dy = pt.y - lidDragStartRef.current.y;
      setLowerLidOffsetX(lidDragStartOffsetRef.current.x + dx);
      setLowerLidOffset(lidDragStartOffsetRef.current.y + dy);
    } else if (mode === "upper-lid-rotate" || mode === "lower-lid-rotate") {
      const pv = lidRotatePivotRef.current;
      const curAngle = Math.atan2(pt.y - pv.y, pt.x - pv.x);
      const delta = curAngle - lidRotateStartAngleRef.current;
      const newAngle = lidRotateBaseAngleRef.current + delta;
      if (mode === "upper-lid-rotate") setUpperLidAngle(newAngle);
      else setLowerLidAngle(newAngle);
    }
  }, [clientToSvg]);

  type DragMode = "radius" | "center" | "upper-lid" | "lower-lid" | "upper-lid-rotate" | "lower-lid-rotate";

  const startDrag = useCallback(
    (e: React.PointerEvent, mode: DragMode, pivot?: { x: number; y: number }) => {
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

      if (mode === "upper-lid" || mode === "lower-lid") {
        const pt = clientToSvg(e.clientX, e.clientY);
        lidDragStartRef.current = { x: pt?.x ?? 0, y: pt?.y ?? 0 };
        lidDragStartOffsetRef.current = mode === "upper-lid"
          ? { x: upperLidOffsetX, y: upperLidOffset }
          : { x: lowerLidOffsetX, y: lowerLidOffset };
      }

      if ((mode === "upper-lid-rotate" || mode === "lower-lid-rotate") && pivot) {
        const pt = clientToSvg(e.clientX, e.clientY);
        lidRotatePivotRef.current = pivot;
        lidRotateStartAngleRef.current = pt ? Math.atan2(pt.y - pivot.y, pt.x - pivot.x) : 0;
        lidRotateBaseAngleRef.current = mode === "upper-lid-rotate" ? upperLidAngle : lowerLidAngle;
      }

      const onMove = (ev: PointerEvent) => {
        if (ev.pointerId !== dragPointerIdRef.current) return;
        applyPointerMove(ev.clientX, ev.clientY);
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
    [applyPointerMove, clientToSvg, upperLidOffset, upperLidOffsetX, lowerLidOffset, lowerLidOffsetX, upperLidAngle, lowerLidAngle]
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
    if (!result || !effectiveAssessment) return "No analysis yet.";
    const parts: string[] = [];

    if (effectiveAssessment.isTooSmall) {
      parts.push("Miosis: pupil appears constricted relative to the iris.");
    }
    if (effectiveAssessment.isTooLarge) {
      parts.push("Mydriasis: pupil appears dilated.");
    }

    const shapeWarnings = result.warnings.filter(
      (w) => w.toLowerCase().includes("distortion") || w.toLowerCase().includes("perfect circle") || w.toLowerCase().includes("deviates"),
    );
    parts.push(...shapeWarnings);

    if (effectiveAssessment.isIrregular) {
      parts.push("Corectopia: pupil center appears displaced relative to the iris.");
    }
    if (effectiveAssessment.isOccludedByLid && pupilVisiblePercent != null) {
      parts.push(`Pupil dips under the eyelid (${pupilVisiblePercent.toFixed(1)}% visible).`);
    }

    if (parts.length === 0) {
      parts.push("Pupil appears within expected range based on this heuristic check (non-medical).");
    }

    return parts.join(" ");
  }, [result, effectiveAssessment, pupilVisiblePercent]);

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
      ["pupil_visible_percent", pupilVisiblePercent != null ? pupilVisiblePercent.toFixed(1) : ""],
      ["eyelid_manually_adjusted", (upperLidOffset !== 0 || upperLidOffsetX !== 0 || lowerLidOffset !== 0 || lowerLidOffsetX !== 0 || upperLidAngle !== 0 || lowerLidAngle !== 0) ? "yes" : "no"],
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
    pupilVisiblePercent,
    upperLidOffset,
    upperLidOffsetX,
    lowerLidOffset,
    lowerLidOffsetX,
    upperLidAngle,
    lowerLidAngle,
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
      setUpperLidOffset(0);
      setUpperLidOffsetX(0);
      setLowerLidOffset(0);
      setLowerLidOffsetX(0);
      setUpperLidAngle(0);
      setLowerLidAngle(0);
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
                  Ocular Biomarker Analyzer
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
                                onPointerDown={(e) => startDrag(e, "center")}
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
                                onPointerDown={(e) => startDrag(e, "radius")}
                                aria-label="Drag to resize iris circle"
                              />
                            </>
                          )}
                          {upperLidDisplay.length > 1 && (() => {
                            const mid = Math.floor(upperLidDisplay.length / 2);
                            const pivot = { x: upperLidDisplay[mid].x, y: upperLidDisplay[mid].y };
                            return (
                              <>
                                <polyline
                                  points={upperLidDisplay.map((p) => `${p.x},${p.y}`).join(" ")}
                                  fill="none"
                                  stroke="rgb(244 63 94)"
                                  strokeWidth={1.8}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  opacity={0.85}
                                  pointerEvents="none"
                                />
                                <circle
                                  cx={pivot.x}
                                  cy={pivot.y}
                                  r={7}
                                  fill="rgb(244 63 94)"
                                  stroke="white"
                                  strokeWidth={1.5}
                                  opacity={0.9}
                                  cursor="move"
                                  style={{ touchAction: "none" }}
                                  onPointerDown={(e) => startDrag(e, "upper-lid")}
                                  aria-label="Drag to move upper eyelid line"
                                />
                                <rect
                                  x={upperLidDisplay[0].x - 5}
                                  y={upperLidDisplay[0].y - 5}
                                  width={10}
                                  height={10}
                                  rx={2}
                                  fill="rgb(244 63 94)"
                                  stroke="white"
                                  strokeWidth={1.5}
                                  opacity={0.9}
                                  cursor="grab"
                                  style={{ touchAction: "none" }}
                                  onPointerDown={(e) => startDrag(e, "upper-lid-rotate", pivot)}
                                  aria-label="Drag to rotate upper eyelid line"
                                />
                                <rect
                                  x={upperLidDisplay[upperLidDisplay.length - 1].x - 5}
                                  y={upperLidDisplay[upperLidDisplay.length - 1].y - 5}
                                  width={10}
                                  height={10}
                                  rx={2}
                                  fill="rgb(244 63 94)"
                                  stroke="white"
                                  strokeWidth={1.5}
                                  opacity={0.9}
                                  cursor="grab"
                                  style={{ touchAction: "none" }}
                                  onPointerDown={(e) => startDrag(e, "upper-lid-rotate", pivot)}
                                  aria-label="Drag to rotate upper eyelid line"
                                />
                              </>
                            );
                          })()}
                          {lowerLidDisplay.length > 1 && (() => {
                            const mid = Math.floor(lowerLidDisplay.length / 2);
                            const pivot = { x: lowerLidDisplay[mid].x, y: lowerLidDisplay[mid].y };
                            return (
                              <>
                                <polyline
                                  points={lowerLidDisplay.map((p) => `${p.x},${p.y}`).join(" ")}
                                  fill="none"
                                  stroke="rgb(244 63 94)"
                                  strokeWidth={1.8}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  opacity={0.85}
                                  pointerEvents="none"
                                />
                                <circle
                                  cx={pivot.x}
                                  cy={pivot.y}
                                  r={7}
                                  fill="rgb(244 63 94)"
                                  stroke="white"
                                  strokeWidth={1.5}
                                  opacity={0.9}
                                  cursor="move"
                                  style={{ touchAction: "none" }}
                                  onPointerDown={(e) => startDrag(e, "lower-lid")}
                                  aria-label="Drag to move lower eyelid line"
                                />
                                <rect
                                  x={lowerLidDisplay[0].x - 5}
                                  y={lowerLidDisplay[0].y - 5}
                                  width={10}
                                  height={10}
                                  rx={2}
                                  fill="rgb(244 63 94)"
                                  stroke="white"
                                  strokeWidth={1.5}
                                  opacity={0.9}
                                  cursor="grab"
                                  style={{ touchAction: "none" }}
                                  onPointerDown={(e) => startDrag(e, "lower-lid-rotate", pivot)}
                                  aria-label="Drag to rotate lower eyelid line"
                                />
                                <rect
                                  x={lowerLidDisplay[lowerLidDisplay.length - 1].x - 5}
                                  y={lowerLidDisplay[lowerLidDisplay.length - 1].y - 5}
                                  width={10}
                                  height={10}
                                  rx={2}
                                  fill="rgb(244 63 94)"
                                  stroke="white"
                                  strokeWidth={1.5}
                                  opacity={0.9}
                                  cursor="grab"
                                  style={{ touchAction: "none" }}
                                  onPointerDown={(e) => startDrag(e, "lower-lid-rotate", pivot)}
                                  aria-label="Drag to rotate lower eyelid line"
                                />
                              </>
                            );
                          })()}
                        </svg>
                        <div className="absolute bottom-2 left-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-ink-muted bg-white/95 backdrop-blur-sm rounded-md px-2 py-1 border border-line shadow-sm">
                          <span><span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-1" />Pupil</span>
                          <span><span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />Iris</span>
                          <span className="text-sky-400">— Fitted circle</span>
                          {(upperLidDisplay.length > 0 || lowerLidDisplay.length > 0) && (
                            <span><span className="inline-block w-2 h-2 rounded-full bg-rose-500 mr-1" />Eyelid</span>
                          )}
                          {result?.irisPoints?.length ? (
                            <span className="text-amber-600/90">· Drag handles to adjust iris/lids</span>
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
                      centering.normalized < 0.25
                        ? "border-emerald-300 bg-gradient-to-br from-emerald-50 via-teal-50/80 to-cyan-50/50 ring-2 ring-emerald-200/50"
                        : centering.normalized < 0.40
                          ? "border-slate-200 bg-gradient-to-br from-white to-slate-50/90 ring-1 ring-slate-200/60"
                          : "border-rose-300 bg-gradient-to-br from-rose-50 via-orange-50/70 to-amber-50/40 ring-2 ring-rose-200/50"
                    }`}>
                      <p className="font-medium text-ink">Pupil centering (vs iris)</p>
                      <p className="text-ink-muted">
                        Offset: {centering.offsetPx.toFixed(1)} px ({centering.normalized.toFixed(2)}× iris radius)
                        <span className={`ml-1 font-medium ${
                          centering.normalized < 0.25 ? "text-emerald-800" : centering.normalized < 0.40 ? "text-ink-muted" : "text-rose-700"
                        }`}>
                          — {centering.label}
                        </span>
                      </p>
                      <p className="text-[11px] text-ink-faint mt-0.5">
                        How far the pupil center is from the iris center. Adjust the iris circle to match the eye, then read this.
                      </p>
                    </div>
                  )}

                  {hasResult && pupilVisiblePercent != null && (
                    <div className={`mt-3 rounded-xl border-2 px-3 py-3 text-xs space-y-1 shadow-sm ${
                      pupilVisiblePercent >= 90
                        ? "border-emerald-300 bg-gradient-to-br from-emerald-50 via-teal-50/80 to-cyan-50/50 ring-2 ring-emerald-200/50"
                        : pupilVisiblePercent >= 60
                          ? "border-slate-200 bg-gradient-to-br from-white to-slate-50/90 ring-1 ring-slate-200/60"
                          : "border-rose-300 bg-gradient-to-br from-rose-50 via-orange-50/70 to-amber-50/40 ring-2 ring-rose-200/50"
                    }`}>
                      <p className="font-medium text-ink">Pupil visible</p>
                      <p className="text-ink-muted">
                        {pupilVisiblePercent.toFixed(1)}% of pupil area is between the eyelid lines
                        <span className={`ml-1 font-medium ${
                          pupilVisiblePercent >= 90 ? "text-emerald-800" : pupilVisiblePercent >= 60 ? "text-ink-muted" : "text-rose-700"
                        }`}>
                          — {pupilVisiblePercent >= 90 ? "fully exposed" : pupilVisiblePercent >= 60 ? "partially occluded" : "significantly occluded"}
                        </span>
                      </p>
                      <p className="text-[11px] text-ink-faint mt-0.5">
                        {upperLidOffset !== 0 || upperLidOffsetX !== 0 || lowerLidOffset !== 0 || lowerLidOffsetX !== 0 || upperLidAngle !== 0 || lowerLidAngle !== 0
                          ? "Eyelid lines adjusted manually. Drag circle to move, squares to rotate."
                          : "Auto-detected eyelid edges. Drag circle to move, squares to rotate."}
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
                      <ul className="mt-1 grid grid-cols-4 gap-1.5 text-[11px]">
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
                        <li
                          className={`rounded-full border px-2 py-1 text-center font-medium transition-shadow ${
                            effectiveAssessment.isOccludedByLid
                              ? "border-2 border-rose-400 bg-gradient-to-r from-rose-500 to-pink-500 text-white shadow-md shadow-rose-400/35 ring-2 ring-rose-200 ring-offset-2 ring-offset-white"
                              : "border border-slate-200/90 bg-gradient-to-b from-white to-slate-50 text-slate-400"
                          }`}
                        >
                          Lid dip
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

