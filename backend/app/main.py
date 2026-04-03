from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple
from io import BytesIO
import base64

import cv2
import numpy as np
from PIL import Image


class Point2D(BaseModel):
    x: float
    y: float


class FittedCircle(BaseModel):
    centerX: float
    centerY: float
    radius: float
    rmsDeviationPx: float
    maxDeviationPx: float


class PupilAssessment(BaseModel):
    isTooLarge: bool
    isTooSmall: bool
    isIrregular: bool


class ProportionalityAssessment(BaseModel):
    pupilToIrisRatio: Optional[float]
    isProportional: Optional[bool]
    note: str


class AnalysisResponse(BaseModel):
    pupilDiameterPixels: Optional[float]
    circularity: Optional[float]
    contourArea: Optional[float]
    warnings: List[str]
    assessment: PupilAssessment
    processingNotes: Optional[str] = None
    # New: plot points and circle comparison
    pupilPoints: Optional[List[Point2D]] = None
    irisPoints: Optional[List[Point2D]] = None
    fittedCircle: Optional[FittedCircle] = None
    deviationFromPerfectCircle: Optional[str] = None
    proportionality: Optional[ProportionalityAssessment] = None
    imageWidth: Optional[int] = None
    imageHeight: Optional[int] = None
    processedImageBase64: Optional[str] = None


app = FastAPI(title="Eye Metrics Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def load_and_normalize_image(data: bytes) -> Tuple[np.ndarray, int, int]:
    image = Image.open(BytesIO(data))
    image = image.convert("L")

    max_side = 640
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    arr = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    h, w = arr.shape
    return arr, w, h


def sample_contour_points(contour: np.ndarray, n_points: int = 64) -> List[Point2D]:
    """Sample n_points evenly along the contour for plotting."""
    contour = contour.reshape(-1, 2)
    if len(contour) < 3:
        return [Point2D(x=float(p[0]), y=float(p[1])) for p in contour]
    seg_dists = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    cumdist = np.concatenate([[0], np.cumsum(seg_dists)])
    total = cumdist[-1]
    if total == 0:
        return [Point2D(x=float(contour[0][0]), y=float(contour[0][1]))] * n_points
    indices = np.linspace(0, total * (n_points - 1) / n_points, n_points, endpoint=False)
    out: List[Point2D] = []
    for t in indices:
        i = np.searchsorted(cumdist, t, side="right") - 1
        i = min(i, len(contour) - 2)
        frac = (t - cumdist[i]) / (cumdist[i + 1] - cumdist[i]) if cumdist[i + 1] > cumdist[i] else 0
        x = contour[i][0] + frac * (contour[i + 1][0] - contour[i][0])
        y = contour[i][1] + frac * (contour[i + 1][1] - contour[i][1])
        out.append(Point2D(x=float(x), y=float(y)))
    return out


def fit_circle_and_deviation(points: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fit a circle to points; return (cx, cy, radius, rms_deviation, max_deviation)."""
    pts = points.reshape(-1, 2).astype(np.float64)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    radius = float(np.mean(dists))
    deviations = np.abs(dists - radius)
    rms = float(np.sqrt(np.mean(deviations ** 2)))
    max_dev = float(np.max(deviations))
    return cx, cy, radius, rms, max_dev


def detect_iris(gray: np.ndarray, pupil_cx: float, pupil_cy: float, pupil_r: float) -> Optional[np.ndarray]:
    """Detect iris boundary: outer edge of the dark (pupil+iris) region or Hough circle fallback."""
    h, w = gray.shape
    pt = (int(pupil_cx), int(pupil_cy))
    kernel = np.ones((5, 5), np.uint8)

    # Method 1: Iris = outer boundary of Otsu dark region (pupil+iris)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (9, 9), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    iris_candidates = []
    for c in contours:
        if cv2.pointPolygonTest(c, pt, False) < 0:
            continue
        area = cv2.contourArea(c)
        (ix, iy), ir = cv2.minEnclosingCircle(c)
        # Iris must be larger than pupil
        if ir <= pupil_r * 1.1:
            continue
        if area > np.pi * (pupil_r * 1.2) ** 2:
            iris_candidates.append((area, c))
    if iris_candidates:
        iris_candidates.sort(key=lambda x: -x[0])
        return iris_candidates[0][1].astype(np.float32)

    # Method 2: HoughCircles fallback (works for dilated pupils: iris ~1.2-2.5x pupil)
    min_r = max(int(pupil_r * 1.15), 10)
    max_r = min(int(pupil_r * 3.5), min(w, h) // 2 - 2)
    if min_r >= max_r:
        return None
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.1, minDist=int(pupil_r * 1.5),
        param1=40, param2=25, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    best = None
    best_score = -1.0
    for (x, y, r) in circles[0]:
        dist = np.hypot(x - pupil_cx, y - pupil_cy)
        if dist > pupil_r:
            continue
        score = 1.0 / (1.0 + dist) + 0.05 * min(r / (pupil_r + 1e-6), 3.0)
        if score > best_score:
            best_score = score
            best = (x, y, r)
    if best is None:
        return None
    x, y, r = best
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    contour = np.column_stack([x + r * np.cos(theta), y + r * np.sin(theta)])
    return contour.astype(np.float32)


def detect_pupil(gray: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    img_cx, img_cy = w / 2.0, h / 2.0
    min_dim = min(w, h)
    kernel = np.ones((5, 5), np.uint8)

    # Eyelashes: at lid margins (top/bottom 20%). Pupil: central.
    margin = 0.2 * h
    center_ymin, center_ymax = margin, h - margin

    # Pupil: very circular (0.7+), central, medium area. Eyelash clusters: irregular, often at edges.
    min_pupil_area = 450
    max_pupil_area = int(0.28 * w * h)
    max_dist_from_center = 0.35 * min_dim

    def passes(c: np.ndarray) -> tuple[bool, float, float, float]:
        area = cv2.contourArea(c)
        if area < min_pupil_area or area > max_pupil_area:
            return False, 0, 0, 0
        m = cv2.moments(c)
        if m["m00"] == 0:
            return False, 0, 0, 0
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        if cy < center_ymin or cy > center_ymax:
            return False, 0, 0, 0
        dist = np.hypot(cx - img_cx, cy - img_cy)
        if dist > max_dist_from_center:
            return False, 0, 0, 0
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            return False, 0, 0, 0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.65:
            return False, 0, 0, 0
        return True, area, circularity, dist

    def find_best(contours_list: list) -> Optional[np.ndarray]:
        candidates = []
        for c in contours_list:
            ok, area, circ, dist = passes(c)
            if not ok:
                continue
            centrality = np.exp(-(dist ** 2) / (2 * (0.15 * min_dim) ** 2))
            score = circ * 3.0 + centrality * 10.0
            candidates.append((score, area, c))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[1], -x[0]))
        return candidates[0][2]

    # Strict threshold first: only blackest pixels (pupil). Iris and lashes are lighter.
    for thresh_val in [50, 45, 60, 35]:
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = find_best(contours)
        if best is not None:
            return best, ""

    # Otsu fallback
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = find_best(contours)
    if best is not None:
        return best, ""

    return None, "Could not confidently isolate a pupil region."


def characterize_pupil(contour: np.ndarray) -> tuple[float, float, float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = float(radius * 2.0)
    return float(diameter), float(circularity), float(area)


def build_pupil_assessment(
    diameter: float,
    circularity: float,
    *,
    ratio: Optional[float] = None,
    offset_norm: Optional[float] = None,
    rms_dev: Optional[float] = None,
    circle_radius: Optional[float] = None,
) -> tuple[PupilAssessment, List[str]]:
    """
    isTooSmall: only when pupil/iris ratio is below typical (small pupil vs iris).
    isTooLarge: absolute diameter (px) and/or high pupil/iris ratio.
    isIrregular: only decentering vs detected iris (offset_norm).
    """
    typical_min, typical_max = 0.15, 0.55
    centering_threshold = 0.15
    shape_circularity_cutoff = 0.82
    rms_rel_threshold = 0.10

    too_small = ratio is not None and ratio < typical_min
    too_large = diameter > 140
    if ratio is not None and ratio > typical_max:
        too_large = True

    rms_rel = 0.0
    if rms_dev is not None and circle_radius is not None and circle_radius > 1e-6:
        rms_rel = rms_dev / circle_radius

    irregular_shape = circularity < shape_circularity_cutoff or rms_rel > rms_rel_threshold
    irregular_prop = ratio is not None and (ratio < typical_min or ratio > typical_max)
    irregular_center = offset_norm is not None and offset_norm >= centering_threshold

    warnings: List[str] = []
    if too_small:
        warnings.append("Pupil looks small relative to the iris (low pupil/iris ratio).")
    if too_large:
        warnings.append("Pupil appears relatively large in this image.")
    if irregular_shape:
        warnings.append("Pupil outline deviates noticeably from a perfect circle.")
    if irregular_prop:
        warnings.append("Pupil/iris proportion looks atypical for this view.")
    if irregular_center:
        warnings.append("Pupil center appears offset relative to the iris.")

    if not warnings:
        warnings.append("Pupil size and shape look within a typical range for this image.")

    assessment = PupilAssessment(
        isTooLarge=too_large,
        isTooSmall=too_small,
        isIrregular=irregular_center,
    )
    return assessment, warnings


@app.post("/analyze-eye", response_model=AnalysisResponse)
async def analyze_eye(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        gray, img_w, img_h = load_and_normalize_image(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode this image format.")

    contour, note = detect_pupil(gray)
    if contour is None:
        response = AnalysisResponse(
            pupilDiameterPixels=None,
            circularity=None,
            contourArea=None,
            warnings=[note or "No clear pupil region could be detected."],
            assessment=PupilAssessment(
                isTooLarge=False,
                isTooSmall=False,
                isIrregular=False,
            ),
            processingNotes="Try a closer, sharper photo with the eye centered and minimal reflections.",
        )
        return JSONResponse(content=response.model_dump())

    diameter, circularity, area = characterize_pupil(contour)

    pupil_points = sample_contour_points(contour, 64)
    pts_arr = contour.reshape(-1, 2)
    cx, cy, radius, rms_dev, max_dev = fit_circle_and_deviation(pts_arr)
    fitted = FittedCircle(
        centerX=float(cx),
        centerY=float(cy),
        radius=float(radius),
        rmsDeviationPx=round(rms_dev, 2),
        maxDeviationPx=round(max_dev, 2),
    )
    dev_note = (
        f"RMS deviation from fitted circle: {rms_dev:.2f} px; max: {max_dev:.2f} px. "
        f"Lower values indicate the pupil is closer to a perfect circle."
    )

    iris_points: Optional[List[Point2D]] = None
    proportionality: Optional[ProportionalityAssessment] = None

    pupil_r = float(radius)
    ratio: Optional[float] = None
    offset_norm: Optional[float] = None
    iris_contour = detect_iris(gray, cx, cy, pupil_r)

    if iris_contour is not None and len(iris_contour) >= 3:
        ic = iris_contour.reshape(-1, 2)
        (ix, iy), iris_r = cv2.minEnclosingCircle(ic.astype(np.float32))
        iris_r = float(iris_r)
        if iris_r > pupil_r * 1.05:
            ratio = pupil_r / iris_r
            offset_norm = float(np.hypot(cx - ix, cy - iy) / iris_r)
            iris_points = sample_contour_points(iris_contour, 64)
            typical_min, typical_max = 0.15, 0.55
            is_prop = typical_min <= ratio <= typical_max
            proportionality = ProportionalityAssessment(
                pupilToIrisRatio=round(ratio, 3),
                isProportional=is_prop,
                note=(
                    f"Pupil/iris ratio {ratio:.1%} from detected iris boundary. "
                    + ("Within typical range." if is_prop else "Outside typical range for this heuristic.")
                ),
            )

    if iris_points is None:
        # Fallback: synthetic concentric iris when CV does not find a boundary
        iris_r = pupil_r / 0.33
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        iris_points = [
            Point2D(x=float(cx + iris_r * np.cos(t)), y=float(cy + iris_r * np.sin(t)))
            for t in theta
        ]
        ratio = pupil_r / iris_r if iris_r > 0 else None
        if ratio is not None:
            typical_min, typical_max = 0.15, 0.55
            is_prop = typical_min <= ratio <= typical_max
            proportionality = ProportionalityAssessment(
                pupilToIrisRatio=round(ratio, 3),
                isProportional=is_prop,
                note=(
                    f"Pupil/iris ratio {ratio:.1%} (estimated ring; adjust iris in the UI if needed). "
                    + ("Within typical range." if is_prop else "Outside typical range for this heuristic.")
                ),
            )

    assessment, warnings = build_pupil_assessment(
        diameter,
        circularity,
        ratio=ratio,
        offset_norm=offset_norm,
        rms_dev=rms_dev,
        circle_radius=float(radius),
    )

    _, jpeg = cv2.imencode(".jpg", gray)
    processed_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("utf-8")

    response = AnalysisResponse(
        pupilDiameterPixels=diameter,
        circularity=circularity,
        contourArea=area,
        warnings=warnings,
        assessment=assessment,
        processingNotes=(
            "Heuristic pixel-based estimate from a single 2D photo without physical calibration."
        ),
        pupilPoints=pupil_points,
        irisPoints=iris_points,
        fittedCircle=fitted,
        deviationFromPerfectCircle=dev_note,
        proportionality=proportionality,
        imageWidth=img_w,
        imageHeight=img_h,
        processedImageBase64=processed_b64,
    )

    return JSONResponse(content=response.model_dump())

