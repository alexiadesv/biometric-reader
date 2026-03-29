# Eye Metrics Analyzer

Next.js (TypeScript) + Tailwind frontend and FastAPI backend for uploading an eye photo, normalizing it, and estimating pupil size and shape.

> **Important:** This is a heuristic, non-medical tool. It works on 2D photos without any real‑world calibration and **must not** be used as a diagnostic device.

---

## Project structure

- **frontend** – Next.js 14 + TypeScript + Tailwind UI
- **backend** – FastAPI service with OpenCV / Pillow image processing
- **docker-compose.yml** – Local dev stack (frontend + backend)

---

## Running locally without Docker

### Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be at `http://localhost:8000` (health check: `GET /health`).

### Frontend (Next.js)

In a separate terminal:

```bash
cd frontend
npm install

# point the frontend at the local backend (default already matches this)
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"  # PowerShell
npm run dev
```

Frontend will be at `http://localhost:3000`.

---

## Running locally with Docker Compose

From the repo root:

```bash
docker compose build
docker compose up
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

The frontend container is configured with `NEXT_PUBLIC_API_BASE_URL=http://backend:8000` so it can reach the backend service by name.

---

## How the analysis works (high level)

1. **Upload** – Frontend sends the selected image as `multipart/form-data` to `POST /analyze-eye`.
2. **Normalization** (backend):
   - Decodes any common image format.
   - Converts to grayscale.
   - Resizes down to max ~640 px on the longest side.
   - Applies CLAHE (contrast-limited adaptive histogram equalization) to stabilize lighting.
3. **Pupil detection**:
   - Gaussian blur + Otsu thresholding (inverted) to highlight dark regions.
   - Morphological open to clean small noise.
   - Finds contours and scores them by:
     - Area (too tiny is ignored),
     - Circularity \(4 \pi A / P^2\),
     - Proximity to image center.
   - Picks the best candidate contour as the pupil.
4. **Measurements**:
   - Uses the best contour to compute:
     - **Diameter (pixels)** – from the minimum enclosing circle.
     - **Circularity** – 1.0 is a perfect circle.
     - **Area (pixels²)** – from contour area.
5. **Assessment** (heuristic thresholds in pixels):
   - **Too small**: diameter \< 30 px.
   - **Too large**: diameter \> 120 px.
   - **Irregular**: circularity \< 0.8.
   - The backend returns:
     - `pupilDiameterPixels`
     - `circularity`
     - `contourArea`
     - `assessment`: `isTooLarge`, `isTooSmall`, `isIrregular`
     - `warnings`: human-readable text

These thresholds are arbitrary and purely for demonstration; tune them as needed.

---

## API

### `GET /health`

Simple health check.

**Response**

```json
{ "status": "ok" }
```

### `POST /analyze-eye`

Accepts an image and returns heuristic pupil metrics.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` – image file (JPG, PNG, etc.).

**Sample cURL**

```bash
curl -X POST "http://localhost:8000/analyze-eye" ^
  -F "file=@example-eye.jpg"
```

**Response (example)**

```json
{
  "pupilDiameterPixels": 84.3,
  "circularity": 0.92,
  "contourArea": 5560.0,
  "warnings": [
    "Within this image, the pupil looks roughly circular and not extremely large or small."
  ],
  "assessment": {
    "isTooLarge": false,
    "isTooSmall": false,
    "isIrregular": false
  },
  "processingNotes": "Heuristic pixel-based estimate from a single 2D photo without physical calibration."
}
```

If a clear pupil region cannot be found, metrics come back as `null` with guidance in `warnings` and `processingNotes`.

---

## Frontend UI notes

- Built with **Next.js app router** and **Tailwind CSS**.
- Key file: `app/page.tsx`
  - Clean card-style layout with dark theme.
  - Image file picker with preview.
  - Calls backend using Axios and displays:
    - Pupil diameter (px),
    - Circularity,
    - Area (px²),
    - Textual summary and chips for
      **Large / Small / Irregular**.

You can tweak the Tailwind theme in `tailwind.config.ts` and global styles in `app/globals.css`.

---

## Tests

### Backend

```bash
cd backend
pytest
```

What’s covered:

- `tests/test_analyze_eye.py`:
  - `GET /health` returns `{"status": "ok"}`.
  - A synthetic grayscale eye image with a round dark pupil returns non-null metrics.

### Frontend

```bash
cd frontend
npm test
```

Currently includes a **placeholder Jest test** (`__tests__/page.test.tsx`) to verify the configuration is wired correctly. You can extend it with React Testing Library tests for the UI and Axios interaction.

---

## Next steps / ideas

- Add overlay visualization (bounding circle) returned from backend as an additional endpoint (e.g., annotated image).
- Support calibrated measurements (millimeters) using a known-size reference in the frame.
- Add multi-eye comparison and history tracking for a user.
- Harden error handling and logging around image decoding and contour detection.

