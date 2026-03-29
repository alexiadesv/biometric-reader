import io

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


client = TestClient(app)


def make_synthetic_eye() -> bytes:
    img = Image.new("L", (256, 256), color=220)
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 80, 216, 216), fill=100)
    draw.ellipse((96, 120, 160, 184), fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_eye_on_synthetic_image():
    data = make_synthetic_eye()
    files = {"file": ("synthetic.png", data, "image/png")}
    response = client.post("/analyze-eye", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["pupilDiameterPixels"] is not None
    assert body["circularity"] is not None
    assert body["contourArea"] is not None
    assert "pupilPoints" in body and len(body["pupilPoints"]) > 0
    assert "fittedCircle" in body and body["fittedCircle"]["rmsDeviationPx"] >= 0

