"""API tests against tiny random ONNX models (make with: python scripts/make_dummy_onnx.py)."""
import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "onnx_models_dummy"
if not (MODELS / "task4_generator.onnx").exists():
    pytest.skip("run scripts/make_dummy_onnx.py first", allow_module_level=True)
os.environ["MODEL_DIR"] = str(MODELS)
os.environ["SAMPLES_DIR"] = str(ROOT / "backend" / "samples")
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def png(size=(90, 70)):
    b = io.BytesIO()
    Image.new("RGB", size, (120, 80, 200)).save(b, "PNG")
    return b.getvalue()


def files(data=None, ctype="image/png"):
    return {"file": ("x.png", data or png(), ctype)}


def test_health_all_loaded():
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and all(j["workspaces"].values())


@pytest.mark.parametrize("sev", ["low", "medium", "high"])
@pytest.mark.parametrize("c", ["salt_pepper", "blur", "occlusion"])
def test_universal_with_corruption(c, sev):
    r = client.post("/api/restore/universal", files=files(), data={"corruption": c, "severity": sev})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["corruption"]["applied"] and j["corruption"]["type"] == c and j["inference_ms"] > 0
    assert j["output_image"].startswith("data:image/png;base64,") and j["quality"]["psnr_restored_db"] > 0


def test_universal_uploaded_already_corrupted_no_quality():
    j = client.post("/api/restore/universal", files=files()).json()
    assert j["corruption"] == {"applied": False} and j["quality"] is None


def test_custom_settings_and_validation():
    d = {"corruption": "salt_pepper", "severity": "custom", "sp_p": 0.1}
    assert client.post("/api/restore/universal", files=files(), data=d).json()["corruption"]["p"] == 0.1
    assert client.post("/api/restore/universal", files=files(), data={**d, "sp_p": 0.9}).status_code == 422
    bad = {"corruption": "blur", "severity": "custom", "blur_kernel": 4}
    assert client.post("/api/restore/universal", files=files(), data=bad).status_code == 422


def test_hard_routing_fields_and_oracle():
    r = client.post("/api/restore/hard", files=files(), data={"corruption": "blur", "severity": "high"}).json()
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3 and r["predicted"] in r["probabilities"]
    assert set(r["timing_ms"]) == {"classifier", "expert", "total"}
    o = client.post("/api/restore/hard", files=files(), data={"corruption": "blur", "mode": "oracle"}).json()
    assert o["routed_to"] == "blur" and o["expert"] == "blur specialist"
    assert client.post("/api/restore/hard", files=files(), data={"mode": "oracle"}).status_code == 400


def test_soft_weights_sum_to_one():
    j = client.post("/api/restore/soft", files=files(), data={"corruption": "occlusion"}).json()
    assert len(j["weights"]) == 4 and abs(sum(j["weights"].values()) - 1) < 1e-3


@pytest.mark.parametrize("style", [1, 2, 3])
def test_sketch(style):
    j = client.post("/api/sketch", files=files(), data={"style": style}).json()
    assert j["style"] == f"Style {style}" and j["output_image"].startswith("data:image/png")


@pytest.mark.parametrize("fit", ["crop", "pad", "stretch"])
def test_sketch_fit_modes_on_non_square_photo(fit):
    r = client.post("/api/sketch", files=files(png((60, 120))), data={"style": 2, "fit": fit})
    assert r.status_code == 200 and r.json()["fit"] == fit
    assert client.post("/api/sketch", files=files(), data={"style": 1, "fit": "zoom"}).status_code == 422


def test_bad_inputs():
    assert client.post("/api/sketch", files=files(), data={"style": 4}).status_code == 422
    assert client.post("/api/sketch", files=files(b"not an image", "image/png")).status_code == 400
    assert client.post("/api/sketch", files=files(png(), "application/pdf")).status_code == 415
    assert client.post("/api/sketch", files=files(b"x" * (11 * 1024 * 1024))).status_code == 413
    assert client.post("/api/sketch", data={"style": 1}).status_code == 400
    assert client.post("/api/sketch", data={"style": 1, "sample": "../../etc/passwd"}).status_code == 404


def test_sample_endpoint():
    names = client.get("/api/samples").json()["samples"]
    assert names
    assert client.post("/api/restore/soft", data={"sample": names[0], "corruption": "blur"}).status_code == 200
