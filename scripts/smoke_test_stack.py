"""End-to-end smoke test of the RUNNING application (Docker Compose or dev servers) using only the standard library.

  python scripts/smoke_test_stack.py [--base http://localhost:3000]

Exercises every endpoint through the same URL a browser uses (nginx -> FastAPI -> ONNX Runtime).
"""
import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:3000")
BASE = ap.parse_args().base.rstrip("/")
results = []


def multipart(fields, files=None):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (name, data, ctype) in (files or {}).items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
        body.write(data + b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def call(method, path, fields=None, files=None):
    data, headers = None, {}
    if method == "POST":
        data, ctype = multipart(fields or {}, files)
        headers["Content-Type"] = ctype
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if "json" in r.headers.get("Content-Type", "") else raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def tiny_png(size=(90, 70)):
    # minimal valid PNG via PIL if present, else a hard-coded 1x1 PNG
    try:
        from PIL import Image
        b = io.BytesIO(); Image.new("RGB", size, (120, 80, 200)).save(b, "PNG"); return b.getvalue()
    except ImportError:
        import base64
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


import time
for _ in range(30):  # the backend may still be starting (e.g. right after `docker compose up`)
    st, h = call("GET", "/api/health")
    if st == 200 and isinstance(h, dict):
        break
    time.sleep(2)
if not (st == 200 and isinstance(h, dict)):
    print(f"FAIL backend not reachable at {BASE} (HTTP {st}); is `docker compose up` running?")
    sys.exit(1)
check("health: status ok and all 4 workspaces ready", h["status"] == "ok" and all(h["workspaces"].values()), str(h.get("workspaces")))
check("health: all 7 ONNX models loaded", sum(m["loaded"] for m in h["models"].values()) == 7)
st, page = call("GET", "/")
check("frontend index served", st == 200 and b"root" in page)
st, _ = call("GET", "/api/docs")
check("API docs page", st == 200)
st, s = call("GET", "/api/samples")
samples = s["samples"] if st == 200 else []
pets = [x for x in samples if x.startswith("pet_")]
faces = [x for x in samples if x.startswith("face_")]
check("samples listed (pets + faces)", len(pets) >= 3 and len(faces) >= 2, f"{len(pets)} pets, {len(faces)} faces")

for ep in ("universal", "hard", "soft"):
    for c in ("salt_pepper", "blur", "occlusion"):
        for sev in ("low", "medium", "high"):
            st, j = call("POST", f"/api/restore/{ep}", {"sample": pets[0], "corruption": c, "severity": sev})
            ok = st == 200 and j["output_image"].startswith("data:image/png") and j["corruption"]["type"] == c and j["quality"] is not None
            if ep == "hard" and ok:
                ok = abs(sum(j["probabilities"].values()) - 1) < 1e-3 and j["timing_ms"]["total"] > 0
            if ep == "soft" and ok:
                ok = len(j["weights"]) == 4 and abs(sum(j["weights"].values()) - 1) < 1e-3
            check(f"restore/{ep} {c}/{sev}", ok)

st, j = call("POST", "/api/restore/universal", {"sample": pets[1], "corruption": "salt_pepper", "severity": "custom", "sp_p": 0.1})
check("universal custom severity", st == 200 and j["corruption"]["p"] == 0.1)
st, j = call("POST", "/api/restore/universal", {"sample": pets[1], "corruption": "none"})
check("universal as-uploaded (no corruption, no quality numbers)", st == 200 and j["quality"] is None)
st, j = call("POST", "/api/restore/hard", {"sample": pets[0], "corruption": "blur", "severity": "high", "mode": "oracle"})
check("hard routing oracle mode picks the blur specialist", st == 200 and j["routed_to"] == "blur")
st, j = call("POST", "/api/restore/hard", {"sample": pets[0], "corruption": "none", "mode": "oracle"})
check("hard routing oracle without corruption -> 400", st == 400)
st, j = call("POST", "/api/restore/hard", {"sample": pets[0], "corruption": "none"})
check("hard routing on a clean image uses identity bypass", st == 200 and j["predicted"] == "clean" and j["expert"].startswith("identity"), f"predicted={j.get('predicted')}")

for style in (1, 2, 3):
    for fit in ("crop", "pad", "stretch"):
        st, j = call("POST", "/api/sketch", {"sample": faces[0], "style": style, "fit": fit})
        check(f"sketch style {style} fit={fit}", st == 200 and j["style"] == f"Style {style}" and j["output_image"].startswith("data:image/png"))

png = tiny_png()
st, j = call("POST", "/api/sketch", {"style": 1}, {"file": ("x.png", png, "image/png")})
check("upload: valid PNG accepted", st == 200)
st, j = call("POST", "/api/sketch", {"style": 1}, {"file": ("x.png", b"not an image", "image/png")})
check("upload: corrupt file -> 400", st == 400)
st, j = call("POST", "/api/sketch", {"style": 1}, {"file": ("x.pdf", png, "application/pdf")})
check("upload: wrong type -> 415", st == 415)
st, j = call("POST", "/api/sketch", {"style": 1}, {"file": ("big.png", b"x" * (11 * 1024 * 1024), "image/png")})
check("upload: >10MB -> 413", st == 413)
st, j = call("POST", "/api/sketch", {"style": 4}, {"file": ("x.png", png, "image/png")})
check("invalid style -> 422", st == 422)
st, j = call("POST", "/api/sketch", {"style": 1, "sample": "../../etc/passwd"})
check("path traversal in sample name blocked", st == 404)
st, j = call("POST", "/api/restore/universal", {"corruption": "blur"})
check("no image supplied -> 400", st == 400)
st, j = call("POST", "/api/restore/universal", {"sample": pets[0], "corruption": "blur", "severity": "custom", "blur_kernel": 4})
check("even blur kernel rejected -> 422", st == 422)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
