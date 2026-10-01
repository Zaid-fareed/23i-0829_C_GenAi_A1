"""Loads the exported ONNX models once at start-up and runs timed inference."""
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_FILES = {
    "universal": "task1_universal_dae.onnx",
    "classifier": "task2_classifier.onnx",
    "specialist_salt_pepper": "task2_specialist_salt_pepper.onnx",
    "specialist_blur": "task2_specialist_blur.onnx",
    "specialist_occlusion": "task2_specialist_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "sketch": "task4_generator.onnx",
}
WORKSPACE_NEEDS = {
    "universal": ["universal"],
    "hard": ["classifier", "specialist_salt_pepper", "specialist_blur", "specialist_occlusion"],
    "soft": ["soft_moe"],
    "sketch": ["sketch"],
}


class ModelRegistry:
    def __init__(self, model_dir: str | Path | None = None):
        self.dir = Path(model_dir or os.environ.get("MODEL_DIR", "/models"))
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.errors: dict[str, str] = {}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        for key, fname in MODEL_FILES.items():
            p = self.dir / fname
            if not p.exists():
                self.errors[key] = f"missing file {fname}"
                continue
            try:
                self.sessions[key] = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
            except Exception as e:  # corrupt/incompatible model -> report in /health instead of crashing
                self.errors[key] = f"failed to load: {e}"

    def status(self) -> dict:
        return {
            "models": {k: {"loaded": k in self.sessions, "file": MODEL_FILES[k], "error": self.errors.get(k)}
                       for k in MODEL_FILES},
            "workspaces": {w: all(k in self.sessions for k in need) for w, need in WORKSPACE_NEEDS.items()},
            "providers": (next(iter(self.sessions.values())).get_providers() if self.sessions else ort.get_available_providers()),
            "onnxruntime": ort.__version__,
        }

    def require(self, workspace: str):
        from fastapi import HTTPException
        missing = [MODEL_FILES[k] for k in WORKSPACE_NEEDS[workspace] if k not in self.sessions]
        if missing:
            raise HTTPException(503, f"Model file(s) not available: {', '.join(missing)}. "
                                     f"Place the ONNX files in the models folder and restart.")

    def run(self, key: str, feeds: dict) -> tuple[list[np.ndarray], float]:
        """Returns (outputs, milliseconds)."""
        sess = self.sessions[key]
        t0 = time.perf_counter()
        out = sess.run(None, feeds)
        return out, (time.perf_counter() - t0) * 1000.0
