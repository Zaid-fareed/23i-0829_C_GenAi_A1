"""Export the Task 2 classifier and all 3 specialists to ONNX, each verified vs PyTorch.

  python -m src.export.export_task2 --out_dir onnx_models
"""
import argparse
from pathlib import Path

import torch

from src.export.onnx_utils import export_and_verify
from src.models.autoencoder import build_ae
from src.models.classifier import build_classifier
from src.utils import paths
from src.utils.common import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=str(paths.checkpoint_dir() / "task2_classifier.pt"))
    ap.add_argument("--salt_pepper", default=str(paths.checkpoint_dir() / "task2_specialist_salt_pepper.pt"))
    ap.add_argument("--blur", default=str(paths.checkpoint_dir() / "task2_specialist_blur.pt"))
    ap.add_argument("--occlusion", default=str(paths.checkpoint_dir() / "task2_specialist_occlusion.pt"))
    ap.add_argument("--out_dir", default="onnx_models")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    seed_everything(42)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dummy = (torch.rand(a.batch, 3, 128, 128),)

    ck = torch.load(a.classifier, map_location="cpu")
    clf = build_classifier(ck["params"]); clf.load_state_dict(ck["state_dict"])
    export_and_verify(clf, dummy, out_dir / "task2_classifier.onnx", input_names=["image"], output_names=["logits"])

    for name, path in (("salt_pepper", a.salt_pepper), ("blur", a.blur), ("occlusion", a.occlusion)):
        ck = torch.load(path, map_location="cpu")
        m = build_ae(ck["params"]); m.load_state_dict(ck["state_dict"])
        export_and_verify(m, dummy, out_dir / f"task2_specialist_{name}.onnx",
                          input_names=["image"], output_names=["restored"])


if __name__ == "__main__":
    main()
