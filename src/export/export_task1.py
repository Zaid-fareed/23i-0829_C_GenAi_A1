"""Export the Task 1 universal autoencoder to ONNX and verify it against PyTorch.

  python -m src.export.export_task1 --ckpt checkpoints/task1_dae.pt --out onnx_models/task1_universal_dae.onnx
"""
import argparse
from pathlib import Path

import torch

from src.export.onnx_utils import export_and_verify
from src.models.autoencoder import build_ae
from src.utils.common import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/task1_dae.pt")
    ap.add_argument("--out", default="onnx_models/task1_universal_dae.onnx")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    seed_everything(42)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_ae(ck["params"])
    model.load_state_dict(ck["state_dict"])

    dummy = (torch.rand(a.batch, 3, 128, 128),)
    export_and_verify(model, dummy, a.out, input_names=["image"], output_names=["restored"])


if __name__ == "__main__":
    main()
