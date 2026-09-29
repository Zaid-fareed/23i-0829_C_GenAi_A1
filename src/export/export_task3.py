"""Export the FULL Task 3 soft-MoE pipeline (gate + 3 experts + identity + mixing) as ONE ONNX file.

The exported graph outputs (restored, weights) so the app can show the 4 gate weights without a second
forward pass. Tracing works because SoftMoE.forward loops over a fixed Python list of expert names
(not data-dependent), so the loop unrolls into a static graph.

  python -m src.export.export_task3 --ckpt checkpoints/task3_moe.pt --out onnx_models/task3_soft_moe.onnx
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.export.onnx_utils import export_and_verify
from src.models.autoencoder import build_ae
from src.models.classifier import build_classifier
from src.models.moe import SoftMoE
from src.utils.common import seed_everything


class MoEExportWrapper(nn.Module):
    """Thin wrapper so torch.onnx.export sees a plain (image) -> (restored, weights) signature."""

    def __init__(self, moe: SoftMoE):
        super().__init__()
        self.moe = moe

    def forward(self, image):
        out, weights, _ = self.moe(image, return_weights=True)
        return out.clamp(0, 1), weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/task3_moe.pt")
    ap.add_argument("--out", default="onnx_models/task3_soft_moe.onnx")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    seed_everything(42)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    gate = build_classifier(ck["gate_params"])
    experts = {t: build_ae(ck["expert_params"][t]) for t in ck["expert_params"]}
    moe = SoftMoE(gate, experts, temperature=ck["params"]["temperature"])
    moe.load_state_dict(ck["state_dict"])
    wrapper = MoEExportWrapper(moe)

    dummy = (torch.rand(a.batch, 3, 128, 128),)
    export_and_verify(wrapper, dummy, a.out, input_names=["image"], output_names=["restored", "weights"])


if __name__ == "__main__":
    main()
