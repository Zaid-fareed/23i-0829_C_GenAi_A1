"""Export the Task 4 generator to ONNX (inputs: photo (B,3,128,128) in [-1,1], style int64 (B,)).

  python -m src.export.export_task4 --ckpt checkpoints/task4_generator.pt --out onnx_models/task4_generator.onnx
"""
import argparse
from pathlib import Path

import torch

from src.export.onnx_utils import export_and_verify
from src.models.cgan import UNetGenerator
from src.utils.common import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/task4_generator.pt")
    ap.add_argument("--out", default="onnx_models/task4_generator.onnx")
    a = ap.parse_args()
    seed_everything(42)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    p = ck["params"]
    G = UNetGenerator(p["base"], p["emb_dim"], p["dropout"])
    G.load_state_dict(ck["state_dict"])
    dummy = (torch.rand(4, 3, 128, 128) * 2 - 1, torch.tensor([0, 1, 2, 0]))
    export_and_verify(G, dummy, a.out, input_names=["photo", "style"], output_names=["sketch"], atol=1e-3)


if __name__ == "__main__":
    main()
