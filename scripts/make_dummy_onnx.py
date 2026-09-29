"""Create RANDOMLY-INITIALISED small ONNX models with the real input/output names, for developing/testing the
backend + frontend before Kaggle training is finished. NOT for submission (outputs are meaningless).

  python scripts/make_dummy_onnx.py --out onnx_models_dummy
"""
import argparse
from pathlib import Path

import torch

from src.export.export_task3 import MoEExportWrapper
from src.export.onnx_utils import export_and_verify
from src.models.autoencoder import build_ae
from src.models.cgan import UNetGenerator
from src.models.classifier import build_classifier
from src.models.moe import SoftMoE

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="onnx_models_dummy")
out = Path(ap.parse_args().out)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
x = (torch.rand(2, 3, 128, 128),)
ae_p = {"channels": "c16", "bottleneck_dim": 16, "dropout": 0.0}
clf_p = {"channels": "c16", "dropout": 0.0}

export_and_verify(build_ae(ae_p), x, out / "task1_universal_dae.onnx", ["image"], ["restored"])
export_and_verify(build_classifier(clf_p), x, out / "task2_classifier.onnx", ["image"], ["logits"])
experts = {}
for t in ("salt_pepper", "blur", "occlusion"):
    experts[t] = build_ae(ae_p)
    export_and_verify(experts[t], x, out / f"task2_specialist_{t}.onnx", ["image"], ["restored"])
moe = SoftMoE(build_classifier(clf_p), experts, temperature=1.0)
export_and_verify(MoEExportWrapper(moe), x, out / "task3_soft_moe.onnx", ["image"], ["restored", "weights"])
G = UNetGenerator(base=16, emb_dim=8, dropout=0.0)
export_and_verify(G, (torch.rand(2, 3, 128, 128) * 2 - 1, torch.tensor([0, 2])), out / "task4_generator.onnx",
                  ["photo", "style"], ["sketch"], atol=1e-3)
print("dummy models written to", out)
