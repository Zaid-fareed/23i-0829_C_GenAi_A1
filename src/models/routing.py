"""Hard-routed restoration: classify -> route to the matching specialist, or bypass if predicted clean.

This module is imported by both the Task 2 evaluation script and the FastAPI backend's
/hard-routing endpoint, so the routing logic is defined exactly once.
"""
import time

import torch

from src.data.corruptions import TYPE2ID, TYPES
from src.models.autoencoder import build_ae
from src.models.classifier import build_classifier

CORRUPT_TYPES = [t for t in TYPES if t != "clean"]


def load_classifier(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = build_classifier(ck["params"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck


def load_specialists(ckpt_paths: dict, device):
    """ckpt_paths: {'salt_pepper': path, 'blur': path, 'occlusion': path}"""
    models = {}
    for ctype, p in ckpt_paths.items():
        ck = torch.load(p, map_location=device)
        m = build_ae(ck["params"]).to(device).eval()
        m.load_state_dict(ck["state_dict"])
        models[ctype] = m
    return models


@torch.no_grad()
def restore_batch(noisy: torch.Tensor, labels: torch.Tensor, specialists: dict, device) -> torch.Tensor:
    """labels: LongTensor (B,) of class ids (0=clean bypass, 1=salt, 2=blur, 3=occlusion).
    Rows predicted clean are copied through unchanged (identity bypass)."""
    out = noisy.clone()
    for ctype in CORRUPT_TYPES:
        mask = labels == TYPE2ID[ctype]
        if mask.any():
            out[mask] = specialists[ctype](noisy[mask].to(device)).cpu()
    return out


@torch.no_grad()
def hard_route_single(image: torch.Tensor, classifier, specialists: dict, device) -> dict:
    """Single-image inference for the app/backend. image: (3,H,W) float in [0,1].
    Returns probs, predicted class name, chosen expert (or 'identity'), output image, timings (ms)."""
    x = image.unsqueeze(0).to(device)
    t0 = time.perf_counter()
    probs = torch.softmax(classifier(x), dim=1)[0].cpu()
    t_cls = (time.perf_counter() - t0) * 1000
    pred = int(probs.argmax())
    pred_type = TYPES[pred]
    t0 = time.perf_counter()
    if pred_type == "clean":
        out = image.clone()
        expert = "identity"
    else:
        out = specialists[pred_type](x)[0].clamp(0, 1).cpu()
        expert = pred_type
    t_expert = (time.perf_counter() - t0) * 1000
    return {"probs": {TYPES[i]: float(probs[i]) for i in range(len(TYPES))}, "predicted_class": pred_type,
           "chosen_expert": expert, "output": out, "classifier_ms": t_cls, "expert_ms": t_expert,
           "total_ms": t_cls + t_expert}
