"""Task 3: soft mixture-of-experts restoration.

4 branches, in the SAME order as src.data.corruptions.TYPES (0=clean,1=salt_pepper,2=blur,3=occlusion):
  branch 0 : identity (no parameters)
  branch 1..3 : the Task 2 specialist autoencoders (salt_pepper, blur, occlusion)

gate(x) -> logits (B,4); mixing weights = softmax(logits / temperature); output = sum_i w_i * branch_i(x).
The gate is architecturally identical to the Task 2 classifier (so Task 2's trained weights can be loaded
directly as initialization); CE loss during MoE training therefore uses the RAW (non-temperature-scaled)
logits, keeping "does the gate know the true class" separate from "how sharply do we mix" (temperature).
"""
import torch
import torch.nn as nn

from src.data.corruptions import TYPE2ID, TYPES
from src.models.autoencoder import build_ae
from src.models.classifier import build_classifier

CORRUPT_TYPES = [t for t in TYPES if t != "clean"]  # branch order 1,2,3 == TYPES[1:]


class SoftMoE(nn.Module):
    def __init__(self, gate: nn.Module, experts: dict, temperature: float = 1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleDict(experts)  # keys: 'salt_pepper','blur','occlusion'
        self.temperature = temperature

    def forward(self, x, return_weights: bool = False):
        logits = self.gate(x)                                   # (B,4), order == TYPES
        weights = torch.softmax(logits / self.temperature, dim=1)
        branch_outputs = [x] + [self.experts[t](x) for t in CORRUPT_TYPES]   # 4 x (B,3,H,W), order == TYPES
        stacked = torch.stack(branch_outputs, dim=1)             # (B,4,3,H,W)
        out = (weights.view(*weights.shape, 1, 1, 1) * stacked).sum(dim=1)
        if return_weights:
            return out, weights, logits
        return out


def set_experts_trainable(model: SoftMoE, trainable: bool):
    """Freeze/unfreeze the 3 specialist branches for the warm-up / joint fine-tune stages.
    Frozen experts are also put in eval() mode so their BatchNorm stats and Dropout don't drift
    while only the gate is being trained."""
    for m in model.experts.values():
        for p in m.parameters():
            p.requires_grad_(trainable)
        m.train(trainable)
    return model


def build_moe_from_checkpoints(classifier_ckpt, specialist_ckpts: dict, device, temperature: float = 1.0):
    """specialist_ckpts: {'salt_pepper': path, 'blur': path, 'occlusion': path}
    Returns (model, gate_params, expert_params) -- the latter two are the ORIGINAL Task 2 hyperparameters,
    kept for provenance/logging (Task 3 does not re-search architecture, only reuses Task 2's)."""
    gate_ck = torch.load(classifier_ckpt, map_location=device)
    gate = build_classifier(gate_ck["params"])
    gate.load_state_dict(gate_ck["state_dict"])

    experts, expert_params = {}, {}
    for ctype, path in specialist_ckpts.items():
        ck = torch.load(path, map_location=device)
        m = build_ae(ck["params"])
        m.load_state_dict(ck["state_dict"])
        experts[ctype] = m
        expert_params[ctype] = ck["params"]

    model = SoftMoE(gate, experts, temperature).to(device)
    return model, gate_ck["params"], expert_params
