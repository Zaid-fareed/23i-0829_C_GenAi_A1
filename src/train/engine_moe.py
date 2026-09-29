"""Task 3 training engine.

Loss = alpha*L1 + (1-alpha)*(1-SSIM) + w_cls*CE(gate_logits, true_label) + w_bal*balance_loss
  - alpha ties w1(L1)+w2(SSIM)=1, exactly like Task 1's ReconLoss: this avoids the degenerate optimum of
    scaling both reconstruction weights to 0, while Optuna still tunes the L1/SSIM trade-off (`alpha`) and the
    relative importance of classification (`w_cls`) and load-balancing (`w_bal`) independently.
  - balance_loss = sum_i (mean_batch(weight_i) - 1/4)^2   (mean gate weight per branch should be ~1/4)
"""
import copy
import math

import mlflow
import optuna
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from src.data.corruptions import TYPES
from src.models.moe import SoftMoE
from src.train import resume as R
from src.utils.common import psnr

N_BRANCHES = len(TYPES)


def moe_loss(out, clean, logits, weights, labels, alpha, w_cls, w_bal):
    out, clean = out.float(), clean.float()
    l1 = F.l1_loss(out, clean)
    s = ssim(out, clean, data_range=1.0, size_average=True)
    recon = alpha * l1 + (1 - alpha) * (1 - s)
    ce = F.cross_entropy(logits, labels)
    balance = ((weights.mean(0) - 1.0 / N_BRANCHES) ** 2).sum()
    loss = recon + w_cls * ce + w_bal * balance
    return loss, {"recon": recon.detach(), "l1": l1.detach(), "ssim": s.detach(), "ce": ce.detach(),
                 "balance": balance.detach()}


@torch.no_grad()
def evaluate_moe(model: SoftMoE, loader, device):
    model.eval()
    l1s, ss, ps, labs, ws = [], [], [], [], []
    for noisy, clean, label, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out, weights, logits = model(noisy, return_weights=True)
        out = out.float().clamp(0, 1)
        l1s.append((out - clean).abs().flatten(1).mean(1).cpu())
        ss.append(ssim(out, clean, data_range=1.0, size_average=False).cpu())
        ps.append(psnr(out, clean).cpu())
        labs.append(label)
        ws.append(weights.cpu())
    l1, s, p, lab, w = torch.cat(l1s), torch.cat(ss), torch.cat(ps), torch.cat(labs), torch.cat(ws)
    res = {"l1": l1.mean().item(), "ssim": s.mean().item(), "psnr": p.mean().item()}
    res["obj"] = 0.5 * res["l1"] + 0.5 * (1 - res["ssim"])  # same scale as Task 1, for comparability
    res["gate_argmax_accuracy"] = (w.argmax(1) == lab).float().mean().item()
    res["mean_weights_overall"] = w.mean(0).tolist()
    for i, t in enumerate(TYPES):
        m = lab == i
        if m.any():
            res[f"ssim_{t}"], res[f"psnr_{t}"] = s[m].mean().item(), p[m].mean().item()
            res[f"mean_weights_{t}"] = w[m].mean(0).tolist()
    return res


def fit_moe(model: SoftMoE, train_loader, val_loader, *, lr, epochs, device, alpha, w_cls, w_bal,
           weight_decay=1e-4, trial=None, ckpt_path=None, ckpt_extra=None, log_prefix="", resume=False, resume_path=None):
    """Trains whatever parameters currently have requires_grad=True (caller uses set_experts_trainable
    beforehand to select warm-up (gate only) vs joint fine-tune (everything))."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state, history = math.inf, None, []
    rpath = R.default_path(ckpt_path, resume_path)
    objs = {"model": model, "opt": opt, "sched": sched}
    start = 0
    if resume and (r := R.load(rpath, objs=objs, device=device)):
        start, best, best_state, history = r
    for epoch in range(start, epochs):
        model.train()
        # keep frozen experts (if any) in eval mode even though model.train() was just called
        for m in model.experts.values():
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()
        tot = {"loss": 0.0, "recon": 0.0, "l1": 0.0, "ssim": 0.0, "ce": 0.0, "balance": 0.0}
        n = 0
        for noisy, clean, label, _ in train_loader:
            noisy, clean, label = noisy.to(device), clean.to(device), label.to(device)
            out, weights, logits = model(noisy, return_weights=True)
            loss, parts = moe_loss(out, clean, logits, weights, label, alpha, w_cls, w_bal)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            b = noisy.size(0)
            n += b
            tot["loss"] += loss.item() * b
            for k, v in parts.items():
                tot[k] += v.item() * b
        sched.step()
        val = evaluate_moe(model, val_loader, device)
        rec = {"epoch": epoch, **{f"train_{k}": v / n for k, v in tot.items()},
              "val_obj": val["obj"], "val_ssim": val["ssim"], "val_psnr": val["psnr"],
              "val_gate_acc": val["gate_argmax_accuracy"]}
        history.append({**rec, "val_full": val})
        print(f"{log_prefix}ep {epoch + 1}/{epochs} train_loss {rec['train_loss']:.4f} val_obj {val['obj']:.4f} "
              f"val_ssim {val['ssim']:.4f} gate_acc {val['gate_argmax_accuracy']:.3f}")
        if mlflow.active_run():
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
        if val["obj"] < best:
            best, best_state = val["obj"], copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save({"state_dict": best_state, "val": val, "epoch": epoch, **(ckpt_extra or {})}, ckpt_path)
        R.save(rpath, epoch=epoch, objs=objs, best=best, best_state=best_state, history=history)
        if trial is not None:
            trial.report(val["obj"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, history
