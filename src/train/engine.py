"""Generic autoencoder train/eval loops (reused for Task 1 and the Task 2 specialists)."""
import copy
import math

import mlflow
import optuna
import torch
from pytorch_msssim import ssim

from src.data.corruptions import TYPES
from src.train import resume as R
from src.utils.common import psnr


@torch.no_grad()
def evaluate_ae(model, loader, device):
    """Alpha-independent objective: obj = 0.5*L1 + 0.5*(1-SSIM) (so trials with different alpha are comparable)."""
    model.eval()
    l1s, ss, ps, labs = [], [], [], []
    for noisy, clean, label, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy).float().clamp(0, 1)
        l1s.append((out - clean).abs().flatten(1).mean(1).cpu())
        ss.append(ssim(out, clean, data_range=1.0, size_average=False).cpu())
        ps.append(psnr(out, clean).cpu())
        labs.append(label)
    l1, s, p, lab = torch.cat(l1s), torch.cat(ss), torch.cat(ps), torch.cat(labs)
    res = {"l1": l1.mean().item(), "ssim": s.mean().item(), "psnr": p.mean().item()}
    res["obj"] = 0.5 * res["l1"] + 0.5 * (1 - res["ssim"])
    for i, t in enumerate(TYPES):
        m = lab == i
        if m.any():
            res[f"ssim_{t}"], res[f"psnr_{t}"] = s[m].mean().item(), p[m].mean().item()
    return res


def fit_ae(model, loss_fn, train_loader, val_loader, *, lr, epochs, device, weight_decay=1e-4,
           trial=None, ckpt_path=None, ckpt_extra=None, log_prefix="", resume=False, resume_path=None):
    """Returns (best_val_obj, history). Logs to the ACTIVE mlflow run if there is one.
    If `trial` is given: reports val obj per epoch and raises optuna.TrialPruned when the pruner says so."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, history = math.inf, None, []
    rpath = R.default_path(ckpt_path, resume_path)
    objs = {"model": model, "opt": opt, "sched": sched, "scaler": scaler}
    start = 0
    if resume and (r := R.load(rpath, objs=objs, device=device)):
        start, best, best_state, history = r
    for epoch in range(start, epochs):
        model.train()
        tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
        n = 0
        for noisy, clean, _, _ in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(noisy)
            loss, l1, s = loss_fn(out, clean)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            b = noisy.size(0)
            n += b
            tot["loss"] += loss.item() * b
            tot["l1"] += l1.item() * b
            tot["ssim"] += s.item() * b
        sched.step()
        val = evaluate_ae(model, val_loader, device)
        rec = {"epoch": epoch, **{f"train_{k}": v / n for k, v in tot.items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(f"{log_prefix}ep {epoch + 1}/{epochs} train_loss {rec['train_loss']:.4f} "
              f"val_obj {val['obj']:.4f} val_ssim {val['ssim']:.4f} val_psnr {val['psnr']:.2f}")
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
