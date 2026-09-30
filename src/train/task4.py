"""Task 4: style-conditioned face-to-sketch cGAN (U-Net G + PatchGAN D). Run from the repo root.

  python -m src.train.task4 optuna --n_trials 8 --epochs 15 --max_train 400
  python -m src.train.task4 final  --epochs 100                       # uses configs/task4_best.json

Logged separately to MLflow every epoch: D real loss, D fake loss, G adversarial loss, G L1 loss, and validation
L1 / SSIM / PSNR. The SAME 8 validation photos are rendered every `--sample_every` epochs (all under one style
per row) and logged as an image so generator progress is visible over time.
Optuna objective = val 0.5*L1 + 0.5*(1-SSIM) (sketch range [0,1]); the official test set is never touched here.
"""
import argparse
import copy
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import optuna
import torch
import torch.nn as nn
from pytorch_msssim import ssim

from src.data.fs2k import build_loader
from src.models.cgan import build_gan
from src.train import resume as R
from src.utils import paths
from src.utils.common import get_device, psnr, remaining_trials, seed_everything, setup_mlflow

EXPERIMENT = "task4_face2sketch_cgan"


def suggest(trial):
    return {
        "lr_g": trial.suggest_float("lr_g", 5e-5, 1e-3, log=True),
        "lr_d": trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "base": trial.suggest_categorical("base", [32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "emb_dim": trial.suggest_categorical("emb_dim", [8, 16, 32]),
        "lambda_l1": trial.suggest_float("lambda_l1", 10, 200, log=True),
    }


@torch.no_grad()
def evaluate(G, loader, device):
    G.eval()
    l1s, ss, ps = [], [], []
    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)
        out = G(x, s)
        o01, y01 = (out + 1) / 2, (y + 1) / 2
        l1s.append((o01 - y01).abs().flatten(1).mean(1).cpu())
        ss.append(ssim(o01, y01, data_range=1.0, size_average=False).cpu())
        ps.append(psnr(o01, y01).cpu())
    l1, s, p = torch.cat(l1s).mean().item(), torch.cat(ss).mean().item(), torch.cat(ps).mean().item()
    return {"l1": l1, "ssim": s, "psnr": p, "obj": 0.5 * l1 + 0.5 * (1 - s)}


@torch.no_grad()
def sample_grid(G, fixed_x, device, path):
    """Rows: photo, then generated sketch for Style 1/2/3 - same fixed validation photos every time."""
    G.eval()
    x = fixed_x.to(device)
    rows = [((x + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()]
    for s in range(3):
        o = G(x, torch.full((x.size(0),), s, dtype=torch.long, device=device))
        rows.append(((o + 1) / 2).squeeze(1).cpu().numpy())
    fig, ax = plt.subplots(4, x.size(0), figsize=(1.6 * x.size(0), 6.4))
    for r, row in enumerate(rows):
        for c in range(x.size(0)):
            ax[r, c].imshow(row[c], cmap=None if r == 0 else "gray", vmin=0, vmax=1)
            ax[r, c].axis("off")
        ax[r, 0].set_title(["photo", "Style 1", "Style 2", "Style 3"][r], fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fit_gan(p, epochs, device, max_train=None, trial=None, ckpt_path=None, workers=None, sample_every=0,
            log_prefix="", resume=False, photometric=False, lr_decay=False, tag=""):
    """Returns (best_val_obj, history). Best checkpoint = lowest val obj (generator + params saved)."""
    train_loader = build_loader("train", p["batch_size"], max_items=max_train, workers=workers,
                                photometric=photometric)
    val_loader = build_loader("val", 32, workers=workers)
    fixed_x = next(iter(build_loader("val", 8, shuffle=False, workers=0)))[0]
    G, D = build_gan(p)
    G, D = G.to(device), D.to(device)
    oG = torch.optim.Adam(G.parameters(), lr=p["lr_g"], betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=p["lr_d"], betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    best, best_state, history = math.inf, None, []
    rpath = R.default_path(ckpt_path)
    objs = {"G": G, "D": D, "oG": oG, "oD": oD}
    if lr_decay:  # pix2pix schedule: constant lr for the first half, then linear decay to 0
        half = epochs // 2
        f = lambda e: 1.0 if e < half else max(0.0, (epochs - e) / max(1, epochs - half))  # noqa: E731
        sG, sD = (torch.optim.lr_scheduler.LambdaLR(o, f) for o in (oG, oD))
        objs.update(sG=sG, sD=sD)
    start = 0
    if resume and (r := R.load(rpath, objs=objs, device=device)):
        start, best, best_state, history = r
    for epoch in range(start, epochs):
        G.train(); D.train()
        acc, n = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}, 0
        for x, y, s in train_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            fake = G(x, s)
            # --- discriminator: real pair -> 1, generated pair -> 0
            pr, pf = D(x, y, s), D(x, fake.detach(), s)
            d_real, d_fake = bce(pr, torch.ones_like(pr)), bce(pf, torch.zeros_like(pf))
            oD.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            oD.step()
            # --- generator: fool D + stay close to the paired ground-truth sketch
            pf = D(x, fake, s)
            g_adv, g_l1 = bce(pf, torch.ones_like(pf)), l1(fake, y)
            oG.zero_grad(set_to_none=True)
            (g_adv + p["lambda_l1"] * g_l1).backward()
            oG.step()
            b = x.size(0)
            n += b
            for k, v in (("d_real", d_real), ("d_fake", d_fake), ("g_adv", g_adv), ("g_l1", g_l1)):
                acc[k] += v.item() * b
        if lr_decay:
            sG.step(); sD.step()
        val = evaluate(G, val_loader, device)
        rec = {"epoch": epoch, **{k: v / n for k, v in acc.items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(f"{log_prefix}ep {epoch + 1}/{epochs} D_real {rec['d_real']:.3f} D_fake {rec['d_fake']:.3f} "
              f"G_adv {rec['g_adv']:.3f} G_L1 {rec['g_l1']:.3f} | val_l1 {val['l1']:.4f} "
              f"val_ssim {val['ssim']:.4f} val_psnr {val['psnr']:.2f}")
        if mlflow.active_run():
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
            if sample_every and ((epoch + 1) % sample_every == 0 or epoch == 0):
                out = paths.REPO_ROOT / "reports" / f"task4{tag}"
                out.mkdir(parents=True, exist_ok=True)
                fp = out / f"samples_ep{epoch + 1:03d}.png"
                sample_grid(G, fixed_x, device, fp)
                mlflow.log_artifact(str(fp), artifact_path="samples")
        if val["obj"] < best:
            best, best_state = val["obj"], copy.deepcopy(G.state_dict())
            if ckpt_path:
                torch.save({"state_dict": best_state, "params": p, "val": val, "epoch": epoch, "task": "task4"},
                           ckpt_path)
        R.save(rpath, epoch=epoch, objs=objs, best=best, best_state=best_state, history=history)
        if trial is not None:
            trial.report(val["obj"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, history


def sfx(args):
    return f"_{args.tag}" if args.tag else ""


def objective(trial, args, device):
    p = suggest(trial)
    mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)
    status = "FINISHED"
    try:
        mlflow.log_params(p)
        best, _ = fit_gan(p, args.epochs, device, args.max_train, trial=trial, workers=args.workers,
                          log_prefix=f"[t{trial.number}] ", photometric=args.photometric,
                          lr_decay=args.lr_decay)
        mlflow.log_metric("best_val_obj", best)
        return best
    except optuna.TrialPruned:
        mlflow.set_tag("pruned", "true")
        status = "KILLED"
        raise
    except Exception:
        status = "FAILED"
        raise
    finally:
        mlflow.end_run(status)


def cmd_optuna(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    (paths.REPO_ROOT / "optuna_studies").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=f"task4_cgan{sfx(args)}",
        storage=f"sqlite:///{paths.REPO_ROOT / 'optuna_studies' / f'task4{sfx(args)}.db'}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3), load_if_exists=True)
    with mlflow.start_run(run_name=f"optuna_study_task4{sfx(args)}"):
        mlflow.log_params({"n_trials": args.n_trials, "trial_epochs": args.epochs, "max_train": args.max_train,
                           "photometric": args.photometric, "lr_decay": args.lr_decay})
        study.optimize(lambda t: objective(t, args, device), n_trials=remaining_trials(study, args.n_trials))
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    (paths.REPO_ROOT / "configs").mkdir(exist_ok=True)
    json.dump(study.best_params, open(paths.REPO_ROOT / "configs" / f"task4_best{sfx(args)}.json", "w"), indent=2)
    study.trials_dataframe().to_csv(paths.REPO_ROOT / "optuna_studies" / f"task4{sfx(args)}_trials.csv", index=False)
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        for fn, name in ((plot_optimization_history, "history"), (plot_param_importances, "importance")):
            fn(study).figure.savefig(paths.REPO_ROOT / "optuna_studies" / f"task4{sfx(args)}_{name}.png", dpi=120,
                                     bbox_inches="tight")
    except Exception as e:
        print("optuna plots skipped:", e)
    print("BEST", study.best_value, study.best_params)


def cmd_final(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    p = json.load(open(args.params or paths.REPO_ROOT / "configs" / f"task4_best{sfx(args)}.json"))
    seed_everything(42)
    ckpt = paths.checkpoint_dir() / f"task4_generator{sfx(args)}.pt"
    with mlflow.start_run(run_name=f"final_task4{sfx(args)}"):
        mlflow.log_params({**p, "epochs": args.epochs, "max_train": args.max_train,
                           "photometric": args.photometric, "lr_decay": args.lr_decay})
        best, hist = fit_gan(p, args.epochs, device, args.max_train, ckpt_path=ckpt, workers=args.workers,
                             sample_every=args.sample_every, resume=args.resume,
                             photometric=args.photometric, lr_decay=args.lr_decay, tag=sfx(args))
        mlflow.log_metric("best_val_obj", best)
        mlflow.log_artifact(str(ckpt))
    json.dump(hist, open(paths.checkpoint_dir() / f"task4_history{sfx(args)}.json", "w"))
    print("saved", ckpt, "best val obj", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["optuna", "final"])
    ap.add_argument("--n_trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--params", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/*.resume")
    ap.add_argument("--tag", default="", help="suffix for study/config/checkpoint names, e.g. v2 (keeps v1 untouched)")
    ap.add_argument("--photometric", action="store_true", help="brightness/contrast jitter on the photo only")
    ap.add_argument("--lr_decay", action="store_true", help="linear lr decay over the second half of training")
    a = ap.parse_args()
    seed_everything(42)
    cmd_optuna(a) if a.mode == "optuna" else cmd_final(a)
