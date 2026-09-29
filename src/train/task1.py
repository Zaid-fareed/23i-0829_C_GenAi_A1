"""Task 1: universal denoising autoencoder. Run from the repo root.

  python -m src.train.task1 optuna --n_trials 10 --epochs 4 --max_train 1500
  python -m src.train.task1 final  --epochs 40                      # uses configs/task1_best.json

Optuna objective = val 0.5*L1 + 0.5*(1-SSIM) on the deterministic val manifest (independent of the tuned alpha).
Everything is logged to MLflow (sqlite:///mlflow.db); the Optuna study lives in optuna_studies/task1.db.
"""
import argparse
import json

import mlflow
import optuna
import torch

from src.data.pets import build_eval_loader, build_train_loader
from src.losses.recon import ReconLoss
from src.models.autoencoder import CHANNEL_OPTIONS, build_ae
from src.train.engine import fit_ae
from src.utils import paths
from src.utils.common import get_device, remaining_trials, seed_everything, setup_mlflow

EXPERIMENT = "task1_universal_restoration"


def suggest(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", [32, 64, 128, 256]),
        "channels": trial.suggest_categorical("channels", list(CHANNEL_OPTIONS)),
        "dropout": trial.suggest_float("dropout", 0.0, 0.15),
        "alpha": trial.suggest_float("alpha", 0.2, 0.95),  # weight of L1 (1-alpha on 1-SSIM)
    }


def run_training(params, epochs, max_train, device, trial=None, ckpt_path=None, workers=None, prefix="",
                 resume=False):
    model = build_ae(params)
    train_loader = build_train_loader("ae", params["batch_size"], max_images=max_train, num_workers=workers)
    val_loader = build_eval_loader("val", batch_size=64, num_workers=workers)
    return fit_ae(model, ReconLoss(params["alpha"]), train_loader, val_loader, lr=params["lr"], epochs=epochs,
                  device=device, trial=trial, ckpt_path=ckpt_path,
                  ckpt_extra={"params": params, "task": "task1"}, log_prefix=prefix, resume=resume)


def objective(trial, args, device):
    params = suggest(trial)
    mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)
    status = "FINISHED"
    try:
        mlflow.log_params(params)
        best, _ = run_training(params, args.epochs, args.max_train, device, trial=trial,
                               workers=args.workers, prefix=f"[t{trial.number}] ")
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
        study_name="task1_dae", storage=f"sqlite:///{paths.REPO_ROOT / 'optuna_studies' / 'task1.db'}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1), load_if_exists=True)
    with mlflow.start_run(run_name="optuna_study_task1"):
        mlflow.log_params({"n_trials": args.n_trials, "trial_epochs": args.epochs, "max_train": args.max_train})
        study.optimize(lambda t: objective(t, args, device), n_trials=remaining_trials(study, args.n_trials))
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    (paths.REPO_ROOT / "configs").mkdir(exist_ok=True)
    json.dump(study.best_params, open(paths.REPO_ROOT / "configs" / "task1_best.json", "w"), indent=2)
    study.trials_dataframe().to_csv(paths.REPO_ROOT / "optuna_studies" / "task1_trials.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        for fn, name in ((plot_optimization_history, "history"), (plot_param_importances, "importance")):
            fn(study).figure.savefig(paths.REPO_ROOT / "optuna_studies" / f"task1_{name}.png", dpi=120, bbox_inches="tight")
    except Exception as e:  # plots are optional
        print("optuna plots skipped:", e)
    print("BEST", study.best_value, study.best_params)


def cmd_final(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    params = json.load(open(args.params or paths.REPO_ROOT / "configs" / "task1_best.json"))
    seed_everything(42)
    ckpt = paths.checkpoint_dir() / "task1_dae.pt"
    with mlflow.start_run(run_name="final_task1"):
        mlflow.log_params({**params, "epochs": args.epochs, "max_train": args.max_train})
        best, hist = run_training(params, args.epochs, args.max_train, device, ckpt_path=ckpt, workers=args.workers,
                             resume=args.resume)
        mlflow.log_metric("best_val_obj", best)
        mlflow.log_artifact(str(ckpt))
    json.dump(hist, open(paths.checkpoint_dir() / "task1_history.json", "w"))
    print("saved", ckpt, "best val obj", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["optuna", "final"])
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/*.resume")
    a = ap.parse_args()
    seed_everything(42)
    cmd_optuna(a) if a.mode == "optuna" else cmd_final(a)
