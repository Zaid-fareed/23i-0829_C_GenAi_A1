"""Task 2, part B: 3 specialist autoencoders (salt_pepper, blur, occlusion), independently trained.

Architecture/hyperparameters are chosen by ONE shared Optuna search: each trial trains a small version of
ALL THREE specialists with the same (lr, bottleneck_dim, channels, batch_size, alpha) and the objective is the
MEAN of their validation reconstruction objectives. This finds one architecture that works well across all
three corruption types, in the spirit of "shared search for common architecture". After the search, the three
specialists are trained to convergence INDEPENDENTLY (separate weights, separate checkpoints, separate MLflow runs).

Pruning: omitted for the shared search (NopPruner) -- each trial trains 3 models sequentially so a single
intermediate value per trial isn't informative for a step-wise pruner like MedianPruner.

  python -m src.train.task2_specialists optuna --n_trials 10 --epochs 3 --max_train 1000
  python -m src.train.task2_specialists final  --epochs 30
"""
import argparse
import json

import mlflow
import numpy as np
import optuna

from src.data.corruptions import TYPES
from src.data.pets import build_eval_loader_by_type, build_single_type_loader
from src.losses.recon import ReconLoss
from src.models.autoencoder import CHANNEL_OPTIONS, build_ae
from src.train.engine import fit_ae
from src.utils import paths
from src.utils.common import get_device, remaining_trials, seed_everything, setup_mlflow

EXPERIMENT = "task2_specialists"
CORRUPT_TYPES = [t for t in TYPES if t != "clean"]  # salt_pepper, blur, occlusion


def suggest(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", [32, 64, 128, 256]),
        "channels": trial.suggest_categorical("channels", list(CHANNEL_OPTIONS)),
        "dropout": trial.suggest_float("dropout", 0.0, 0.15),
        "alpha": trial.suggest_float("alpha", 0.2, 0.95),
    }


def train_one_specialist(ctype, params, epochs, max_train, device, ckpt_path=None, workers=None, prefix="",
                         resume=False):
    model = build_ae(params)
    train_loader = build_single_type_loader(ctype, params["batch_size"], max_images=max_train, num_workers=workers)
    val_loader = build_eval_loader_by_type("val", ctype, batch_size=64, num_workers=workers)
    return fit_ae(model, ReconLoss(params["alpha"]), train_loader, val_loader, lr=params["lr"], epochs=epochs,
                  device=device, ckpt_path=ckpt_path,
                  ckpt_extra={"params": params, "task": "task2_specialist", "corruption_type": ctype},
                  log_prefix=prefix, resume=resume)


def objective(trial, args, device):
    params = suggest(trial)
    mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)
    try:
        mlflow.log_params(params)
        objs = {}
        for ctype in CORRUPT_TYPES:
            best, _ = train_one_specialist(ctype, params, args.epochs, args.max_train, device,
                                           workers=args.workers, prefix=f"[t{trial.number}-{ctype}] ")
            objs[ctype] = best
            mlflow.log_metric(f"val_obj_{ctype}", best)
        mean_obj = float(np.mean(list(objs.values())))
        mlflow.log_metric("mean_val_obj", mean_obj)
        return mean_obj
    finally:
        mlflow.end_run("FINISHED")


def cmd_optuna(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    (paths.REPO_ROOT / "optuna_studies").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name="task2_specialists_shared_arch",
        storage=f"sqlite:///{paths.REPO_ROOT / 'optuna_studies' / 'task2_specialists.db'}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.NopPruner(),
        load_if_exists=True)
    with mlflow.start_run(run_name="optuna_study_task2_specialists"):
        mlflow.log_params({"n_trials": args.n_trials, "trial_epochs": args.epochs, "max_train": args.max_train})
        study.optimize(lambda t: objective(t, args, device), n_trials=remaining_trials(study, args.n_trials))
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    (paths.REPO_ROOT / "configs").mkdir(exist_ok=True)
    json.dump(study.best_params, open(paths.REPO_ROOT / "configs" / "task2_specialists_best.json", "w"), indent=2)
    study.trials_dataframe().to_csv(paths.REPO_ROOT / "optuna_studies" / "task2_specialists_trials.csv", index=False)
    print("BEST (mean val obj across 3 specialists)", study.best_value, study.best_params)


def cmd_final(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    params = json.load(open(args.params or paths.REPO_ROOT / "configs" / "task2_specialists_best.json"))
    seed_everything(42)
    for ctype in CORRUPT_TYPES:
        ckpt = paths.checkpoint_dir() / f"task2_specialist_{ctype}.pt"
        with mlflow.start_run(run_name=f"final_specialist_{ctype}"):
            mlflow.log_params({**params, "epochs": args.epochs, "corruption_type": ctype, "max_train": args.max_train})
            best, hist = train_one_specialist(ctype, params, args.epochs, args.max_train, device, ckpt_path=ckpt,
                                              workers=args.workers, prefix=f"[{ctype}] ", resume=args.resume)
            mlflow.log_metric("best_val_obj", best)
            mlflow.log_artifact(str(ckpt))
        json.dump(hist, open(paths.checkpoint_dir() / f"task2_specialist_{ctype}_history.json", "w"))
        print("saved", ckpt, "best val obj", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["optuna", "final"])
    ap.add_argument("--n_trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/*.resume")
    a = ap.parse_args()
    seed_everything(42)
    cmd_optuna(a) if a.mode == "optuna" else cmd_final(a)
