"""Task 3: soft mixture-of-experts. Gate = Task 2 classifier (loaded), experts = Task 2 specialists (loaded).
Architecture is NOT re-searched here (it's inherited from Task 2); Optuna only tunes the MoE training/loss
hyperparameters, as specified: joint lr, temperature, classification weight, balance weight, reconstruction
weighting (alpha, tying L1+SSIM weights -- see src/train/engine_moe.py docstring).

Two-stage training every run:
  1. warm-up  : experts frozen, only the gate is trained (fixed --warmup_lr, --warmup_epochs; not Optuna-tuned)
  2. joint    : everything unfrozen, trained with the (smaller, Optuna-tuned) `lr`

  python -m src.train.task3 optuna --n_trials 10 --warmup_epochs 1 --epochs 3 --max_train 1500
  python -m src.train.task3 final  --warmup_epochs 2 --epochs 25
"""
import argparse
import json

import mlflow
import optuna

from src.data.pets import build_eval_loader, build_train_loader
from src.models.moe import build_moe_from_checkpoints, set_experts_trainable
from src.train.engine_moe import fit_moe
from src.utils import paths
from src.utils.common import get_device, remaining_trials, seed_everything, setup_mlflow

EXPERIMENT = "task3_soft_moe"


def sfx(args):
    return f"_{args.tag}" if args.tag else ""


def suggest(trial: optuna.Trial, alpha_min: float = 0.2) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),           # joint fine-tune lr (smaller than warm-up)
        "temperature": trial.suggest_float("temperature", 0.3, 3.0, log=True),
        "w_cls": trial.suggest_float("w_cls", 0.01, 2.0, log=True),      # classification weight
        "w_bal": trial.suggest_float("w_bal", 0.0, 2.0),                 # balance weight
        "alpha": trial.suggest_float("alpha", alpha_min, 0.95),         # reconstruction L1/SSIM weighting
    }


def run_training(params, args, device, trial=None, ckpt_path=None, prefix="", resume=False):
    model, gate_params, expert_params = build_moe_from_checkpoints(
        args.classifier, {"salt_pepper": args.salt_pepper, "blur": args.blur, "occlusion": args.occlusion},
        device, temperature=params["temperature"])
    train_loader = build_train_loader("ae", args.batch_size, max_images=args.max_train, num_workers=args.workers)
    val_loader = build_eval_loader("val", batch_size=64, num_workers=args.workers)

    set_experts_trainable(model, False)
    fit_moe(model, train_loader, val_loader, lr=args.warmup_lr, epochs=args.warmup_epochs, device=device,
           alpha=params["alpha"], w_cls=params["w_cls"], w_bal=params["w_bal"], log_prefix=prefix + "[warmup] ", resume=resume,
            resume_path=f"{ckpt_path}.warmup.resume" if ckpt_path else None)

    set_experts_trainable(model, True)
    best, hist = fit_moe(model, train_loader, val_loader, lr=params["lr"], epochs=args.epochs, device=device,
                         alpha=params["alpha"], w_cls=params["w_cls"], w_bal=params["w_bal"], trial=trial,
                         ckpt_path=ckpt_path,
                         ckpt_extra={"params": params, "task": "task3_moe", "gate_params": gate_params,
                                    "expert_params": expert_params}, log_prefix=prefix + "[joint] ",
                         resume=resume)
    return best, hist


def objective(trial, args, device):
    params = suggest(trial, args.alpha_min)
    mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)
    status = "FINISHED"
    try:
        mlflow.log_params({**params, "warmup_epochs": args.warmup_epochs, "warmup_lr": args.warmup_lr,
                           "batch_size": args.batch_size})
        best, _ = run_training(params, args, device, trial=trial, prefix=f"[t{trial.number}] ")
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


def add_ckpt_args(ap):
    ap.add_argument("--classifier", default=str(paths.checkpoint_dir() / "task2_classifier.pt"))
    ap.add_argument("--salt_pepper", default=str(paths.checkpoint_dir() / "task2_specialist_salt_pepper.pt"))
    ap.add_argument("--blur", default=str(paths.checkpoint_dir() / "task2_specialist_blur.pt"))
    ap.add_argument("--occlusion", default=str(paths.checkpoint_dir() / "task2_specialist_occlusion.pt"))


def cmd_optuna(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    (paths.REPO_ROOT / "optuna_studies").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=f"task3_moe{sfx(args)}",
        storage=f"sqlite:///{paths.REPO_ROOT / 'optuna_studies' / f'task3{sfx(args)}.db'}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1), load_if_exists=True)
    with mlflow.start_run(run_name=f"optuna_study_task3{sfx(args)}"):
        mlflow.log_params({"n_trials": args.n_trials, "trial_epochs": args.epochs, "max_train": args.max_train,
                           "alpha_min": args.alpha_min})
        study.optimize(lambda t: objective(t, args, device), n_trials=remaining_trials(study, args.n_trials))
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    (paths.REPO_ROOT / "configs").mkdir(exist_ok=True)
    json.dump(study.best_params, open(paths.REPO_ROOT / "configs" / f"task3_best{sfx(args)}.json", "w"), indent=2)
    study.trials_dataframe().to_csv(paths.REPO_ROOT / "optuna_studies" / f"task3{sfx(args)}_trials.csv", index=False)
    print("BEST", study.best_value, study.best_params)


def cmd_final(args):
    device = get_device()
    setup_mlflow(EXPERIMENT)
    params = json.load(open(args.params or paths.REPO_ROOT / "configs" / f"task3_best{sfx(args)}.json"))
    seed_everything(42)
    ckpt = paths.checkpoint_dir() / f"task3_moe{sfx(args)}.pt"
    with mlflow.start_run(run_name=f"final_task3{sfx(args)}"):
        mlflow.log_params({**params, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
                           "warmup_lr": args.warmup_lr, "batch_size": args.batch_size})
        best, hist = run_training(params, args, device, ckpt_path=ckpt, resume=args.resume)
        mlflow.log_metric("best_val_obj", best)
        mlflow.log_artifact(str(ckpt))
    json.dump(hist, open(paths.checkpoint_dir() / f"task3_history{sfx(args)}.json", "w"))
    print("saved", ckpt, "best val obj", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["optuna", "final"])
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3, help="joint fine-tune epochs")
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--warmup_lr", type=float, default=1e-3, help="fixed, not Optuna-tuned")
    ap.add_argument("--batch_size", type=int, default=32, help="fixed, not Optuna-tuned")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/*.resume")
    ap.add_argument("--tag", default="", help="suffix for study/config/checkpoint names, e.g. v2 (keeps the original files untouched)")
    ap.add_argument("--alpha_min", type=float, default=0.2, help="lower bound of the searched reconstruction weight alpha")
    add_ckpt_args(ap)
    a = ap.parse_args()
    for k in ("salt_pepper", "blur", "occlusion"):  # with --tag, use that tag's specialists unless a path was given
        if a.tag and getattr(a, k) == str(paths.checkpoint_dir() / f"task2_specialist_{k}.pt"):
            setattr(a, k, str(paths.checkpoint_dir() / f"task2_specialist_{k}_{a.tag}.pt"))
    seed_everything(42)
    cmd_optuna(a) if a.mode == "optuna" else cmd_final(a)
