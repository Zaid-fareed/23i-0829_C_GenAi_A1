import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Per-image PSNR for tensors in [0,1], shape (B,C,H,W) -> (B,)."""
    mse = ((x - y) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / (mse + eps))


def setup_mlflow(experiment: str):
    """ALWAYS sqlite backend (file store is deprecated). Run from repo root so mlflow.db lands there."""
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    return mlflow


def remaining_trials(study, total: int) -> int:
    """`--n_trials` means TOTAL trials for the study: a re-run in a new session only adds the missing ones."""
    import optuna
    done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,
                                                       optuna.trial.TrialState.PRUNED)))
    left = max(0, total - done)
    print(f"[optuna] {done} trials already finished, running {left} more (target {total})")
    return left
