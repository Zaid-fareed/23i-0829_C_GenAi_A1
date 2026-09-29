"""Save/continue-training support shared by every engine.

After each epoch the engines write `<checkpoint>.resume` (model + optimizer + scheduler + best-so-far + history)
atomically. Running the same `final` command again with `--resume` loads it and continues from the next epoch,
so a Kaggle timeout / new session does not lose progress. The `<checkpoint>` itself keeps only the best weights.
"""
import os
from pathlib import Path

import torch


def default_path(ckpt_path, resume_path=None):
    if resume_path:
        return str(resume_path)
    return f"{ckpt_path}.resume" if ckpt_path else None


def save(path, *, epoch, objs: dict, best, best_state, history):
    """objs: name -> anything with state_dict() (models, optimizers, schedulers, GradScaler)."""
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save({"epoch": epoch, "best": best, "best_state": best_state, "history": history,
                "objs": {k: v.state_dict() for k, v in objs.items()}}, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous resume file


def load(path, *, objs: dict, device):
    """Returns (start_epoch, best, best_state, history) or None when there is nothing to resume."""
    if not path or not Path(path).exists():
        return None
    st = torch.load(path, map_location=device, weights_only=False)
    for k, v in objs.items():
        v.load_state_dict(st["objs"][k])
    print(f"[resume] {path}: continuing from epoch {st['epoch'] + 2} (best so far {st['best']:.4f})")
    return st["epoch"] + 1, st["best"], st["best_state"], st["history"]
