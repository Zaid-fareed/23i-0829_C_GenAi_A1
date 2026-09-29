"""Train/eval engine for the Task 2 corruption-type classifier."""
import copy

import mlflow
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data.corruptions import TYPES
from src.train import resume as R

N_CLASSES = len(TYPES)


@torch.no_grad()
def predict_classifier(model, loader, device):
    """Returns probs (N,4), labels (N,), entry_idx (N,) -- kept separate from metrics so the same
    predictions can be reused for hard-routing evaluation (Task 2) without re-running the model."""
    model.eval()
    probs, labels, idxs = [], [], []
    for x, _, y, idx in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(y)
        idxs.append(idx)
    return torch.cat(probs), torch.cat(labels), torch.cat(idxs)


def classifier_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    """JSON-serialisable metrics dict: accuracy, macro P/R/F1, per-class P/R/F1/support, raw + row-normalized CM."""
    preds = probs.argmax(1)
    y, p = labels.numpy(), preds.numpy()
    acc = float((preds == labels).float().mean())
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=list(range(N_CLASSES)), zero_division=0)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(N_CLASSES)))
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    ce = float(F.cross_entropy(torch.log(probs.clamp_min(1e-8)), labels).item())
    return {
        "accuracy": acc, "macro_precision": float(m_prec), "macro_recall": float(m_rec), "macro_f1": float(m_f1),
        "ce_loss": ce,
        "per_class": {TYPES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                  "support": int(support[i])} for i in range(N_CLASSES)},
        "confusion_matrix": cm.tolist(), "confusion_matrix_normalized": cm_norm.tolist(),
    }


def fit_classifier(model, train_loader, val_loader, *, lr, epochs, device, weight_decay=1e-4,
                   trial=None, ckpt_path=None, ckpt_extra=None, log_prefix="", resume=False, resume_path=None):
    """Maximizes val macro-F1. Returns (best_macro_f1, history)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_state, history = -1.0, None, []
    rpath = R.default_path(ckpt_path, resume_path)
    objs = {"model": model, "opt": opt, "sched": sched}
    start = 0
    if resume and (r := R.load(rpath, objs=objs, device=device)):
        start, best, best_state, history = r
    for epoch in range(start, epochs):
        model.train()
        tot_loss, n, correct = 0.0, 0, 0
        for x, _, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            b = x.size(0)
            n += b
            tot_loss += loss.item() * b
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        probs, labels, _ = predict_classifier(model, val_loader, device)
        val = classifier_metrics(probs, labels)
        rec = {"epoch": epoch, "train_loss": tot_loss / n, "train_acc": correct / n,
              "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "val_ce": val["ce_loss"]}
        history.append(rec)
        print(f"{log_prefix}ep {epoch + 1}/{epochs} train_loss {rec['train_loss']:.4f} train_acc {rec['train_acc']:.3f} "
              f"val_acc {val['accuracy']:.3f} val_macroF1 {val['macro_f1']:.3f}")
        if mlflow.active_run():
            mlflow.log_metrics(rec, step=epoch)
        if val["macro_f1"] > best:
            best, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save({"state_dict": best_state, "val": val, "epoch": epoch, **(ckpt_extra or {})}, ckpt_path)
        R.save(rpath, epoch=epoch, objs=objs, best=best, best_state=best_state, history=history)
        if trial is not None:
            trial.report(val["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, history
