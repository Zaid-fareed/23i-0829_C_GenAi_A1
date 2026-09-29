"""A run that crashes mid-training and is resumed must end exactly where an uninterrupted run ends."""
import copy

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.losses.recon import ReconLoss
from src.models.autoencoder import build_ae
from src.train.engine import fit_ae

PARAMS = {"channels": "c16", "bottleneck_dim": 8, "dropout": 0.0}


class Fake(Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(16, 3, 32, 32, generator=g)

    def __len__(self):
        return 16

    def __getitem__(self, i):
        return self.x[i] * 0.8, self.x[i], i % 4, i


class Boom(Exception):
    pass


class CrashingLoader:
    """Wraps a loader; raises when iterated for the n-th time (simulates a Kaggle session dying)."""

    def __init__(self, loader, crash_on):
        self.loader, self.crash_on, self.calls = loader, crash_on, 0

    def __iter__(self):
        self.calls += 1
        if self.calls == self.crash_on:
            raise Boom()
        return iter(self.loader)


def run(ckpt, crash_on=None, resume=False):
    torch.manual_seed(1)
    model = build_ae(PARAMS)
    tr = DataLoader(Fake(), batch_size=8, shuffle=False)
    va = DataLoader(Fake(), batch_size=8, shuffle=False)
    if crash_on:
        va = CrashingLoader(va, crash_on)
    best, hist = fit_ae(model, ReconLoss(0.8), tr, va, lr=1e-3, epochs=4, device=torch.device("cpu"),
                        ckpt_path=str(ckpt), resume=resume)
    return model, best, hist


def test_resume_matches_uninterrupted(tmp_path):
    ref_model, ref_best, ref_hist = run(tmp_path / "ref.pt")
    with pytest.raises(Boom):
        run(tmp_path / "a.pt", crash_on=3)                    # dies while validating epoch 3
    model, best, hist = run(tmp_path / "a.pt", resume=True)   # continues from the last completed epoch
    assert [h["epoch"] for h in hist] == [0, 1, 2, 3] and len(hist) == len(ref_hist)
    assert best == pytest.approx(ref_best, abs=1e-6)
    for (k, a), (_, b) in zip(model.state_dict().items(), ref_model.state_dict().items()):
        assert torch.allclose(a.float(), b.float(), atol=1e-5), k


def test_resume_of_finished_run_is_noop(tmp_path):
    run(tmp_path / "b.pt")
    _, _, hist = run(tmp_path / "b.pt", resume=True)
    assert len(hist) == 4
