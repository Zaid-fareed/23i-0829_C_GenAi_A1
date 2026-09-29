"""The backend's numpy corruptions must equal the torch training-time ones (skipped if torch is unavailable)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import corruptions as bc  # noqa: E402
from src.data import corruptions as tc  # noqa: E402


@pytest.mark.parametrize("ctype", ["clean", "salt_pepper", "blur", "occlusion"])
@pytest.mark.parametrize("severity", ["low", "medium", "high", "random"])
def test_matches_training_code(ctype, severity):
    rng_img = np.random.default_rng(0)
    img = rng_img.random((3, 128, 128)).astype(np.float32)
    for seed in range(4):
        spec_t = tc.make_spec(ctype, np.random.default_rng(seed), severity)
        spec_b = bc.make_spec(ctype, np.random.default_rng(seed), severity)
        assert spec_t == spec_b
        out_t = tc.apply_spec(torch.from_numpy(img), spec_t).numpy()
        out_b = bc.apply_spec(img, spec_b)
        assert np.abs(out_t - out_b).max() < 1e-5
