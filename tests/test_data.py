"""Run: python -m pytest tests -q   (or: python -m unittest discover tests)"""
import json
import unittest

import numpy as np
import torch

from src.data.corruptions import (BLUR_KERNELS, BLUR_SIGMA, FIXED, OCC_AREA, OCC_TOL, SP_P, TYPES, apply_spec,
                                  gaussian_blur, make_spec, rects_to_mask)
from src.data.make_manifests import TEST_CONDITIONS, build_entries
from src.data.pets import BalancedCorruptionBatchSampler, PetsManifestDataset, PetsTrainCorrupted
from src.utils import paths

GRAY = torch.full((3, 128, 128), 0.5)


class TestCorruptions(unittest.TestCase):
    def test_salt_pepper(self):
        for p in (0.03, 0.08, 0.15):
            spec = make_spec("salt_pepper", np.random.default_rng(1), "low")
            spec.update(p=p)
            out = apply_spec(GRAY, spec)
            changed = (out != 0.5).any(0)
            self.assertEqual(int(changed.sum()), round(p * 128 * 128))
            vals = out[:, changed]
            self.assertTrue(((vals == 0) | (vals == 1)).all())
            self.assertTrue((vals[0] == vals[1]).all())  # same value across channels
            self.assertTrue(0.4 < vals[0].mean() < 0.6)  # ~50/50 black/white

    def test_blur(self):
        out = gaussian_blur(GRAY, 7, 2.5)
        self.assertTrue(torch.allclose(out, GRAY, atol=1e-5))  # kernel normalised, constant image unchanged
        img = torch.rand(3, 128, 128)
        self.assertLess(gaussian_blur(img, 5, 1.5).std(), img.std())

    def test_ranges_random(self):
        rng = np.random.default_rng(0)
        counts = {t: 0 for t in TYPES}
        for _ in range(4000):
            t = TYPES[rng.integers(4)]
            counts[t] += 1
        for c in counts.values():
            self.assertTrue(850 < c < 1150)
        for _ in range(300):
            s = make_spec("salt_pepper", rng)
            self.assertTrue(SP_P[0] <= s["p"] <= SP_P[1])
            b = make_spec("blur", rng)
            self.assertIn(b["kernel"], BLUR_KERNELS)
            self.assertTrue(BLUR_SIGMA[0] <= b["sigma"] <= BLUR_SIGMA[1])

    def test_occlusion_random_coverage(self):
        rng = np.random.default_rng(0)
        ns = set()
        for _ in range(300):
            s = make_spec("occlusion", rng)
            ns.add(s["n_rects"])
            self.assertTrue(OCC_AREA[0] <= s["coverage"] <= OCC_AREA[1], s["coverage"])
            self.assertEqual(len(s["rects"]), s["n_rects"])
            self.assertAlmostEqual(rects_to_mask(s["rects"]).mean(), s["coverage"])
            for x0, y0, x1, y1 in s["rects"]:
                self.assertTrue(0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128)
            out = apply_spec(GRAY, s)
            self.assertAlmostEqual(float((out == 0).all(0).float().mean()), s["coverage"], places=6)
        self.assertEqual(ns, {1, 2, 3})

    def test_occlusion_fixed_severities(self):
        rng = np.random.default_rng(3)
        for sev, cfg in FIXED["occlusion"].items():
            for _ in range(50):
                s = make_spec("occlusion", rng, sev)
                self.assertEqual(s["n_rects"], cfg["n_rects"])
                self.assertLessEqual(abs(s["coverage"] - cfg["target"]), OCC_TOL + 1e-9)

    def test_apply_deterministic(self):
        img = torch.rand(3, 128, 128)
        rng = np.random.default_rng(5)
        for t in TYPES:
            s = make_spec(t, rng)
            self.assertTrue(torch.equal(apply_spec(img, s), apply_spec(img, s)))


class TestManifests(unittest.TestCase):
    names = [f"img_{i}" for i in range(240)]

    def test_deterministic(self):
        a = build_entries(self.names, "test", 42)
        b = build_entries(self.names, "test", 42)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c = build_entries(self.names, "test", 43)
        self.assertNotEqual(json.dumps(a, sort_keys=True), json.dumps(c, sort_keys=True))

    def test_test_fixed_and_balanced(self):
        for mode in ("cycle", "full"):
            es = build_entries(self.names, "test", 42, mode)
            for e in es:
                if e["type"] == "salt_pepper":
                    self.assertEqual(e["p"], FIXED["salt_pepper"][e["severity"]]["p"])
                if e["type"] == "blur":
                    self.assertEqual((e["kernel"], e["sigma"]),
                                     (FIXED["blur"][e["severity"]]["kernel"], FIXED["blur"][e["severity"]]["sigma"]))
                if e["type"] == "occlusion":
                    self.assertEqual(e["n_rects"], FIXED["occlusion"][e["severity"]]["n_rects"])
                self.assertTrue({"seed", "type", "severity", "label"} <= set(e))
            if mode == "cycle":
                self.assertEqual(len(es), 240)
                lab = np.bincount([e["label"] for e in es], minlength=4)
                self.assertTrue((lab == 60).all())
            else:
                self.assertEqual(len(es), 240 * len(TEST_CONDITIONS))

    def test_val_balanced(self):
        es = build_entries(self.names, "val", 42)
        self.assertTrue((np.bincount([e["label"] for e in es], minlength=4) == 60).all())

    def test_replay_identical(self):
        images = torch.randint(0, 256, (240, 3, 128, 128), dtype=torch.uint8)
        es = build_entries(self.names, "test", 42, "cycle")
        ds = PetsManifestDataset(es, images)
        for i in (0, 7, 100, 239):
            a, _, lab, _ = ds[i]
            b, _, _, _ = ds[i]
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(lab, es[i]["label"])


class TestTrainData(unittest.TestCase):
    def test_balanced_sampler(self):
        s = BalancedCorruptionBatchSampler(100, 16, seed=1)
        n = 0
        for batch in s:
            n += 1
            self.assertTrue((np.bincount([l for _, l in batch], minlength=4) == 4).all())
        self.assertEqual(n, 6)

    def test_train_dataset_runtime_random(self):
        images = torch.randint(0, 256, (8, 3, 128, 128), dtype=torch.uint8)
        ds = PetsTrainCorrupted(images)
        a, clean, lab, idx = ds[(3, 1)]
        self.assertEqual((lab, idx, tuple(a.shape)), (1, 3, (3, 128, 128)))
        b, *_ = ds[(3, 1)]
        self.assertFalse(torch.equal(a, b))  # different corruption sampled per load
        labels = [ds[i][2] for i in range(400) if i < 8 or True] if False else [ds[i % 8][2] for i in range(400)]
        self.assertTrue(all(70 < labels.count(k) < 130 for k in range(4)))


class TestSplit(unittest.TestCase):
    def test_split_sizes(self):
        p = paths.split_json_path()
        if not p.exists():
            self.skipTest("pets_split.json not present")
        d = json.load(open(p))
        self.assertEqual((len(d["train"]), len(d["val"]), len(d["test"])), (2944, 736, 3669))
        self.assertEqual(len(set(d["train"]) & set(d["val"])), 0)
        self.assertEqual(len(set(d["train"] + d["val"]) & set(d["test"])), 0)


if __name__ == "__main__":
    unittest.main()
