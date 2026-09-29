# Training on Kaggle (Tasks 1-4)

## 0. One-time setup
1. On your PC, from the repo root: `python scripts/pack_for_kaggle.py` -> creates `kaggle_code.zip`.
2. kaggle.com -> **Datasets -> New Dataset** -> upload `kaggle_code.zip`, name it `genai-code`.
3. Also add the FS2K dataset for Task 4 later (search Kaggle datasets for "FS2K"; if none, download from the
   official repo's Google Drive link and upload it as your own dataset). It must contain `anno_train.json`.
4. **New Notebook** -> right side: *Settings* -> Accelerator **GPU T4 x1**, **Internet ON**, Persistence: Files.
   *Add Input* -> your `genai-code` dataset (and FS2K when doing Task 4).

## 1. Notebook setup cells (run every new session; each block = its own cell, in order)
Cell 1 - unpack the code (works whether or not Kaggle auto-extracted the zip):
```python
import os, glob, subprocess
R = "/kaggle/working/repo"; os.makedirs(R, exist_ok=True)
z = glob.glob("/kaggle/input/genai-code/**/kaggle_code.zip", recursive=True)
if z: subprocess.run(["unzip", "-oq", z[0], "-d", R])
else: subprocess.run(f"cp -r /kaggle/input/genai-code/* {R}/", shell=True)
os.chdir(R); print(os.listdir(R))
```
Cell 2 - install packages:
```python
!pip install -q optuna mlflow pytorch-msssim onnx onnxruntime
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
```
Cell 3 - download Pets + create the 80/20 split (seed 42); needs Internet ON. Expect `{'train': 2944, 'val': 736, 'test': 3669}`:
```python
!python scripts/make_pets_split.py
```
Cell 4 - manifests (first session only; use the PETS_IMAGE_DIR path that Cell 3 printed):
```python
os.environ["PETS_IMAGE_DIR"] = "/kaggle/working/data/oxford-iiit-pet/images"
!python -m src.data.make_manifests
```
Cell 5 - timing test (checks training runs, shows the time of one epoch):
```python
!python -m src.train.task1 optuna --n_trials 1 --epochs 1 --max_train 600
```

## 2. Reduced-data strategy (your professor's advice)
Train on a **subset** first, and check whether the model *learns* (val metrics improve) or *memorizes*
(train metrics great, val metrics poor). Everything takes `--max_train N` (number of training images).

| Stage | Data | Why |
|---|---|---|
| Optuna trials | `--max_train 600`, 3-4 epochs each | many cheap trials |
| Final "quick" run | `--max_train 736` (about 25% of train), 25-30 epochs | ~1 h total, proves the pipeline |
| Final "full" run (if time) | no `--max_train`, 30-40 epochs | best quality |

Validation always uses the full 736-image val manifest, and test stays untouched until the end.
**First, time one epoch** (`--epochs 1`) and scale the numbers below to your patience.

## 3. Task 1
```python
!python -m src.train.task1 optuna --n_trials 8 --epochs 3 --max_train 600
!python -m src.train.task1 final --epochs 30 --max_train 736
# memorization check: train-subset vs val metrics + curves
!python -m src.eval.generalization --ckpt checkpoints/task1_dae.pt --history checkpoints/task1_history.json --n_train 736
!python -m src.eval.task1_eval
!python -m src.export.export_task1
```
Read `reports/task1_generalization/generalization.json`: `gap_ssim` near 0 (say < 0.03) = generalizes;
train SSIM much higher than val SSIM = memorizing (use more data, more dropout, or fewer epochs).

## 4. Task 2
```python
!python -m src.train.task2_classifier optuna --n_trials 8 --epochs 3 --max_train 600
!python -m src.train.task2_classifier final --epochs 15 --max_train 736
!python -m src.train.task2_specialists optuna --n_trials 6 --epochs 3 --max_train 600
!python -m src.train.task2_specialists final --epochs 25 --max_train 736
!python -m src.eval.task2_classifier_eval
!python -m src.eval.task2_routing_eval
!python -m src.export.export_task2
```

## 5. Task 3 (needs Task 2 checkpoints)
```python
!python -m src.train.task3 optuna --n_trials 8 --warmup_epochs 1 --epochs 3 --max_train 600
!python -m src.train.task3 final --warmup_epochs 2 --epochs 12 --max_train 736
!python -m src.eval.task3_eval
!python -m src.export.export_task3
```

## 6. Task 4 (FS2K)  - add the FS2K dataset as an input first
```python
!python -m src.train.task4 optuna --n_trials 8 --epochs 15 --max_train 400
!python -m src.train.task4 final --epochs 100 --sample_every 10
!python -m src.eval.task4_eval
!python -m src.export.export_task4
```
If the first run crashes with "sketch for ... not found", run this and paste me the output (FS2K folder naming
is the one thing I could not verify without the data):
```python
!find /kaggle/input -maxdepth 4 | head -40
!python -c "import json,glob;p=glob.glob('/kaggle/input/**/anno_train.json',recursive=True)[0];print(json.load(open(p))[:2])"
```

## 7. Save everything (Kaggle wipes /kaggle/working between sessions unless saved!)
```python
!zip -rq results.zip checkpoints onnx_models reports configs optuna_studies mlflow.db manifests
```
Download `results.zip` from the notebook's *Output* panel. Do this after EVERY task (and before a session ends).

## 8. Continuing later WITHOUT losing progress
What is saved and reused automatically:
* **Training epochs:** after every epoch a `checkpoints/<name>.pt.resume` file (model + optimizer + scheduler +
  history) is written. Re-run the SAME `final` command with `--resume` and it continues from the next epoch.
  If a stage already finished, `--resume` just skips it. (Task 3 keeps a separate warm-up resume file.)
* **Optuna:** studies live in `optuna_studies/*.db` and are reloaded; `--n_trials N` means N TOTAL trials, so
  re-running only adds the missing ones. The best config is re-written to `configs/*_best.json` each time.
* **Best weights:** `checkpoints/<name>.pt` always holds the best-validation weights so far.

New session recipe:
1. Kaggle -> Datasets -> New Dataset -> upload your downloaded `results.zip` (name it e.g. `genai-results`;
   for later updates use the dataset's **New Version** instead of creating another one). Add it as notebook input
   together with `genai-code`.
2. In the new notebook, after the unzip/pip cells from section 1, restore your progress and re-create the data cache:
```python
!cd /kaggle/working/repo && unzip -oq /kaggle/input/genai-results/results.zip
!python scripts/make_pets_split.py    # REQUIRED every session: re-downloads the images; the split it writes is identical (seed 42)
```
   (`results.zip` already contains `manifests/`, so the split and val/test manifests stay identical - do NOT
   re-run `make_manifests` after restoring.) Set `PETS_IMAGE_DIR` again as in section 1.
3. Re-run the same commands with `--resume` on the `final` lines, e.g.
```python
!python -m src.train.task1 final --epochs 30 --max_train 736 --resume
!python -m src.train.task4 final --epochs 100 --sample_every 10 --resume
```
   and simply re-run the `optuna` lines unchanged (they pick up where they left off).
Tip: if a session dies unexpectedly, Kaggle keeps output only for *committed* runs. For long jobs use
**Save Version -> Save & Run All (Commit)**, then use that version's Output as an input to the next notebook.

## Troubleshooting
* CUDA out of memory -> lower batch size range in `suggest()`; * session limit -> run tasks in separate sessions,
  each ending with step 7 (upload the previous `results.zip` as an input dataset to continue Task 3 after Task 2).
* MLflow UI locally: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (this is needed for the demo video).
