# Generative AI - Assignment 1

Image restoration on the Oxford-IIIT Pet dataset with three architectures (universal autoencoder, hard-routed
specialists, soft mixture-of-experts) and style-conditioned face-to-sketch synthesis with a cGAN on FS2K, all served
by a FastAPI + React/Tailwind web application in Docker.

* **Technical report (IEEE format):** [`report/i230829_C_Report.pdf`](report/i230829_C_Report.pdf) - LaTeX source in `report/`
  (Overleaf package: `report/i230829_C_Report_latex.zip`) - Word version: `report/i230829_C_Report.docx`
* **Demonstration video:** <https://youtu.be/xUows4eME8M>
* **Trained ONNX models:** attached to the GitHub release `models-v1` of this repository (see step 2 below)

## Results at a glance (test set, 36,690 deterministic inputs = 3,669 images x 10 conditions)

| Task | Model | Headline result |
|---|---|---|
| 1 | Universal denoising autoencoder (4.0 M params, no skips) | SSIM 0.636 -> 0.767 on corrupted inputs; clean images 0.814 (identity would be 1.0) |
| 2 | Classifier + 3 specialists, hard routing | classifier accuracy 99.41 %, macro-F1 0.990; overall SSIM 0.779; clean 0.996 |
| 3 | Soft mixture of experts (gate + 3 experts + identity) | overall SSIM 0.783; clean 0.991; no inactive expert |
| 4 | Style-conditioned cGAN (U-Net 42 M + PatchGAN) | FS2K test: L1 0.105, SSIM 0.486, PSNR 15.8 dB (soft sketches, see report) |

The report discusses failure cases, a first autoencoder that under-fitted, specialists that lost colour (fixed by
restricting the L1/SSIM weight), and why SSIM and PSNR disagree on occlusion.

## 1. Run the application (evaluator quick start)

Requires Docker Desktop and Python 3 (for the model download only).

```bash
git clone https://github.com/Zaid-fareed/23i-0829_C_GenAi_A1.git
cd 23i-0829_C_GenAi_A1
python scripts/download_models.py        # fetches the 7 ONNX models into ./onnx_models and checks their SHA-256
docker compose up --build                # starts the back end and the front end
```

Open **http://localhost:3000**. Health check: http://localhost:3000/api/health. API docs: http://localhost:3000/api/docs.
Experiment tracking (MLflow UI with all recorded runs): `docker compose --profile tracking up` -> http://localhost:5000
(choose **Model training** in the top-left switch, then an experiment).

Workspaces: **Universal Restoration**, **Hard-Routed Restoration** (classifier probabilities, selected expert, oracle or
predicted routing), **Soft Mixture-of-Experts Restoration** (four routing weights), **Face-to-Sketch Generator**
(upload or webcam, Style 1/2/3, side by side, download). Each shows the model input, the output, the absolute error map,
the inference time and the corruption settings. Built-in samples come from the official **test** splits.

Smoke test of the running stack (54 checks): `python scripts/smoke_test_stack.py`.

## 2. Model files

The ONNX models are not stored in Git (160 MB for the sketch generator alone). They are attached to the release
`models-v1`. Checksums: `onnx_models/SHA256SUMS.txt`. Manual download: <https://github.com/Zaid-fareed/23i-0829_C_GenAi_A1/releases/tag/models-v1>,
then copy the seven `.onnx` files into `onnx_models/`.

| File | Task |
|---|---|
| `task1_universal_dae.onnx` | 1 - universal autoencoder |
| `task2_classifier.onnx`, `task2_specialist_{salt_pepper,blur,occlusion}.onnx` | 2 - classifier and specialists |
| `task3_soft_moe.onnx` | 3 - whole mixture (gate + experts + mixing), outputs image and weights |
| `task4_generator.onnx` | 4 - sketch generator (inputs: photo in [-1,1], style id) |

## 3. Repository layout

```
src/            data pipeline, models, losses, training (Optuna + MLflow), evaluation, ONNX export
backend/        FastAPI service (ONNX Runtime), tests, demo samples
frontend/       React + Tailwind (Vite) application, nginx config, Dockerfile
stitch_design/  Google Stitch exports (two design rounds) and the prompts used
report/         IEEE LaTeX report, figure/table build scripts, Word build script
scripts/        split creation, Kaggle packing, model download, smoke test, verification, screenshots
manifests/      pets_split.json, validation and test corruption manifests (single-condition and full)
configs/        selected hyper-parameters of every model
optuna_studies/ Optuna SQLite studies and trial tables
reports/        evaluation outputs (per task) and the full-protocol per-image metrics
mlflow.db       all MLflow experiment records (merged)
```

## 4. Reproduce the training (Kaggle T4)

Step-by-step cells are in [`KAGGLE_GUIDE.md`](KAGGLE_GUIDE.md). Order: split and manifests -> Task 1 -> Task 2
(classifier, then specialists) -> Task 3 -> Task 4, each `optuna` -> `final` -> eval -> export. Every `final` command
accepts `--resume`. The corrected second run of the specialists and the mixture uses `--tag v2 --alpha_min 0.55`.
Training takes roughly 3.5 GPU-hours in total including the corrected second run (per final run on a T4: Task 1 15 min,
each specialist 13 min, Task 3 21 min, Task 4 23 min; the Optuna studies add about 35 min per round).

Evaluation of the deployed models on the full protocol:

```bash
python -m src.eval.onnx_full_eval                              # ~1.5 h on 8 CPU threads (supports --start/--stop to split)
PYTHONPATH=backend python scripts/verify_onnx_results.py       # independent re-measurement (NumPy corruptions, scikit-image SSIM)
python report/build_tables.py && python report/build_qualitative.py && PYTHONPATH=. python report/build_analysis.py
```

## 5. Development without Docker

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt -r backend/requirements.txt
python scripts/download_models.py
MODEL_DIR=onnx_models PYTHONPATH=backend uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev            # http://localhost:5173 (proxies /api to :8000)
```

Tests: `pytest tests backend/tests` (back-end API tests need models in `onnx_models/` or `scripts/make_dummy_onnx.py` output).

## 6. Notes and limitations

* Datasets are not included: Oxford-IIIT Pet is downloaded by `scripts/make_pets_split.py`; FS2K from the authors'
  repository (see `KAGGLE_GUIDE.md`).
* The Task 3 mini-batches draw the corruption class uniformly (balanced in expectation, not exactly).
* Retraining checkpoints of the final specialists / mixture were not archived from the cloud session; the ONNX files
  (verified against PyTorch at export time) are the deliverable. See "Limitations and Deviations" in the report.
* AI assistants were used for code, design and drafting; the AI-use statement is Appendix A of the report.
