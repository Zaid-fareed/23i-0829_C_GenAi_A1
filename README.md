# Generative AI - Assignment 1

Universal / hard-routed / soft-MoE image restoration on Oxford-IIIT Pets (Tasks 1-3) and style-conditioned
face-to-sketch cGAN on FS2K (Task 4), served by a FastAPI + React/Tailwind app in Docker.

```
src/            data pipeline, models, losses, training (Optuna + MLflow), evaluation, ONNX export
backend/        FastAPI service (ONNX Runtime inference, upload validation, corruption, tests)
frontend/       React + Tailwind (Vite) app, served by nginx in Docker
scripts/        split creation, Kaggle packing, dummy-model generator
KAGGLE_GUIDE.md training instructions   |   STITCH_PROMPT.md UI design prompt
```

## 1. Run the application (evaluator quick start)
Requires Docker Desktop.
```bash
git clone <REPO_URL> && cd genai-assignment1
# download the trained ONNX models (link: <MODELS_DOWNLOAD_URL>) into ./onnx_models :
#   task1_universal_dae.onnx  task2_classifier.onnx  task2_specialist_{salt_pepper,blur,occlusion}.onnx
#   task3_soft_moe.onnx       task4_generator.onnx
docker compose up --build
```
Open http://localhost:3000 (backend health check: http://localhost:3000/api/health).
Optional MLflow UI with the recorded experiments: `docker compose --profile tracking up` -> http://localhost:5000
(needs `mlflow.db` from training in the repo root).

Sample images offered in the app come from `backend/samples/` (replace with a few Oxford-Pets / face images).

## 2. Develop without Docker
```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt -r backend/requirements.txt
python scripts/make_dummy_onnx.py                     # random-weight models, only for UI/API development
MODEL_DIR=onnx_models_dummy PYTHONPATH=backend uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev             # http://localhost:5173 (proxies /api to :8000)
```
Tests: `pytest tests backend/tests` (API tests need the dummy models above).

## 3. Reproduce training
See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md). Order: split + manifests -> Task 1 -> Task 2 -> Task 3 -> Task 4, each
`optuna` -> `final` -> `eval` -> `export`. Trained ONNX files are not committed (see `.gitignore`); download link above.

*(README to be completed with results, Optuna summaries and the AI-use appendix reference.)*
