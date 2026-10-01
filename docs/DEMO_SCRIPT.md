# Demonstration video script (target 6:30, allowed 5-7 minutes)

The assignment requires the video to show: **application start-up, image upload, runtime corruption, universal
restoration, hard routing, soft expert weights, face-to-sketch generation, result downloading, experiment-tracking
records.** Record the screen with the browser at full width (1366x768 or larger); speak naturally, the lines below are guides.

**Before recording:** open Docker Desktop; make sure `onnx_models/` holds the 7 files; close the app if it is running;
have a terminal open in the repository folder, a browser window ready, and `mlflow.db` in place.

| Time | What you do on screen | What you say |
|---|---|---|
| 0:00 | Title slide or the README top | "This is my Generative AI Assignment 1: image restoration with three architectures and a style-conditioned face-to-sketch GAN, deployed as a Dockerised web app." |
| 0:20 | Terminal: `docker compose up --build` (cut/speed up the build if long), show both containers healthy; open http://localhost:3000 | "One command starts everything. The back end loads seven ONNX models; the front end was designed in Google Stitch and built with React and Tailwind." Point at the sidebar: all four workspaces show green. |
| 1:00 | **Universal Restoration**: Sample tab, click a pet image; corruption = salt-and-pepper, severity High; press **Run model** | "I pick a test image. The corruption is applied at runtime from a seed, so the same seed gives the same result. The model is not told which corruption was used." Show input / restored / error map, time (ms), PSNR before -> after, "Copy JSON". |
| 1:50 | Same page: switch to Gaussian blur High, Run; then occlusion High, Run; then **Upload** an already-corrupted image with corruption "None" | "Three corruption types and severities, or upload an image that is already damaged." Mention: restores noise well, helps heavy occlusion, does not help mild blur. |
| 2:40 | **Hard-Routed Restoration**: salt-and-pepper Medium, Run; point at the four probabilities, the Predicted badge, selected expert; toggle **Oracle** and Run | "A classifier first predicts the corruption (99.4 % accurate), then exactly one specialist restores. Oracle mode uses the true label." Then choose corruption **None** with a clean pet image, Run: "A clean image goes through the identity bypass." |
| 3:40 | **Soft Mixture-of-Experts**: occlusion Medium, Run; point at the four weights, the stacked bar, Dominant badge, entropy; then a clean image or low-severity case | "The gate gives every branch a continuous weight. Here the occlusion expert dominates; for a clean or mild image the weights spread and the identity branch is used." |
| 4:30 | **Face-to-Sketch Generator**: Sample tab, a face; Style 1 -> Generate; Style 2 -> Generate; Style 3 -> Generate; try Webcam briefly (optional); press **Download sketch (PNG)** | "A U-Net generator with a learned style embedding, trained on FS2K. The same photo in three styles; style 2 is the heaviest. The result can be downloaded." Show the downloaded file in the folder. |
| 5:20 | Also click **↓ PNG** on an image of a restoration workspace | "Every result image can be downloaded." |
| 5:30 | Open http://localhost:5000 (`docker compose --profile tracking up`), click **Model training**, open `task3_soft_moe` -> a final run -> Model metrics tab; show `task2_specialists` runs list | "Every training run, Optuna trial, loss curve and the fixed validation sample grids are tracked with MLflow; Optuna studies are in the repository." |
| 6:10 | Show the report PDF title page / repo, `onnx_models` | "ONNX outputs were verified against PyTorch, all code, Dockerfiles, Optuna studies and the report are in the GitHub repository." |
| 6:30 | End | "Thank you." |

Tips
* Do one dry run first; keep each click deliberate so the viewer can read the numbers.
* If the Docker build is slow, record the start-up once, then trim the waiting time when editing.
* Say the limitation honestly once: the universal model degrades clean images, which is why routing and the mixture exist.
* The webcam needs the browser's camera permission; skip it if it fails (upload works for everything).
