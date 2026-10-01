# Submission checklist - what only YOU can do

Everything else (code, models, report, Word version, figures, scripts, README) is done and committed locally.
Tick these in order. Items marked **[assignment]** are explicit requirements of the PDF.

## A. Check the report (about 1.5-2 h)
- [ ] Open `report/main.pdf` (or `report/Report_Assignment1.docx`) and read it once. The assignment says you may be asked to
      explain any part, so you must understand it. Sections 4-7 (the four tasks) matter most.
- [ ] **Author block** in `report/main.tex` (and `make_word.py`): name `Zaid Fareed`, roll `23I-0829`, section `C`. Change if wrong.
- [ ] **Three red `TO DO` markers** in the report:
  1. the YouTube link (title page, once the video is uploaded);
  2. the screenshots of your **Google Stitch project page / prompt history** [assignment: "evidence of the original Stitch design"];
  3. optional: your own MLflow screenshots (automatic ones are already included).
- [ ] **Appendix A (AI-use statement)** [assignment]: confirm every line is true; add the prompts *you* wrote.
- [ ] **References**: they were written from the papers' bibliographic data. Open at least the main ones (denoising
      autoencoders, SSIM, Zhao et al. loss functions, pix2pix, mixture of experts, Optuna, FS2K) and check authors, year,
      pages. The assignment requires you to have consulted real sources.
- [ ] Numbers: they are generated from the result files (no hand-typing). If you change anything, re-run
      `python report/build_tables.py` and recompile.

## B. GitHub [assignment]
- [ ] Create the release: GitHub -> repository -> Releases -> Draft a new release. Tag **`models-v1`**, title "Trained ONNX models",
      attach the 7 files from `onnx_models/` (`task1_universal_dae.onnx`, `task2_classifier.onnx`,
      `task2_specialist_salt_pepper.onnx`, `task2_specialist_blur.onnx`, `task2_specialist_occlusion.onnx`,
      `task3_soft_moe.onnx`, `task4_generator.onnx`), Publish.
- [ ] Push the code: `git push origin main` (all work is already committed locally).
- [ ] Test the evaluator path on a clean folder: `git clone ...`, `python scripts/download_models.py`, `docker compose up --build`.
      (The download only works once the repository is **public** or you are logged in; see next item.)
- [ ] Before submitting: make the repository **public** (or add the instructor as a collaborator) and open the repo and
      release links in a private/incognito window to confirm they load.
- [ ] README: paste the YouTube link at the top.

## C. Demo video [assignment: 5-7 minutes, YouTube, link only in the report]
- [ ] Follow `docs/DEMO_SCRIPT.md` (timed script covering every item the assignment lists).
- [ ] Upload to YouTube (Unlisted is fine), paste the link in `report/main.tex` (title page + README), recompile.
- [ ] Do **not** upload the video to Google Classroom.

## D. Submit [assignment]
- [ ] Compile the final PDF from LaTeX: upload `report/Report_Assignment1_latex.zip` to Overleaf (New Project -> Upload
      Project), main document `main.tex`, compiler pdfLaTeX, then download the PDF.
      (Or use `report/main.pdf`, built locally with the same sources.)
- [ ] Google Classroom: submit the **PDF report** (IEEE LaTeX). The repository link and the video link are inside it.
      Optionally also attach the Word version.
- [ ] **Check the deadline.** The assignment PDF says March 16, 2024, which looks like a typo; confirm the real date.

## E. Optional but recommended
- [ ] Run the application once yourself (`docker compose up --build`) and click through all four workspaces.
- [ ] Practise explaining: why the bottleneck hurts clean images (Task 1), why specialists are not better than the
      universal model (Task 2), what the gate temperature does (Task 3), why sketches are soft (Task 4).

## Known honest limitations (already stated in the report, Section 10)
Small Optuna searches; specialists not better than the universal model; SSIM vs PSNR disagree on occlusion; Task 3
batches balanced only in expectation; final checkpoints not archived (ONNX are); sketches are soft.
