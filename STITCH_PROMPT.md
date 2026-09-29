# Google Stitch design prompt

The assignment requires the UI to be designed in **Google Stitch first**, with evidence in the report.
The React app in `frontend/` is already built with the structure below; do this in Stitch, then send me the
result (screenshots) and I will align the colours/layout of the frontend to it.

1. Open https://stitch.withgoogle.com , choose **Web**, paste the prompt below.
2. Generate, iterate 1-3 times (e.g. "make the sidebar darker", "add a dark mode").
3. **Screenshot the generated screens AND the prompt/history panel** (this is the "evidence of the original Stitch design"
   for the report) and export the design (Figma / code) if offered. Save into `stitch_design/`.

## Prompt
> Design a clean, modern web application called "Restoration Studio" for an AI image-processing course project.
> Layout: a left sidebar (collapses to a horizontal tab bar on mobile) with four workspaces: "Universal Restoration",
> "Hard-Routed Restoration", "Soft Mixture-of-Experts", "Face-to-Sketch Generator", each with a small task label and a green/amber
> status dot showing whether its models are loaded. At the bottom of the sidebar show backend info (status, runtime, device,
> models loaded count).
> Each restoration workspace has a two-column layout. Left column (fixed ~360px) contains stacked cards:
> "1 · Input image" with Upload / Sample / (Webcam) tabs and a dashed drop area plus thumbnail; "2 · Runtime corruption" with a
> dropdown (None, Salt-and-pepper, Gaussian blur, Rectangular occlusion), severity segmented buttons (Low, Medium, High, Custom),
> sliders that appear for Custom, and a random-seed field; for the hard-routing page a "Routing mode" toggle (Predicted / Oracle);
> then a full-width primary "Run model" button. Right column: a "Result" card with three square image panels
> (Model input, Restored output, Absolute error map) each with a download link; a "System information" card with stat tiles
> (Inference time, PSNR before → after, selected expert or dominant branch) and a corruption-settings line;
> for hard routing a "Classifier probabilities" card with four horizontal progress bars; for the soft mixture page a
> "Routing weights" card with four horizontal bars and a stacked colour strip showing each expert's contribution.
> The Face-to-Sketch page has a photo input (upload or webcam capture), three Style buttons (Style 1/2/3), a "Generate sketch" button,
> and a side-by-side comparison of the original photograph and the generated grayscale sketch with a download button and inference time.
> Style: light theme, indigo primary colour (#4F46E5), white rounded-2xl cards with subtle shadows on a very light slate background,
> Inter font, generous spacing, accessible contrast. Provide a matching dark theme variant.
