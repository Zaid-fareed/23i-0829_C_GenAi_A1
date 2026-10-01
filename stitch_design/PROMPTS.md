# Google Stitch prompts (evidence of the design process)

Two prompting rounds were used. The prompts were drafted with the help of an AI assistant (see Appendix A of the report);
the author's own request that triggered round 2 is quoted verbatim.

## Round 1 - initial design
Exports: `stitch_restoration_studio_ai/` (four screens with `screen.png`, `code.html`, and `restoration_studio/DESIGN.md`).

Prompt: see `STITCH_PROMPT.md` in the repository root (four workspaces, sidebar with status dots, numbered cards for input
image / runtime corruption / routing mode, result panels with image, restored output and error map, system-information
tiles, classifier probability bars, soft routing weights, face-to-sketch page; indigo primary #4F46E5, Inter font,
white rounded cards on a light slate background, dark variant).

## Problem found with round 1
Author's own words: *"the front end is not very easy to use and makes it a bitter experience ... we don't have to scroll
down to choose option and then scroll up to see results."*

## Round 2 - layout revision (follow-up prompt)
Exports: `v2/stitch_restoration_studio_ai/` (`*_2` folders are the revised screens; the sketch screen's image came out
empty, its HTML is present).

Prompt (sent as a follow-up on the existing design):

> The design looks good, but it's hard to use because I have to scroll down to choose options and press Run, then scroll back
> up to see the result. Please change the layout of all four screens so everything for one run fits on a single laptop
> screen without scrolling. Keep the controls in a narrower panel on the left, but make the image picker smaller with just a
> thumbnail, and move the custom sliders and random seed into a collapsed "More options" section. Pin the Run button at the
> bottom of that left panel so it is always visible. Move the results to the right and make them larger, with the before and
> after images side by side and the error map as a smaller third image, and shrink the info tiles into one compact row
> underneath. On the hard-routing screen put the four probability bars beside the images instead of below them, and do the
> same for the expert weights on the soft mixture screen. On the face-to-sketch screen show the photo and the sketch side by
> side, with the Download button next to them. Also remove the fake GPU stats, the recent runs card and the breadcrumbs, but
> keep the colours, fonts and cards as they are.

## How the designs were used
The exported HTML is a static mock-up with invented data (GPU model, LPIPS, stroke sliders). Its structure, colours,
typography and layout were re-implemented as React components (`frontend/src/`) that show only real data. The final layout
was verified in a real browser (`scripts/take_screenshots.py`): at 1366x768 and 1920x1080 the Run button is always visible
and the page does not scroll.

> **Author to add:** screenshots of the Stitch project page / prompt history for both rounds (place them in this folder
> and reference them in the report).
