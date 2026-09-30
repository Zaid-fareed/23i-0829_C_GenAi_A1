---
name: Restoration Studio
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#464555'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#777587'
  outline-variant: '#c7c4d8'
  surface-tint: '#4d44e3'
  primary: '#3525cd'
  on-primary: '#ffffff'
  primary-container: '#4f46e5'
  on-primary-container: '#dad7ff'
  inverse-primary: '#c3c0ff'
  secondary: '#4648d4'
  on-secondary: '#ffffff'
  secondary-container: '#6063ee'
  on-secondary-container: '#fffbff'
  tertiary: '#00505f'
  on-tertiary: '#ffffff'
  tertiary-container: '#006a7c'
  on-tertiary-container: '#93e8ff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#0f0069'
  on-primary-fixed-variant: '#3323cc'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#acedff'
  tertiary-fixed-dim: '#4cd7f6'
  on-tertiary-fixed: '#001f26'
  on-tertiary-fixed-variant: '#004e5c'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.025em
  headline-xl-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: -0.015em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  label-code-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-code-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
  label-code-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-sm: 1rem
  margin: 2rem
  margin-sm: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style

This design system embodies high-precision computational imaging, scientific rigor, and focused studio ergonomics. Tailored for computer vision researchers, ML engineers, and advanced visual specialists, the aesthetic balances the exactitude of a scientific lab with the clarity of contemporary creative software.

The aesthetic direction is **Modern Precision Minimalist**: 
- **Purity and Precision:** Clean, hyper-structured layouts avoid decorative noise, letting raw tensor diagnostics, artifacts, and visual restorations remain the primary focal points.
- **Instrument-grade Typography:** Direct pairing of utilitarian sans-serif typography with calibrated monospace figures for quantitative confidence.
- **Controlled Tactility:** Restrained surfaces, crisp sub-pixel bordering, and functional chromatic signals that deliver immediate status recognition without visual fatigue during prolonged research sessions.

## Colors

The palette is engineered for precision data density, clear threshold delineation, and high-fidelity image evaluation against calibrated neutral fields.

### Functional Palette Structure
- **Primary Indigo (`#4F46E5`) & Accent Violet (`#6366F1`):** Applied to active run executions, confirmed inference states, primary CTAs, and active slider thumbs.
- **Diagnostic Cyan (`#06B6D4`):** Dedicated to quantitative evaluation metrics, SSIM/PSNR visual overlays, and tensor inspector nodes.
- **Operational Status Spectrum:**
  - **Emerald (`#10B981`):** Healthy inference node, complete restoration, optimal convergence.
  - **Amber (`#F59E0B`):** GPU thermal throttling, warming kernels, queuing pipelines.
  - **Rose (`#EF4444`):** Divergent loss, degraded SSIM thresholds, NaN tensors, error masks.
- **Neutral Matrix:**
  - `slate-50` (`#F8FAFC`): Base workbench background.
  - `white` (`#FFFFFF`): Primary elevated card containers and data sheets.
  - `slate-200` (`#E2E8F0`): Structural separation lines and tool borders.
  - `slate-500` (`#64748B`): Secondary parameters, metadata, disabled anchors.
  - `slate-900` (`#0F172A`): Primary headings, critical diagnostics, active values.

## Typography

Typography enforces a strict hierarchy between human guidance and computational telemetry.

- **Proportional Type (Inter):** Leveraged across structural navigation, card headings, explanatory research notes, and primary actions. Inter's geometric clarity guarantees legibility at dense layout configurations.
- **Monospaced Data (JetBrains Mono):** Mandatory for all non-scalar entities: seeds, epoch increments, execution latencies, PSNR/SSIM numeric outputs, dimension matrices (e.g., `1x3x512x512`), and loss gradients. Tabular alignment ensures values never shift layout footprints during high-frequency streaming inferences.

## Layout & Spacing

The layout is built upon an asymmetric, dual-column research workbench designed for multi-tier comparison and real-time parameter tweaking.

- **Dual-Column Workbench Dynamic:**
  - **Primary Canvas (65-70% width):** Viewport container for split-screen comparisons, difference maps, and super-resolution zoom viewports.
  - **Telemetry & Parameter Rail (30-35% width):** High-density vertical stack containing checkpoint selection, loss function sweeps, metric panels, and tensor shapes.
- **Grid Adaptability:**
  - **Desktop (>= 1280px):** Fixed-variable dual column split with unified 24px gutters and outer 32px safe canvas margins.
  - **Tablet (768px - 1279px):** Fluid single column with collapsible telemetry panel accessible as a structured drawer; gutters compress to 16px.
  - **Mobile (< 768px):** Linear stacked feed; preview frame takes full aspect ratio with swipeable metric cards underneath.

## Elevation & Depth

Visual hierarchy relies on structural tonal separation supported by featherweight ambient shadowing rather than artificial depth cues.

- **Ground Layer:** Clean, untextured `slate-50` backdrop providing a quiet field for color-calibrated image outputs.
- **Surface Elevation (Cards & Panels):** Pure white (`#FFFFFF`) framed with a razor-thin 1px border of `slate-200`. Accompanied by an ambient shadow (`box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.03)`).
- **Interactive Floating Layers (Flyouts, Inspectors, Tooltips):** Higher elevation (`box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04)`) over layered canvas panels to maintain crisp layering during zoom and pan inspections.
- **Border Treatment:** High-stress interactive zones (dropzones, active image compare dividers) leverage a 1.5px boundary in `indigo-500` or `cyan-500` to indicate focus without relying on heavy outer drop shadows.

## Shapes

The design system maintains clean geometry that softens complex data environments without appearing playful.

- **Base Cards & Canvases:** `rounded-2xl` (1rem / 16px radius) produces a modern, framed container look for high-resolution graphics and modules.
- **Interior Controls & Inputs:** `rounded-lg` (0.5rem / 8px radius) ensures tight, predictable alignment across inline parameter fields, dropdown selectors, and button groups.
- **Pills & Live Indicators:** Fully rounded (`rounded-full`) to delineate continuous background tasks and discrete telemetry states from static rectilinear cards.

## Components

### Buttons & Trigger Controls
- **Primary:** Solid indigo background (`#4F46E5`), white label, smooth transitions on hover to `#4338CA`. Subtle ring focus indicator (`ring-2 ring-indigo-500/20`).
- **Secondary / Ghost:** White background, 1px `slate-200` border, `slate-700` label. Hovers transition to `slate-100`.
- **Run / Inference Action:** Features an embedded spinner or status dot next to the monospace runtime prediction.

### Segmented Controls
- Compact container with `slate-100` background, 4px inner padding, and `rounded-lg` geometry.
- Active segment translates smoothly on an elevated white tab with `shadow-sm` and high-contrast `slate-900` text; inactive segments display muted `slate-500`.

### Status Pills with Pulsing Indicators
- Embedded status indicators with a centered 6px dot and an animated radiating wave ring.
- **Active Node (Emerald):** Background `emerald-50`, text `emerald-700`, inner dot `emerald-500` with pulse.
- **Warming State (Amber):** Background `amber-50`, text `amber-700`, inner dot `amber-500` with steady glow.
- **Fault / Degradation (Rose):** Background `rose-50`, text `rose-700`, static alert dot.

### Image Comparison Frames
- Side-by-side or overlaid slider containers displaying degraded inputs against restored inferences.
- Equipped with a high-contrast hairline divider (`#FFFFFF` with `rgba(0,0,0,0.2)` edge shadows) and an interactive circular scrub handle featuring double-chevron icons. Monospace badge overlays in corners denote model variant and zoom scale (e.g., `400% [Bicubic vs ESRGAN-v2]`).

### Stacked Metric Bars
- Linear visual trackers for SSIM, PSNR, and LPIPS metrics.
- Built on a neutral track (`slate-100`, height 6px, `rounded-full`). Target threshold zones highlighted with subtle bracket ticks; the current score bar dynamically changes hue: cyan for nominal benchmarks, switching to emerald for targets exceeded, or rose if degraded beyond standard tolerances.

### Form Inputs & Numeric Fields
- Precision input fields with integrated monospace suffix labels (e.g., `dB`, `ms`, `px`).
- Inactive state: `slate-200` border, `slate-900` value; Focus state: `indigo-500` border with a 3px ambient `indigo-100` halo.