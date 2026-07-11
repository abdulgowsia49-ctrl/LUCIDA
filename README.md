# LUCIDA — Lightweight Zero-Reference Enhancement for Lunar PSRs

LUCIDA enhances Chandrayaan-2 OHRC imagery of the Moon's Permanently
Shadowed Regions (PSRs) — areas that haven't seen direct sunlight in
billions of years and are prime candidates for water ice — without needing
any paired ground-truth "bright" reference image.

## Why this exists

PSRs can't be photographed under normal light, so there's no ground truth
to train a supervised enhancement model on. The way around that,
established by zero-reference curve estimation, is to train a small network
to predict a per-pixel *tone curve* rather than a direct pixel mapping, and
supervise it with loss functions that describe what a "good" enhancement
should look like structurally (preserve edges, target a reasonable exposure,
stay smooth, don't shift color) instead of comparing to a reference image.

LUCIDA is an independent implementation of that idea, built for cheap
inference on large lunar tiles:

- **`LiteCurveNet`** (`src/model/curve_net.py`) — a curve-estimation CNN
  built from depthwise-separable convolutions instead of full convs, and a
  single shared curve-parameter head applied iteratively instead of a
  distinct head per iteration. Both cut parameter count and FLOPs
  substantially versus a standard DCE-style network at the same width —
  see the note on benchmarking below before quoting a specific number.
- **`LucidaLoss`** (`src/losses/losses.py`) — spatial consistency, exposure
  control, illumination smoothness, and color constancy losses, retuned for
  the lower exposure targets and higher noise floor of real PSR patches.
- **Classical denoising, not generative** (`src/preprocess/denoise.py`) —
  Non-Local Means + CLAHE post-processing, so the pipeline never invents
  detail that isn't in the sensor data.
- **Tiled inference with cosine blending** (`src/inference/pipeline.py`) —
  overlapping tiles blended with a raised-cosine window to avoid seams on
  full OHRC strips.
- **PDS4 loader** (`src/data/pds4_loader.py`) — minimal, dependency-light
  parser for Chandrayaan-2 PDS4 labels + raw `.img` arrays.

## Two-phase training

1. **Synthetic adaptation** — sunlit patches are synthetically darkened and
   given mixed Poisson-Gaussian noise (`src/preprocess/psr_mask.py`), so the
   model learns basic curve estimation before facing real PSR statistics.
2. **Zero-reference fine-tuning** — the Phase-1 checkpoint is fine-tuned
   directly on real PSR patches with the same reference-free loss.

```bash
python scripts/train.py --phase 1 --data data/synthetic_patches --epochs 30
python scripts/train.py --phase 2 --data data/psr_patches --epochs 15 \
    --resume checkpoints/phase1_best.pt
```

## Inference

```bash
python scripts/enhance.py --input path/to/label.xml --output enhanced.png \
    --model checkpoints/phase2_best.pt
```

## Installation

```bash
git clone https://github.com/<your-username>/lucida.git
cd lucida
pip install -r requirements.txt
```

## Prior art & citations

LUCIDA's curve-estimation approach and reference-free loss design are built
on ideas introduced in:

- Guo, C. et al. **"Zero-Reference Deep Curve Estimation for Low-Light
  Image Enhancement."** CVPR 2020.
- Bickel, V. et al. **"Peering into lunar permanently shadowed regions with
  deep learning."** Nature Communications, 2021.
- Chandrayaan-2 Mission Data Handbook, ISRO.

No code from any existing repository was copied into this project — the
above are cited as the algorithmic prior art this implementation builds on,
consistent with standard research practice.

## Status / honest caveats

- The efficiency claims above (parameter count, FLOPs) are architectural,
  from the design choices, not from a benchmark run yet. Run
  `python -m src.model.curve_net` to print live parameter counts, and
  benchmark inference time against a full-conv baseline before citing a
  specific speedup number anywhere public (README, hackathon deck, paper).
- No pretrained weights are included yet — `checkpoints/` is empty until
  you run Phase 1 + Phase 2 training on real patch data.
- `data/`, `notebooks/`, and `tests/` are placeholders for you to fill in
  as you go — this is a working skeleton, not a finished, validated
  pipeline.


