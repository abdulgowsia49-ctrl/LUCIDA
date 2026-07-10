"""
CLI entrypoint: enhance a raw OHRC .img/.xml pair (or a plain .npy/.png
patch) into a denoised, contrast-recovered PNG.

Usage:
    python scripts/enhance.py --input path/to/label.xml --output out.png \
        --model checkpoints/phase2_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.pds4_loader import load_array, normalize_dn
from src.inference.pipeline import TiledInferencer
from src.model.curve_net import LiteCurveNet


def load_input(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".xml":
        raw = load_array(p)
        gray = normalize_dn(raw)
        return np.stack([gray, gray, gray], axis=-1) if gray.ndim == 2 else gray
    if p.suffix.lower() == ".npy":
        return np.load(p).astype(np.float32)
    # fall back to a regular image file
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return img


def main():
    parser = argparse.ArgumentParser(description="Enhance a lunar PSR patch with LUCIDA")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="path to a trained checkpoint (.pt)")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image = load_input(args.input)

    model = LiteCurveNet()
    model.load_state_dict(torch.load(args.model, map_location=args.device))

    inferencer = TiledInferencer(model, device=args.device, tile_size=args.tile_size, overlap=args.overlap)
    result = inferencer.run(image, apply_postprocess=not args.no_postprocess)

    out_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out_u8).save(args.output)
    print(f"saved enhanced output to {args.output}")


if __name__ == "__main__":
    main()
