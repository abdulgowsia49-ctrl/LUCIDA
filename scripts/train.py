"""
Two-phase training driver for LiteCurveNet.

Phase 1 (synthetic adaptation): sunlit patches are synthetically darkened
(src/preprocess/psr_mask.synthetic_darken) and the model is trained with
the same reference-free LucidaLoss -- there's no paired ground truth used
even here, the synthetic darkening just gives the model a gentler, more
uniform noise floor to start learning curve estimation on before facing
real PSR statistics.

Phase 2 (zero-reference fine-tuning): the Phase-1 checkpoint is fine-tuned
directly on real PSR patches, same loss, lower learning rate.

Usage:
    python scripts/train.py --phase 1 --data data/synthetic_patches --epochs 30
    python scripts/train.py --phase 2 --data data/psr_patches --epochs 15 \
        --resume checkpoints/phase1_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.losses.losses import LucidaLoss
from src.model.curve_net import LiteCurveNet


class PatchDataset(Dataset):
    """Loads pre-extracted .npy patches (H, W, 3) float32 in [0, 1] from a
    directory. Patch extraction itself is a separate offline step."""

    def __init__(self, patch_dir: str | Path):
        self.paths = sorted(Path(patch_dir).glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy patches found in {patch_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = np.load(self.paths[idx]).astype(np.float32)
        return torch.from_numpy(patch).permute(2, 0, 1)


def train_one_phase(
    data_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    resume: str | None,
    out_path: str,
    device: str,
):
    model = LiteCurveNet().to(device)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"resumed weights from {resume}")

    loss_fn = LucidaLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = PatchDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            batch = batch.to(device)
            enhanced, alpha = model(batch)
            loss, parts = loss_fn(enhanced, batch, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg = running / len(loader)
        print(f"epoch {epoch:03d}/{epochs}  avg_loss={avg:.4f}  {parts}")

        if avg < best_loss:
            best_loss = avg
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved new best to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LiteCurveNet (LUCIDA)")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--data", type=str, required=True, help="directory of .npy patches")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    default_lr = 1e-4 if args.phase == 1 else 1e-5
    default_out = f"checkpoints/phase{args.phase}_best.pt"

    train_one_phase(
        data_dir=args.data,
        epochs=args.epochs,
        lr=args.lr or default_lr,
        batch_size=args.batch_size,
        resume=args.resume,
        out_path=args.out or default_out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
