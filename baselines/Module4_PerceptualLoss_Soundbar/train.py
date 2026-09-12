"""
Train wrapper for the Soundbar-adapted Module 4 pipeline.

This forwards to the original Module4_PerceptualLoss training script, but
uses the Soundbar-specific local data and checkpoint/output directories.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = "<REPO_ROOT>/baselines/Module4_PerceptualLoss/train.py"
DEFAULT_DATA = os.path.join(HERE, "data", "train_data.npz")
DEFAULT_OUT = os.path.join(HERE, "checkpoints")
DEFAULT_CTC = "<REPO_ROOT>/baselines/UNet/UNet-CTC/checkpoints/best_model.pt"


def main(args):
    cmd = [
        sys.executable,
        ORIG,
        "--data", args.data,
        "--out", args.out,
        "--ctc_ckpt", args.ctc_ckpt,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch", str(args.batch),
        "--base_ch", str(args.base_ch),
        "--l1_w", str(args.l1_w),
        "--mstft_w", str(args.mstft_w),
        "--percep_w", str(args.percep_w),
        "--patience", str(args.patience),
        "--plot_every", str(args.plot_every),
        "--save_every", str(args.save_every),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Soundbar Module 4 model")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--ctc_ckpt", default=DEFAULT_CTC)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--l1_w", type=float, default=1.0)
    parser.add_argument("--mstft_w", type=float, default=1.0)
    parser.add_argument("--percep_w", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--plot_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    main(parser.parse_args())
