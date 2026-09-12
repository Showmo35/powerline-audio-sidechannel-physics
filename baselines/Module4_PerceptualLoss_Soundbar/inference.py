"""
Inference wrapper for the Soundbar-adapted Module 4 pipeline.

This forwards to the original Module4_PerceptualLoss inference script, but
uses the Soundbar-specific local data and output directories.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = "<REPO_ROOT>/baselines/Module4_PerceptualLoss/inference.py"
DEFAULT_DATA = os.path.join(HERE, "data", "train_data.npz")
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "best_model.pt")
DEFAULT_OUT = os.path.join(HERE, "inference_output")


def main(args):
    cmd = [
        sys.executable,
        ORIG,
        "--data", args.data,
        "--ckpt", args.ckpt,
        "--out", args.out,
        "--batch", str(args.batch),
        "--n_samples", str(args.n_samples),
        "--device", args.device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Soundbar Module 4 model")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
