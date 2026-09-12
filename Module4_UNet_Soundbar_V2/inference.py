"""
inference.py
------------
Evaluate the Soundbar UNet on the validation split.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_dataloaders
from model import PowerlineUNet

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "train_data.npz")
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "best_model.pt")
DEFAULT_OUT = os.path.join(HERE, "inference_output")


def compute_metrics(pred, clean):
    mse = np.mean((pred - clean) ** 2)
    mae = np.mean(np.abs(pred - clean))
    p_flat = pred.flatten()
    c_flat = clean.flatten()
    p_mean = p_flat.mean()
    c_mean = c_flat.mean()
    num = np.sum((p_flat - p_mean) * (c_flat - c_mean))
    den = np.sqrt(np.sum((p_flat - p_mean) ** 2) * np.sum((c_flat - c_mean) ** 2)) + 1e-12
    corr = num / den
    return mse, mae, corr


def plot_comparison(noisy, clean, pred, idx, metrics, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    vmin = min(noisy.min(), clean.min(), pred.min())
    vmax = max(noisy.max(), clean.max(), pred.max())
    kw = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")

    axes[0].imshow(noisy, **kw)
    axes[0].set_title("Noisy")
    axes[1].imshow(clean, **kw)
    axes[1].set_title("Clean")
    axes[2].imshow(pred, **kw)
    axes[2].set_title("UNet Prediction")
    for ax in axes:
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel bin")

    mse, mae, corr = metrics
    fig.suptitle(f"Sample {idx} | MSE={mse:.5f}  MAE={mae:.4f}  Corr={corr:.4f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparison_{idx:04d}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def to_jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    return obj


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {args.data} ...")
    _, val_loader, data = make_dataloaders(args.data, batch_size=args.batch, num_workers=args.workers)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = PowerlineUNet(base_ch=ckpt_args.get("base_ch", 64)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0):.5f})")
    print(f"Params: {model.count_params():,}")

    all_preds = []
    noisy_batches = []
    clean_batches = []

    print("\nRunning inference on validation set ...")
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy_device = noisy.to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(noisy_device)
            else:
                pred = model(noisy_device)
            all_preds.append(pred.cpu().numpy())
            noisy_batches.append(noisy.numpy())
            clean_batches.append(clean.numpy())

    predicted = np.concatenate(all_preds, axis=0)
    noisy_val = np.concatenate(noisy_batches, axis=0)
    clean_val = np.concatenate(clean_batches, axis=0)
    print(f"Predictions shape: {predicted.shape}")

    sample_metrics = []
    for i in range(len(predicted)):
        mse, mae, corr = compute_metrics(predicted[i], clean_val[i])
        sample_metrics.append({"idx": i, "mse": mse, "mae": mae, "corr": corr})

    mse_vals = [m["mse"] for m in sample_metrics]
    mae_vals = [m["mae"] for m in sample_metrics]
    corr_vals = [m["corr"] for m in sample_metrics]

    avg_mse = float(np.mean(mse_vals))
    avg_mae = float(np.mean(mae_vals))
    avg_corr = float(np.mean(corr_vals))
    std_mse = float(np.std(mse_vals))
    std_mae = float(np.std(mae_vals))
    std_corr = float(np.std(corr_vals))

    noisy_metrics = [compute_metrics(noisy_val[i], clean_val[i]) for i in range(len(noisy_val))]
    noisy_avg_mse = float(np.mean([m[0] for m in noisy_metrics]))
    noisy_avg_mae = float(np.mean([m[1] for m in noisy_metrics]))
    noisy_avg_corr = float(np.mean([m[2] for m in noisy_metrics]))

    print(f"\n{'=' * 60}")
    print(f"  Reconstruction Quality (val set, N={len(predicted)})")
    print(f"  MSE  : {avg_mse:.6f} +/- {std_mse:.6f}")
    print(f"  MAE  : {avg_mae:.5f} +/- {std_mae:.5f}")
    print(f"  Corr : {avg_corr:.4f} +/- {std_corr:.4f}")
    print(f"{'=' * 60}\n")
    print("  Baseline (noisy vs clean):")
    print(f"  MSE  : {noisy_avg_mse:.6f}")
    print(f"  MAE  : {noisy_avg_mae:.5f}")
    print(f"  Corr : {noisy_avg_corr:.4f}")
    print(f"{'=' * 60}\n")

    n_plot = min(args.n_samples, len(predicted))
    print(f"Saving {n_plot} comparison plots ...")
    for i in range(n_plot):
        plot_comparison(
            noisy_val[i, 0],
            clean_val[i, 0],
            predicted[i, 0],
            i,
            (sample_metrics[i]["mse"], sample_metrics[i]["mae"], sample_metrics[i]["corr"]),
            args.out,
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(mse_vals, bins=30, alpha=0.7, color="steelblue")
    axes[0].axvline(avg_mse, color="red", linestyle="--", label=f"Mean={avg_mse:.5f}")
    axes[0].set_title("MSE Distribution")
    axes[0].legend()

    axes[1].hist(mae_vals, bins=30, alpha=0.7, color="steelblue")
    axes[1].axvline(avg_mae, color="red", linestyle="--", label=f"Mean={avg_mae:.4f}")
    axes[1].set_title("MAE Distribution")
    axes[1].legend()

    axes[2].hist(corr_vals, bins=30, alpha=0.7, color="steelblue")
    axes[2].axvline(avg_corr, color="red", linestyle="--", label=f"Mean={avg_corr:.4f}")
    axes[2].set_title("Correlation Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "metrics_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "checkpoint": args.ckpt,
        "data": args.data,
        "n_val": len(predicted),
        "metrics": {
            "predicted": {"mse": avg_mse, "mae": avg_mae, "corr": avg_corr, "std_mse": std_mse, "std_mae": std_mae, "std_corr": std_corr},
            "noisy": {"mse": noisy_avg_mse, "mae": noisy_avg_mae, "corr": noisy_avg_corr},
        },
        "sample_metrics": sample_metrics[: min(len(sample_metrics), args.n_samples)],
    }

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)

    print(f"Results saved -> {os.path.join(args.out, 'results.json')}")
    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Soundbar UNet")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    main(parser.parse_args())