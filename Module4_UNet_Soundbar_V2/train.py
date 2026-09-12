"""
train.py
--------
Train Soundbar UNet V2 with perception-aware loss (L1 + MSTFT + CTC perceptual).

Improvements over Module4_UNet_Soundbar:
  - CTC Perceptual Loss: frozen CTC encoder guides speech-discriminative features
  - base_ch=64 (full model capacity, ~18M params)
  - patience=30 (was 12) for more stable convergence
  - Training curves include the perceptual loss component
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_dataloaders
from model import PerceptionAwareLoss, PowerlineUNet

HERE       = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA = os.path.join(HERE, "data", "train_data.npz")
LOCAL_OUT  = os.path.join(HERE, "checkpoints")
CTC_CKPT   = os.path.join(HERE, "..", "UNet", "UNet-CTC", "checkpoints", "best_model.pt")


def train_one_epoch(gen, loader, optimizer, loss_fn, device, scaler=None):
    gen.train()
    totals = {"loss": 0.0, "l1": 0.0, "mstft": 0.0, "percep": 0.0}

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = gen(noisy)
                loss, l1, mstft, percep = loss_fn(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = gen(noisy)
            loss, l1, mstft, percep = loss_fn(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            optimizer.step()

        totals["loss"]   += loss.item()
        totals["l1"]     += l1
        totals["mstft"]  += mstft
        totals["percep"] += percep

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(gen, loader, loss_fn, device):
    gen.eval()
    totals = {"loss": 0.0, "l1": 0.0, "mstft": 0.0, "percep": 0.0}

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred = gen(noisy)
                loss, l1, mstft, percep = loss_fn(pred, clean)
        else:
            pred = gen(noisy)
            loss, l1, mstft, percep = loss_fn(pred, clean)

        totals["loss"]   += loss.item()
        totals["l1"]     += l1
        totals["mstft"]  += mstft
        totals["percep"] += percep

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def save_sample_plot(gen, val_loader, device, epoch, out_dir):
    raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
    raw_gen.eval()
    with torch.no_grad():
        noisy, clean = next(iter(val_loader))
        pred = raw_gen(noisy.to(device))

    n = noisy[0, 0].cpu().numpy()
    c = clean[0, 0].numpy()
    p = pred[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(n.min(), c.min(), p.min())
    vmax = max(n.max(), c.max(), p.max())
    kw = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
    axes[0].imshow(n, **kw); axes[0].set_title("Noisy Soundbar")
    axes[1].imshow(c, **kw); axes[1].set_title("Clean Audio")
    axes[2].imshow(p, **kw); axes[2].set_title(f"Predicted epoch {epoch}")
    for ax in axes:
        ax.set_xlabel("Time frame"); ax.set_ylabel("Mel bin")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:04d}.png"), dpi=120)
    plt.close()


def main(args):
    if torch.cuda.is_available():
        device    = torch.device("cuda")
        n_gpus    = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        use_amp   = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU(s) : {n_gpus}x  {gpu_names}")
    else:
        device  = torch.device("cpu")
        n_gpus  = 0
        use_amp = False
        print("Device : CPU  (no GPU found)")

    num_workers = args.workers if args.workers >= 0 else (min(8, os.cpu_count() or 1) if n_gpus > 0 else 0)
    pin_memory  = device.type == "cuda"
    train_loader, val_loader, data = make_dataloaders(
        args.data,
        batch_size=args.batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shard_ordered_batches=not args.disable_shard_batching,
    )

    gen = PowerlineUNet(base_ch=args.base_ch).to(device)
    if n_gpus > 1:
        gen = nn.DataParallel(gen)
        print(f"DataParallel across {n_gpus} GPUs")

    raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
    print(f"Generator params : {raw_gen.count_params():,}")

    loss_fn = PerceptionAwareLoss(
        ctc_ckpt_path=args.ctc_ckpt,
        l1_weight=args.l1_w,
        mstft_weight=args.mstft_w,
        percep_weight=args.percep_w,
    ).to(device)

    optimizer = optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    os.makedirs(args.out, exist_ok=True)
    sample_dir = os.path.join(args.out, "training_samples")

    history = {
        "loss": [], "val_loss": [],
        "l1": [], "val_l1": [],
        "mstft": [], "val_mstft": [],
        "percep": [], "val_percep": [],
        "lr": [],
    }
    best_val        = float("inf")
    patience_counter = 0

    print(
        f"\nTraining {args.epochs} epochs | lr={args.lr} | batch={args.batch} | "
        f"base_ch={args.base_ch} | AMP={use_amp}"
    )
    print(f"Loss weights: L1={args.l1_w}  MSTFT={args.mstft_w}  Percep={args.percep_w}")
    print(f"CTC checkpoint: {args.ctc_ckpt}")
    print(f"Early stopping patience: {args.patience}\n")
    print(f"{'Ep':>4}  {'Loss':>8}  {'Val_L':>8}  {'L1':>7}  {'MSTFT':>7}  "
          f"{'Percep':>7}  {'LR':>9}  {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(gen, train_loader, optimizer, loss_fn, device, scaler)
        vl = evaluate(gen, val_loader, loss_fn, device)

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history["loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["l1"].append(tr["l1"])
        history["val_l1"].append(vl["l1"])
        history["mstft"].append(tr["mstft"])
        history["val_mstft"].append(vl["mstft"])
        history["percep"].append(tr["percep"])
        history["val_percep"].append(vl["percep"])
        history["lr"].append(lr_now)

        print(
            f"{epoch:4d}  {tr['loss']:8.5f}  {vl['loss']:8.5f}  "
            f"{tr['l1']:7.4f}  {tr['mstft']:7.4f}  {tr['percep']:7.4f}  "
            f"{lr_now:9.2e}  {elapsed:5.1f}s",
            flush=True,
        )

        if vl["loss"] < best_val:
            best_val         = vl["loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": raw_gen.state_dict(),
                    "val_loss":    best_val,
                    "args":        vars(args),
                },
                os.path.join(args.out, "best_model.pt"),
            )
            print(f"  ** New best val_loss={best_val:.5f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

        if epoch % args.plot_every == 0 or epoch == 1:
            save_sample_plot(gen, val_loader, device, epoch, sample_dir)

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": raw_gen.state_dict(),
                    "val_loss":    vl["loss"],
                    "args":        vars(args),
                },
                os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"),
            )

    torch.save(
        {
            "epoch":       epoch,
            "model_state": raw_gen.state_dict(),
            "val_loss":    vl["loss"],
            "args":        vars(args),
        },
        os.path.join(args.out, "last_model.pt"),
    )

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["loss"],     label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Total Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["l1"],     label="Train L1")
    axes[0, 1].plot(history["val_l1"], label="Val L1")
    axes[0, 1].set_title("L1 Loss"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history["mstft"],     label="Train MSTFT")
    axes[1, 0].plot(history["val_mstft"], label="Val MSTFT")
    axes[1, 0].set_title("MSTFT Loss"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["percep"],     label="Train Perceptual")
    axes[1, 1].plot(history["val_percep"], label="Val Perceptual")
    axes[1, 1].set_title("Perceptual Loss (1 - cos_sim)")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=130)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogy(history["lr"])
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lr_curve.png"), dpi=130)
    plt.close()

    print(f"\nBest val loss : {best_val:.5f}")
    print(f"Checkpoints  -> {args.out}")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Soundbar UNet V2 with perceptual loss")

    parser.add_argument("--data",     default=LOCAL_DATA)
    parser.add_argument("--out",      default=LOCAL_OUT)
    parser.add_argument("--ctc_ckpt", default=CTC_CKPT,
                        help="Path to pretrained CTC encoder checkpoint")

    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--base_ch",  type=int,   default=64)

    parser.add_argument("--l1_w",     type=float, default=1.0)
    parser.add_argument("--mstft_w",  type=float, default=1.0)
    parser.add_argument("--percep_w", type=float, default=0.1)
    parser.add_argument("--patience", type=int,   default=30)

    parser.add_argument("--plot_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--workers",    type=int, default=-1)
    parser.add_argument(
        "--disable_shard_batching",
        action="store_true",
        help="Disable shard-local batching (kept for compatibility).",
    )

    main(parser.parse_args())
