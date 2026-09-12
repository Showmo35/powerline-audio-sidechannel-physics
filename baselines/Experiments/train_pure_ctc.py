"""
train_pure_ctc.py
-----------------
Experiment 1: Train CTC encoder directly on raw noisy spectrograms.
No U-Net denoiser at all.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_dataloaders
from model import BLANK_IDX, IDX_TO_CHAR, PureCTCEncoder, PureCTCLoss


def tokens_to_text(tokens):
    return "".join(IDX_TO_CHAR.get(int(t), "") for t in tokens
                   if t != BLANK_IDX).strip()


def ctc_greedy_decode(logits_tbv):
    pred_ids = torch.argmax(logits_tbv, dim=-1).transpose(0, 1)
    texts = []
    for seq in pred_ids:
        collapsed = []
        prev = None
        for t in seq.tolist():
            if t == prev:
                continue
            prev = t
            if t != BLANK_IDX:
                collapsed.append(t)
        texts.append(tokens_to_text(collapsed))
    return texts


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)
    return int(dp[n, m])


def avg_cer(preds, refs):
    if not refs: return 0.0
    total_dist = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    total_chars = sum(max(1, len(r)) for r in refs)
    return total_dist / total_chars


def batch_reference_text(text, text_len):
    return [tokens_to_text(text[i, :int(text_len[i])].tolist())
            for i in range(text.shape[0])]


def pick_device(d):
    if d == "cpu": return torch.device("cpu")
    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, loss_fn, device, use_amp):
    model.train()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    total_loss = 0.0

    for noisy, clean, text, text_len in loader:
        noisy    = noisy.to(device, non_blocking=True)
        text     = text.to(device, non_blocking=True)
        text_len = text_len.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(noisy)
                loss = loss_fn(logits, text, text_len)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(noisy)
            loss = loss_fn(logits, text, text_len)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp, decode_batches=4):
    model.eval()
    total_loss = 0.0
    cer_vals = []

    for bi, (noisy, clean, text, text_len) in enumerate(loader):
        noisy    = noisy.to(device, non_blocking=True)
        text     = text.to(device, non_blocking=True)
        text_len = text_len.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(noisy)
                loss = loss_fn(logits, text, text_len)
        else:
            logits = model(noisy)
            loss = loss_fn(logits, text, text_len)

        total_loss += loss.item()

        if bi < decode_batches:
            refs  = batch_reference_text(text, text_len)
            preds = ctc_greedy_decode(logits)
            cer_vals.append(avg_cer(preds, refs))

    out_loss = total_loss / max(1, len(loader))
    out_cer  = float(np.mean(cer_vals)) if cer_vals else 0.0
    return out_loss, out_cer


def save_curves(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("CTC Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_cer"], label="Val CER")
    axes[1].set_title("Validation CER"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].semilogy(history["lr"])
    axes[2].set_title("Learning Rate"); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close()


def main(args):
    device  = pick_device(args.device)
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    train_loader, val_loader, _ = make_dataloaders(
        args.data, batch_size=args.batch, num_workers=args.num_workers)

    model = PureCTCEncoder(
        hidden=args.gru_hidden, n_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)

    raw_model = model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        raw_model = model.module

    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"PureCTC params: {n_params:,}")

    loss_fn   = PureCTCLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.min_lr)

    os.makedirs(args.out, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "lr": []}
    best_val = float("inf")
    patience_ctr = 0

    print(f"\n{'Ep':>4}  {'Train':>9}  {'Val':>9}  "
          f"{'CER':>8}  {'LR':>9}  {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                  device, use_amp)
        vl_loss, vl_cer = evaluate(model, val_loader, loss_fn, device,
                                   use_amp, args.decode_batches)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_cer"].append(vl_cer)
        history["lr"].append(lr_now)

        print(f"{epoch:4d}  {tr_loss:9.5f}  {vl_loss:9.5f}  "
              f"{vl_cer:8.4f}  {lr_now:9.2e}  {dt:5.1f}s", flush=True)

        if vl_loss < best_val:
            best_val = vl_loss
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "val_loss": best_val,
                "val_cer": vl_cer,
                "args": vars(args),
            }, os.path.join(args.out, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(patience={args.patience}).")
                break

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "val_loss": vl_loss,
                "val_cer": vl_cer,
                "args": vars(args),
            }, os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"))

    save_curves(history, args.out)
    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoints: {args.out}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(os.path.dirname(here), "E2E")

    parser = argparse.ArgumentParser(
        description="Exp 1: Pure CTC on raw noisy mel (no UNet)")
    parser.add_argument("--data",
        default=os.path.join(e2e_dir, "data", "e2e_data.npz"))
    parser.add_argument("--out",
        default=os.path.join(here, "exp1_pure_ctc"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
        default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--decode_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gru_hidden", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
