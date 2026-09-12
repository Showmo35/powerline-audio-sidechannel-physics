"""
train_cascade.py
----------------
Two-phase training for the cascade model (pretrained UNet -> CTC encoder).

Phase 1: UNet frozen, only CTC encoder trains.
          Loss = CTC only (alpha=0).

Phase 2: UNet unfrozen, both train end-to-end with differential LR.
          Loss = CTC + reconstruction (alpha > 0).
          UNet LR = main LR * unet_lr_factor (default 0.1x).
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
from model import (BLANK_IDX, CascadeLoss, CascadeModel, IDX_TO_CHAR,
                    load_pretrained_unet)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def tokens_to_text(tokens):
    chars = []
    for t in tokens:
        if t == BLANK_IDX:
            continue
        chars.append(IDX_TO_CHAR.get(int(t), ""))
    return "".join(chars).strip()


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
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)
    return int(dp[n, m])


def batch_reference_text(text, text_len):
    refs = []
    for i in range(text.shape[0]):
        tl = int(text_len[i].item())
        refs.append(tokens_to_text(text[i, :tl].tolist()))
    return refs


def avg_cer(preds, refs):
    if not refs:
        return 0.0
    total_dist = 0
    total_chars = 0
    for p, r in zip(preds, refs):
        total_dist += edit_distance(p, r)
        total_chars += max(1, len(r))
    return total_dist / total_chars


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Training loops ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device, use_amp):
    model.train()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    totals = {"loss": 0.0, "ctc": 0.0, "l1": 0.0, "mstft": 0.0}

    for noisy, clean, text, text_len in loader:
        noisy    = noisy.to(device, non_blocking=True)
        clean    = clean.to(device, non_blocking=True)
        text     = text.to(device, non_blocking=True)
        text_len = text_len.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                pred, ctc_logits = model(noisy)
                loss, ctc, l1, mstft = loss_fn(pred, clean, ctc_logits,
                                                text, text_len)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, ctc_logits = model(noisy)
            loss, ctc, l1, mstft = loss_fn(pred, clean, ctc_logits,
                                            text, text_len)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        totals["loss"]  += loss.item()
        totals["ctc"]   += ctc
        totals["l1"]    += l1
        totals["mstft"] += mstft

    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp, decode_batches=4):
    model.eval()
    totals = {"loss": 0.0, "ctc": 0.0, "l1": 0.0, "mstft": 0.0}
    cer_vals = []

    for bi, (noisy, clean, text, text_len) in enumerate(loader):
        noisy    = noisy.to(device, non_blocking=True)
        clean    = clean.to(device, non_blocking=True)
        text     = text.to(device, non_blocking=True)
        text_len = text_len.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                pred, ctc_logits = model(noisy)
                loss, ctc, l1, mstft = loss_fn(pred, clean, ctc_logits,
                                                text, text_len)
        else:
            pred, ctc_logits = model(noisy)
            loss, ctc, l1, mstft = loss_fn(pred, clean, ctc_logits,
                                            text, text_len)

        totals["loss"]  += loss.item()
        totals["ctc"]   += ctc
        totals["l1"]    += l1
        totals["mstft"] += mstft

        if bi < decode_batches:
            refs  = batch_reference_text(text, text_len)
            preds = ctc_greedy_decode(ctc_logits)
            cer_vals.append(avg_cer(preds, refs))

    n = max(1, len(loader))
    out = {k: v / n for k, v in totals.items()}
    out["cer"] = float(np.mean(cer_vals)) if cer_vals else 0.0
    return out


# ─── Plotting ────────────────────────────────────────────────────────────────

def save_curves(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    if history.get("phase2_start"):
        axes[0].axvline(x=history["phase2_start"], color='r', linestyle='--',
                        label="Phase 2 start", alpha=0.7)
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_cer"], label="Val CER")
    if history.get("phase2_start"):
        axes[1].axvline(x=history["phase2_start"], color='r', linestyle='--',
                        alpha=0.7)
    axes[1].set_title("Validation CER")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(history["lr"])
    if history.get("phase2_start"):
        axes[2].axvline(x=history["phase2_start"], color='r', linestyle='--',
                        alpha=0.7)
    axes[2].set_title("Learning Rate")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def run_phase(phase_name, model, train_loader, val_loader, optimizer,
              scheduler, loss_fn, device, use_amp, epochs, patience,
              out_dir, history, decode_batches, save_every, best_val):
    patience_ctr = 0

    print(f"\n{'='*70}")
    print(f"  {phase_name}")
    print(f"{'='*70}")
    print(f"{'Ep':>4}  {'Train':>9}  {'Val':>9}  {'CTC':>7}  "
          f"{'CER':>8}  {'LR':>9}  {'Time':>6}")
    print("-" * 65)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        global_ep = len(history["train_loss"])

        tr = train_one_epoch(model, train_loader, optimizer, loss_fn,
                             device, use_amp)
        vl = evaluate(model, val_loader, loss_fn, device, use_amp,
                      decode_batches)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["val_cer"].append(vl["cer"])
        history["lr"].append(lr_now)

        print(f"{ep:4d}  {tr['loss']:9.5f}  {vl['loss']:9.5f}  "
              f"{vl['ctc']:7.4f}  {vl['cer']:8.4f}  "
              f"{lr_now:9.2e}  {dt:5.1f}s", flush=True)

        raw_model = (model.module if isinstance(model, nn.DataParallel)
                     else model)

        if vl["loss"] < best_val:
            best_val = vl["loss"]
            patience_ctr = 0
            torch.save({
                "epoch": global_ep,
                "phase": phase_name,
                "model_state": raw_model.state_dict(),
                "val_loss": best_val,
                "val_cer": vl["cer"],
            }, os.path.join(out_dir, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {ep} "
                      f"(patience={patience}).")
                break

        if ep % save_every == 0:
            torch.save({
                "epoch": global_ep,
                "phase": phase_name,
                "model_state": raw_model.state_dict(),
                "val_loss": vl["loss"],
                "val_cer": vl["cer"],
            }, os.path.join(out_dir, f"ckpt_epoch_{global_ep:04d}.pt"))

    return best_val


def main(args):
    device  = pick_device(args.device)
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    train_loader, val_loader, _ = make_dataloaders(
        args.data, batch_size=args.batch, num_workers=args.num_workers)

    model = CascadeModel(
        unet_base_ch=args.unet_base_ch,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)

    # Load pretrained UNet
    load_pretrained_unet(model, args.unet_ckpt, device=device)

    raw_model = model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")
        raw_model = model.module

    n_unet = sum(p.numel() for p in raw_model.unet.parameters())
    n_ctc  = sum(p.numel() for p in raw_model.encoder.parameters())
    print(f"UNet params: {n_unet:,} | CTC params: {n_ctc:,} | "
          f"Total: {n_unet + n_ctc:,}")

    os.makedirs(args.out, exist_ok=True)
    history = {
        "train_loss": [], "val_loss": [], "val_cer": [], "lr": [],
        "phase2_start": None,
    }
    best_val = float("inf")

    # ── Phase 1: Freeze UNet, train CTC only ────────────────────────────
    if args.phase1_epochs > 0:
        raw_model.freeze_unet()
        print(f"\nPhase 1: UNet FROZEN | Training CTC encoder only")

        loss_fn_p1 = CascadeLoss(alpha=0.0, beta=args.beta).to(device)
        optimizer_p1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
        scheduler_p1 = CosineAnnealingLR(
            optimizer_p1, T_max=args.phase1_epochs, eta_min=args.min_lr)

        best_val = run_phase(
            "Phase 1 (UNet frozen)", model, train_loader, val_loader,
            optimizer_p1, scheduler_p1, loss_fn_p1, device, use_amp,
            args.phase1_epochs, args.patience, args.out, history,
            args.decode_batches, args.save_every, best_val)

    # ── Phase 2: Unfreeze UNet, fine-tune both ──────────────────────────
    if args.phase2_epochs > 0:
        history["phase2_start"] = len(history["train_loss"])
        raw_model.unfreeze_unet()
        print(f"\nPhase 2: UNet UNFROZEN | Fine-tuning both "
              f"(UNet LR = {args.lr * args.unet_lr_factor:.1e})")

        loss_fn_p2 = CascadeLoss(alpha=args.alpha, beta=args.beta).to(device)

        # Differential learning rate: lower for UNet
        param_groups = [
            {"params": raw_model.unet.parameters(),
             "lr": args.lr * args.unet_lr_factor},
            {"params": raw_model.encoder.parameters(),
             "lr": args.lr},
        ]
        optimizer_p2 = optim.AdamW(
            param_groups, weight_decay=args.weight_decay)
        scheduler_p2 = CosineAnnealingLR(
            optimizer_p2, T_max=args.phase2_epochs, eta_min=args.min_lr)

        best_val = run_phase(
            "Phase 2 (fine-tune both)", model, train_loader, val_loader,
            optimizer_p2, scheduler_p2, loss_fn_p2, device, use_amp,
            args.phase2_epochs, args.patience, args.out, history,
            args.decode_batches, args.save_every, best_val)

    # ── Save final state ────────────────────────────────────────────────
    raw_model_final = (model.module if isinstance(model, nn.DataParallel)
                       else model)
    torch.save({
        "model_state": raw_model_final.state_dict(),
        "val_loss": best_val,
        "args": vars(args),
    }, os.path.join(args.out, "last_model.pt"))

    if not args.no_plots:
        save_curves(history, args.out)

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoints saved to: {args.out}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(os.path.dirname(here), "E2E")
    unet_dir = os.path.join(os.path.dirname(here), "UNet")

    parser = argparse.ArgumentParser(
        description="Train cascade: pretrained UNet -> CTC encoder")

    # Data & checkpoints
    parser.add_argument("--data",
        default=os.path.join(e2e_dir, "data", "e2e_data.npz"))
    parser.add_argument("--unet_ckpt",
        default=os.path.join(unet_dir, "checkpoints_bp", "best_model.pt"))
    parser.add_argument("--out",
        default=os.path.join(here, "checkpoints"))
    parser.add_argument("--device",
        choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_plots", action="store_true")

    # Phase 1: frozen UNet
    parser.add_argument("--phase1_epochs", type=int, default=100)
    # Phase 2: fine-tune both
    parser.add_argument("--phase2_epochs", type=int, default=60)

    # Loss weights
    parser.add_argument("--alpha", type=float, default=0.5,
        help="Reconstruction loss weight (phase 2 only)")
    parser.add_argument("--beta", type=float, default=1.0,
        help="CTC loss weight")

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--unet_lr_factor", type=float, default=0.1,
        help="UNet LR = lr * this factor in phase 2")

    # Architecture
    parser.add_argument("--unet_base_ch", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--decode_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)
