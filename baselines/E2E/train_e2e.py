"""
train_e2e.py
------------
Train end-to-end powerline -> text models with two objective families:

1) Direct envelope -> text (CTC head).
2) Envelope-conditioned decoder (autoregressive text head).

Both share the same U-Net envelope reconstruction backbone and can be trained
independently or jointly.
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
from model import BLANK_IDX, E2ELoss, E2EModel, IDX_TO_CHAR


def make_shifted_tokens(text):
    """
    Teacher-forcing input:
      in[:, 0]   = blank
      in[:, 1:]  = text[:, :-1]
    """
    dec_in = torch.zeros_like(text)
    dec_in[:, 1:] = text[:, :-1]
    return dec_in


def tokens_to_text(tokens):
    chars = []
    for t in tokens:
        if t == BLANK_IDX:
            continue
        chars.append(IDX_TO_CHAR.get(int(t), ""))
    return "".join(chars).strip()


def ctc_greedy_decode(logits_tbv):
    # logits_tbv: (T, B, V)
    pred_ids = torch.argmax(logits_tbv, dim=-1).transpose(0, 1)  # (B, T)
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


def seq_greedy_decode(lm_logits_blv):
    # lm_logits_blv: (B, L, V)
    pred_ids = torch.argmax(lm_logits_blv, dim=-1)
    return [tokens_to_text(row.tolist()) for row in pred_ids]


def edit_distance(a, b):
    # Character-level Levenshtein distance.
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
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
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


def train_one_epoch(model, loader, optimizer, loss_fn, device, use_amp, use_seq):
    model.train()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    totals = {"loss": 0.0, "l1": 0.0, "mstft": 0.0, "ctc": 0.0, "seq": 0.0}

    for noisy, clean, text, text_len in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        text_len = text_len.to(device, non_blocking=True)
        dec_in = make_shifted_tokens(text) if use_seq else None

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                pred, ctc_logits, lm_logits = model(noisy, dec_in)
                loss, l1, mstft, ctc, seq = loss_fn(
                    pred, clean, ctc_logits, text, text_len, lm_logits
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, ctc_logits, lm_logits = model(noisy, dec_in)
            loss, l1, mstft, ctc, seq = loss_fn(
                pred, clean, ctc_logits, text, text_len, lm_logits
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        totals["loss"] += float(loss.item())
        totals["l1"] += l1
        totals["mstft"] += mstft
        totals["ctc"] += ctc
        totals["seq"] += seq

    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp, use_ctc, use_seq, decode_batches=4):
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "mstft": 0.0, "ctc": 0.0, "seq": 0.0}
    ctc_cer_vals = []
    seq_cer_vals = []

    for bi, (noisy, clean, text, text_len) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        text_len = text_len.to(device, non_blocking=True)
        dec_in = make_shifted_tokens(text) if use_seq else None

        if use_amp:
            with torch.amp.autocast("cuda"):
                pred, ctc_logits, lm_logits = model(noisy, dec_in)
                loss, l1, mstft, ctc, seq = loss_fn(
                    pred, clean, ctc_logits, text, text_len, lm_logits
                )
        else:
            pred, ctc_logits, lm_logits = model(noisy, dec_in)
            loss, l1, mstft, ctc, seq = loss_fn(
                pred, clean, ctc_logits, text, text_len, lm_logits
            )

        totals["loss"] += float(loss.item())
        totals["l1"] += l1
        totals["mstft"] += mstft
        totals["ctc"] += ctc
        totals["seq"] += seq

        if bi < decode_batches:
            refs = batch_reference_text(text, text_len)
            if use_ctc:
                ctc_preds = ctc_greedy_decode(ctc_logits)
                ctc_cer_vals.append(avg_cer(ctc_preds, refs))
            if use_seq and lm_logits is not None:
                seq_preds = seq_greedy_decode(lm_logits)
                seq_cer_vals.append(avg_cer(seq_preds, refs))

    n = max(1, len(loader))
    out = {k: v / n for k, v in totals.items()}
    out["ctc_cer"] = float(np.mean(ctc_cer_vals)) if ctc_cer_vals else 0.0
    out["seq_cer"] = float(np.mean(seq_cer_vals)) if seq_cer_vals else 0.0
    return out


def save_curves(history, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["train_ctc"], label="Train CTC")
    axes[0, 1].plot(history["val_ctc"], label="Val CTC")
    axes[0, 1].plot(history["train_seq"], label="Train Seq")
    axes[0, 1].plot(history["val_seq"], label="Val Seq")
    axes[0, 1].set_title("Text Losses")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history["val_ctc_cer"], label="Val CTC CER")
    axes[1, 0].plot(history["val_seq_cer"], label="Val Seq CER")
    axes[1, 0].set_title("Validation CER")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].semilogy(history["lr"])
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close()


def configure_mode(mode, beta, gamma):
    if mode == "ctc":
        return True, False, beta, 0.0
    if mode == "conditioned":
        return False, True, 0.0, gamma
    return True, True, beta, gamma


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    device = pick_device(args.device)
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    train_loader, val_loader, _ = make_dataloaders(
        args.data,
        batch_size=args.batch,
        num_workers=args.num_workers,
    )

    use_ctc, use_seq, beta_w, gamma_w = configure_mode(args.mode, args.beta, args.gamma)
    print(
        f"Mode={args.mode} | use_ctc={use_ctc} use_seq={use_seq} "
        f"| weights: recon={args.alpha} ctc={beta_w} seq={gamma_w}"
    )

    model = E2EModel(
        base_ch=args.base_ch,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        dec_d_model=args.dec_d_model,
        dec_heads=args.dec_heads,
        dec_layers=args.dec_layers,
        dec_max_text_len=args.max_text_len,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f"Model params: {sum(p.numel() for p in raw_model.parameters()):,}")

    loss_fn = E2ELoss(alpha=args.alpha, beta=beta_w, gamma=gamma_w).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    os.makedirs(args.out, exist_ok=True)
    history = {
        "train_loss": [], "val_loss": [],
        "train_ctc": [], "val_ctc": [],
        "train_seq": [], "val_seq": [],
        "val_ctc_cer": [], "val_seq_cer": [],
        "lr": [],
    }
    best_val = float("inf")
    patience_ctr = 0

    print(
        f"{'Ep':>4}  {'Train':>9}  {'Val':>9}  {'CTC':>7}  {'Seq':>7}  "
        f"{'CTC-CER':>8}  {'SEQ-CER':>8}  {'LR':>9}  {'Time':>6}"
    )
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, use_seq)
        vl = evaluate(
            model, val_loader, loss_fn, device, use_amp,
            use_ctc=use_ctc, use_seq=use_seq, decode_batches=args.decode_batches
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_ctc"].append(tr["ctc"])
        history["val_ctc"].append(vl["ctc"])
        history["train_seq"].append(tr["seq"])
        history["val_seq"].append(vl["seq"])
        history["val_ctc_cer"].append(vl["ctc_cer"])
        history["val_seq_cer"].append(vl["seq_cer"])
        history["lr"].append(lr_now)

        print(
            f"{epoch:4d}  {tr['loss']:9.5f}  {vl['loss']:9.5f}  "
            f"{tr['ctc']:7.4f}  {tr['seq']:7.4f}  "
            f"{vl['ctc_cer']:8.4f}  {vl['seq_cer']:8.4f}  {lr_now:9.2e}  {dt:5.1f}s",
            flush=True,
        )

        if vl["loss"] < best_val:
            best_val = vl["loss"]
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "val_loss": best_val,
                    "args": vars(args),
                    "mode": args.mode,
                },
                os.path.join(args.out, "best_model.pt"),
            )
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "val_loss": vl["loss"],
                    "args": vars(args),
                    "mode": args.mode,
                },
                os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"),
            )

    torch.save(
        {
            "epoch": epoch,
            "model_state": raw_model.state_dict(),
            "val_loss": vl["loss"],
            "args": vars(args),
            "mode": args.mode,
        },
        os.path.join(args.out, "last_model.pt"),
    )
    if not args.no_plots:
        save_curves(history, args.out)

    print(f"Best val loss: {best_val:.6f}")
    print(f"Saved checkpoints to: {args.out}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Train end-to-end powerline->text models")
    parser.add_argument("--data", default=os.path.join(here, "data", "e2e_data.npz"))
    parser.add_argument("--out", default=os.path.join(here, "checkpoints_joint"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_plots", action="store_true")

    parser.add_argument("--mode", choices=["ctc", "conditioned", "joint"], default="joint")

    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--decode_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--alpha", type=float, default=1.0, help="Recon loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="CTC loss weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Conditioned CE weight")

    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--dec_d_model", type=int, default=256)
    parser.add_argument("--dec_heads", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=3)
    parser.add_argument("--max_text_len", type=int, default=256)

    args = parser.parse_args()
    main(args)
