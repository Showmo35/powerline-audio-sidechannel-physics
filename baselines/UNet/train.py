"""
train.py
--------
Train the PowerlineUNet to reconstruct clean Whisper-format mel spectrograms
from IQ-demodulated powerline measurements.

Loss = L1 + Multi-Scale STFT + Whisper Perceptual (frozen encoder features)

Works on both local CPU and the HPC cluster's GPU (A100/V100).

Usage:
    python train.py
    python train.py --data data/train_data.npz --out checkpoints \
                    --epochs 500 --lr 3e-4 --batch 32 --base_ch 64
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model   import PowerlineUNet, GeneratorLoss, WhisperPerceptualLoss
from dataset import make_dataloaders

_HERE     = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA = os.path.join(_HERE, "data", "train_data.npz")
LOCAL_OUT  = os.path.join(_HERE, "checkpoints")


# ── training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(gen, loader, optimizer, loss_fn, device, scaler=None):
    gen.train()
    totals = {'loss': 0, 'l1': 0, 'mstft': 0, 'percep': 0}

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
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

        totals['loss']   += loss.item()
        totals['l1']     += l1
        totals['mstft']  += mstft
        totals['percep'] += percep

    n = len(loader)
    return {k: v/n for k, v in totals.items()}


@torch.no_grad()
def evaluate(gen, loader, loss_fn, device):
    gen.eval()
    totals = {'loss': 0, 'l1': 0, 'mstft': 0, 'percep': 0}

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                pred = gen(noisy)
                loss, l1, mstft, percep = loss_fn(pred, clean)
        else:
            pred = gen(noisy)
            loss, l1, mstft, percep = loss_fn(pred, clean)

        totals['loss']   += loss.item()
        totals['l1']     += l1
        totals['mstft']  += mstft
        totals['percep'] += percep

    n = len(loader)
    return {k: v/n for k, v in totals.items()}


def save_sample_plot(gen, val_loader, device, epoch, out_dir):
    """Save a 3-panel spectrogram comparison for the first val batch."""
    raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
    raw_gen.eval()
    with torch.no_grad():
        noisy, clean = next(iter(val_loader))
        noisy = noisy.to(device)
        pred  = raw_gen(noisy)

    n = noisy[0, 0].cpu().numpy()
    c = clean[0, 0].numpy()
    p = pred[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(n.min(), c.min(), p.min())
    vmax = max(n.max(), c.max(), p.max())
    kw = dict(aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[0].imshow(n, **kw); axes[0].set_title('Noisy (Powerline IQ)')
    axes[1].imshow(c, **kw); axes[1].set_title('Clean (MP3 Ground Truth)')
    axes[2].imshow(p, **kw); axes[2].set_title(f'Predicted  epoch {epoch}')
    for ax in axes:
        ax.set_xlabel('Time frame'); ax.set_ylabel('Mel bin')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:04d}.png"), dpi=120)
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    # ── device setup ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device    = torch.device('cuda')
        n_gpus    = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        use_amp   = True
        print(f"GPU(s) : {n_gpus}x  {gpu_names}")
    else:
        device  = torch.device('cpu')
        n_gpus  = 0
        use_amp = False
        print("Device : CPU  (no GPU found)")

    # ── data ─────────────────────────────────────────────────────────────
    num_workers = min(4, os.cpu_count() or 1) if n_gpus > 0 else 0
    train_loader, val_loader, data = make_dataloaders(
        args.data, batch_size=args.batch, num_workers=num_workers)

    # ── model ────────────────────────────────────────────────────────────
    gen = PowerlineUNet(base_ch=args.base_ch).to(device)

    if n_gpus > 1:
        gen = nn.DataParallel(gen)
        print(f"DataParallel across {n_gpus} GPUs")

    raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
    print(f"Generator params : {raw_gen.count_params():,}")

    # ── loss ─────────────────────────────────────────────────────────────
    whisper_loss = None
    if args.percep_w > 0:
        print(f"Loading Whisper '{args.whisper_model}' for perceptual loss...")
        whisper_loss = WhisperPerceptualLoss(
            whisper_model_name=args.whisper_model, device=str(device))
        whisper_loss = whisper_loss.to(device)
        print(f"  Whisper perceptual loss ready (layers {whisper_loss.feature_layers})")

    loss_fn = GeneratorLoss(
        l1_weight=args.l1_w, mstft_weight=args.mstft_w,
        percep_weight=args.percep_w, whisper_loss=whisper_loss)

    optimizer = optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    # ── output dirs ──────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    sample_dir = os.path.join(args.out, "training_samples")

    history = {'loss': [], 'val_loss': [], 'l1': [], 'val_l1': [],
               'percep': [], 'val_percep': [], 'lr': []}
    best_val = float('inf')
    patience_counter = 0

    print(f"\nTraining {args.epochs} epochs | lr={args.lr} | batch={args.batch} | "
          f"base_ch={args.base_ch} | AMP={use_amp}")
    print(f"Loss weights: L1={args.l1_w}  MSTFT={args.mstft_w}  Percep={args.percep_w}")
    print(f"Early stopping patience: {args.patience}\n")
    print(f"{'Ep':>4}  {'Loss':>8}  {'Val_L':>8}  {'L1':>7}  {'MSTFT':>7}  "
          f"{'Percep':>7}  {'LR':>9}  {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(gen, train_loader, optimizer, loss_fn,
                             device, scaler)
        vl = evaluate(gen, val_loader, loss_fn, device)

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history['loss'].append(tr['loss'])
        history['val_loss'].append(vl['loss'])
        history['l1'].append(tr['l1'])
        history['val_l1'].append(vl['l1'])
        history['percep'].append(tr['percep'])
        history['val_percep'].append(vl['percep'])
        history['lr'].append(lr_now)

        print(f"{epoch:4d}  {tr['loss']:8.5f}  {vl['loss']:8.5f}  "
              f"{tr['l1']:7.4f}  {tr['mstft']:7.4f}  {tr['percep']:7.4f}  "
              f"{lr_now:9.2e}  {elapsed:5.1f}s", flush=True)

        # Save best checkpoint (based on val loss) + early stopping
        if vl['loss'] < best_val:
            best_val = vl['loss']
            patience_counter = 0
            torch.save({
                'epoch':       epoch,
                'model_state': raw_gen.state_dict(),
                'val_loss':    best_val,
                'args':        vars(args),
            }, os.path.join(args.out, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

        if epoch % args.plot_every == 0 or epoch == 1:
            save_sample_plot(gen, val_loader, device, epoch, sample_dir)

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state': raw_gen.state_dict(),
                'val_loss': vl['loss'],
                'args': vars(args),
            }, os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"))

    # Save final
    torch.save({
        'epoch': epoch,
        'model_state': raw_gen.state_dict(),
        'val_loss': vl['loss'],
        'args': vars(args),
    }, os.path.join(args.out, "last_model.pt"))

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0,0].plot(history['loss'],     label='Train')
    axes[0,0].plot(history['val_loss'], label='Val')
    axes[0,0].set_title('Total Loss'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(history['l1'],     label='Train L1')
    axes[0,1].plot(history['val_l1'], label='Val L1')
    axes[0,1].set_title('L1 Loss'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(history['percep'],     label='Train Percep')
    axes[1,0].plot(history['val_percep'], label='Val Percep')
    axes[1,0].set_title('Whisper Perceptual Loss'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    axes[1,1].semilogy(history['lr'])
    axes[1,1].set_title('Learning Rate'); axes[1,1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=130)
    plt.close()

    print(f"\nBest val loss : {best_val:.5f}")
    print(f"Checkpoints  -> {args.out}")
    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PowerlineUNet")

    parser.add_argument('--data',       default=LOCAL_DATA)
    parser.add_argument('--out',        default=LOCAL_OUT)

    # Hyperparameters
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch',      type=int,   default=32)
    parser.add_argument('--base_ch',    type=int,   default=64)

    # Loss weights
    parser.add_argument('--l1_w',       type=float, default=1.0, help='L1 loss weight')
    parser.add_argument('--mstft_w',    type=float, default=1.0, help='Multi-scale STFT loss weight')
    parser.add_argument('--percep_w',   type=float, default=1.0, help='Whisper perceptual loss weight')
    parser.add_argument('--whisper_model', default='base', help='Whisper model for perceptual loss')
    parser.add_argument('--patience',  type=int, default=30, help='Early stopping patience (epochs)')

    # Logging
    parser.add_argument('--plot_every', type=int,   default=10)
    parser.add_argument('--save_every', type=int,   default=50)

    args = parser.parse_args()
    main(args)
