"""
train.py
--------
Train the PowerlineUNet to reconstruct clean audio spectrograms
from IQ-demodulated powerline measurements.

Works on both local CPU and the HPC cluster's GPU (A100/V100).
Automatically detects GPU and wraps in DataParallel if multiple GPUs found.

Usage:
    python train.py                                   # local defaults
    python train.py --data /fs/scratch/<allocation>/train_data/train_data.npz \
                    --out  /fs/scratch/<allocation>/checkpoints \
                    --epochs 300 --lr 3e-4 --batch 64
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
from model   import PowerlineUNet, SpectralLoss
from dataset import make_dataloaders

# ── default paths (overridden by --data / --out on HPC) ───────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA = os.path.join(_HERE, "data", "train_data.npz")
LOCAL_OUT  = os.path.join(_HERE, "checkpoints")


# ── training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, loss_fn, device, scaler=None):
    model.train()
    total_loss = total_l1 = total_sc = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        optimiser.zero_grad()

        if scaler is not None:                         # AMP (GPU only)
            with torch.amp.autocast('cuda'):
                pred = model(noisy)
                loss, l1, sc = loss_fn(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            pred = model(noisy)
            loss, l1, sc = loss_fn(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        total_loss += loss.item()
        total_l1   += l1
        total_sc   += sc

    n = len(loader)
    return total_loss/n, total_l1/n, total_sc/n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_l1 = total_sc = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
            pred = model(noisy)
        loss, l1, sc = loss_fn(pred, clean)
        total_loss += loss.item()
        total_l1   += l1
        total_sc   += sc
    n = len(loader)
    return total_loss/n, total_l1/n, total_sc/n


def save_sample_plot(model, val_loader, device, epoch, out_dir):
    """Save a 3-panel spectrogram comparison for the first val batch."""
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    with torch.no_grad():
        noisy, clean = next(iter(val_loader))
        noisy = noisy.to(device)
        pred  = raw_model(noisy)

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
    # ── device setup ─────────────────────────────────────────────────────────
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

    # ── data ─────────────────────────────────────────────────────────────────
    num_workers = min(4, os.cpu_count() or 1) if n_gpus > 0 else 0
    train_loader, val_loader, data = make_dataloaders(
        args.data, batch_size=args.batch, num_workers=num_workers)

    # ── model ─────────────────────────────────────────────────────────────────
    model = PowerlineUNet(base_ch=args.base_ch).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {n_gpus} GPUs")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f"Params : {raw_model.count_params():,}")

    loss_fn   = SpectralLoss(l1_weight=0.8, sc_weight=0.2)
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    # ── output dirs ───────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    sample_dir = os.path.join(args.out, "training_samples")

    history  = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')

    print(f"\nTraining {args.epochs} epochs | lr={args.lr} | batch={args.batch} | AMP={use_amp}\n")
    print(f"{'Epoch':>6}  {'Train':>9}  {'Val':>9}  {'LR':>9}  {'Time':>7}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = train_one_epoch(model, train_loader, optimiser,
                                        loss_fn, device, scaler)
        vl_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['lr'].append(lr_now)

        print(f"{epoch:6d}  {tr_loss:9.5f}  {vl_loss:9.5f}  "
              f"{lr_now:9.2e}  {elapsed:6.1f}s", flush=True)

        # Save best checkpoint
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                'epoch':       epoch,
                'model_state': raw_model.state_dict(),
                'val_loss':    best_val,
                'args':        vars(args),
            }, os.path.join(args.out, "best_model.pt"))

        # Periodic sample plots + checkpoint
        if epoch % args.plot_every == 0 or epoch == 1:
            save_sample_plot(model, val_loader, device, epoch, sample_dir)

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'model_state': raw_model.state_dict(),
                'val_loss': vl_loss, 'args': vars(args),
            }, os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"))

    # Save final checkpoint
    torch.save({
        'epoch': args.epochs, 'model_state': raw_model.state_dict(),
        'val_loss': vl_loss,  'args': vars(args),
    }, os.path.join(args.out, "last_model.pt"))

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'],   label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.semilogy(history['lr'])
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('LR (log scale)')
    ax2.set_title('Learning Rate'); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=130)
    plt.close()

    print(f"\nBest val loss : {best_val:.5f}")
    print(f"Checkpoints  -> {args.out}")
    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PowerlineUNet")

    # Paths — use HPC scratch paths when running on cluster
    parser.add_argument('--data',       default=LOCAL_DATA,
                        help='Path to train_data.npz')
    parser.add_argument('--out',        default=LOCAL_OUT,
                        help='Output directory for checkpoints and plots')

    # Hyperparameters
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch',      type=int,   default=64,
                        help='Batch size (use 64-128 on A100)')
    parser.add_argument('--base_ch',    type=int,   default=32,
                        help='U-Net base channels (32 or 64)')

    # Logging
    parser.add_argument('--plot_every', type=int,   default=10,
                        help='Save spectrogram sample every N epochs')
    parser.add_argument('--save_every', type=int,   default=50,
                        help='Save periodic checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
