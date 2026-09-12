"""
train.py
--------
Train FullSubNet+ enhancement model on bandpass-filtered data.
L1 + Multi-Scale STFT loss.

Usage:
    python train.py --data data/train_data.npz --out checkpoints
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import EfficientFullSubNetEnhancer, GeneratorLoss
from dataset import make_dataloaders

_HERE      = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA = os.path.join(_HERE, "data", "train_data.npz")
LOCAL_OUT  = os.path.join(_HERE, "checkpoints")


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    totals = {'loss': 0, 'l1': 0, 'mstft': 0}

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(noisy)
                loss, l1, mstft = loss_fn(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(noisy)
            loss, l1, mstft = loss_fn(pred, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        totals['loss']  += loss.item()
        totals['l1']    += l1
        totals['mstft'] += mstft

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    totals = {'loss': 0, 'l1': 0, 'mstft': 0}

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                pred = model(noisy)
                loss, l1, mstft = loss_fn(pred, clean)
        else:
            pred = model(noisy)
            loss, l1, mstft = loss_fn(pred, clean)

        totals['loss']  += loss.item()
        totals['l1']    += l1
        totals['mstft'] += mstft

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def save_sample_plot(model, val_loader, device, epoch, out_dir):
    model.eval()
    with torch.no_grad():
        noisy, clean = next(iter(val_loader))
        noisy = noisy.to(device)
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                pred = model(noisy)
        else:
            pred = model(noisy)

    n = noisy[0, 0].cpu().numpy()
    c = clean[0, 0].numpy()
    p = pred[0, 0].float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(n.min(), c.min(), p.min())
    vmax = max(n.max(), c.max(), p.max())
    kw = dict(aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')
    axes[0].imshow(n, **kw); axes[0].set_title('Noisy (Bandpass)')
    axes[1].imshow(c, **kw); axes[1].set_title('Clean (MP3)')
    axes[2].imshow(p, **kw); axes[2].set_title(f'FullSubNet epoch {epoch}')
    for ax in axes:
        ax.set_xlabel('Time'); ax.set_ylabel('Mel bin')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:04d}.png"), dpi=120)
    plt.close()


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        use_amp = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        use_amp = False
        print("Device: CPU")

    num_workers = min(4, os.cpu_count() or 1) if device.type == 'cuda' else 0
    train_loader, val_loader, data = make_dataloaders(
        args.data, batch_size=args.batch, num_workers=num_workers)

    model = EfficientFullSubNetEnhancer(
        n_freqs=80, fb_hidden=args.fb_hidden, sb_hidden=args.sb_hidden,
        fb_layers=args.fb_layers, sb_layers=args.sb_layers,
        n_neighbor=args.n_neighbor, dropout=args.dropout,
    ).to(device)
    print(f"FullSubNet params: {model.count_params():,}")

    loss_fn   = GeneratorLoss(l1_weight=args.l1_w, mstft_weight=args.mstft_w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda') if use_amp else None

    os.makedirs(args.out, exist_ok=True)
    sample_dir = os.path.join(args.out, "training_samples")

    history = {'loss': [], 'val_loss': [], 'l1': [], 'val_l1': [], 'lr': []}
    best_val = float('inf')
    patience_counter = 0

    print(f"\nTraining {args.epochs} epochs | lr={args.lr} | batch={args.batch}")
    print(f"Loss: L1={args.l1_w}  MSTFT={args.mstft_w}")
    print(f"Model: fb_hidden={args.fb_hidden} sb_hidden={args.sb_hidden} "
          f"fb_layers={args.fb_layers} sb_layers={args.sb_layers}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        vl = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history['loss'].append(tr['loss'])
        history['val_loss'].append(vl['loss'])
        history['l1'].append(tr['l1'])
        history['val_l1'].append(vl['l1'])
        history['lr'].append(lr_now)

        print(f"{epoch:4d}  loss={tr['loss']:.5f}  val={vl['loss']:.5f}  "
              f"L1={tr['l1']:.4f}  MSTFT={tr['mstft']:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s", flush=True)

        if vl['loss'] < best_val:
            best_val = vl['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': best_val, 'args': vars(args),
            }, os.path.join(args.out, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        if epoch % args.plot_every == 0 or epoch == 1:
            save_sample_plot(model, val_loader, device, epoch, sample_dir)

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': vl['loss'], 'args': vars(args),
            }, os.path.join(args.out, f"ckpt_epoch_{epoch:04d}.pt"))

    # Save final + history
    torch.save({
        'epoch': epoch, 'model_state': model.state_dict(),
        'val_loss': vl['loss'], 'args': vars(args),
    }, os.path.join(args.out, "last_model.pt"))

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['l1'], label='Train L1')
    axes[1].plot(history['val_l1'], label='Val L1')
    axes[1].set_title('L1 Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].semilogy(history['lr'])
    axes[2].set_title('Learning Rate'); axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=130)
    plt.close()

    print(f"\nBest val loss: {best_val:.5f}")
    print(f"Checkpoints: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default=LOCAL_DATA)
    parser.add_argument('--out',        default=LOCAL_OUT)
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch',      type=int,   default=32)
    parser.add_argument('--fb_hidden',  type=int,   default=64)
    parser.add_argument('--sb_hidden',  type=int,   default=32)
    parser.add_argument('--fb_layers',  type=int,   default=2)
    parser.add_argument('--sb_layers',  type=int,   default=2)
    parser.add_argument('--n_neighbor', type=int,   default=3)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--l1_w',       type=float, default=1.0)
    parser.add_argument('--mstft_w',    type=float, default=1.0)
    parser.add_argument('--patience',   type=int,   default=30)
    parser.add_argument('--plot_every', type=int,   default=10)
    parser.add_argument('--save_every', type=int,   default=50)
    args = parser.parse_args()
    main(args)
