"""
train.py
--------
Train FullSubNet+-style complex mask model on 4s waveform windows.

Loss terms:
- complex MSE (real/imag)
- log-magnitude L1
- phase-consistency
- multi-resolution STFT
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_NPZ,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
)
from dataset import make_dataloaders
from losses import CompositeComplexLoss, LossWeights
from model import FullSubNetPlusLite
from stft_ops import (
    apply_complex_ratio_mask,
    build_hann_window,
    stft_features,
    stft_to_waveform,
    waveform_to_stft,
)


def _mean_dict(acc, n):
    return {k: v / max(1, n) for k, v in acc.items()}


def train_or_eval_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    n_fft,
    hop_length,
    win_length,
    scaler=None,
    train=True,
):
    if train:
        model.train()
    else:
        model.eval()

    stats = {
        'total': 0.0,
        'complex_mse': 0.0,
        'log_mag_l1': 0.0,
        'phase_consistency': 0.0,
        'mrstft': 0.0,
        'spectral_convergence': 0.0,
    }

    window = build_hann_window(win_length, device)

    for noisy_wave, clean_wave in loader:
        noisy_wave = noisy_wave.to(device, non_blocking=True)
        clean_wave = clean_wave.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = scaler is not None and device.type == 'cuda'
        with torch.set_grad_enabled(train):
            with torch.amp.autocast('cuda', enabled=autocast_enabled):
                noisy_spec = waveform_to_stft(noisy_wave, n_fft, hop_length, win_length, window)
                clean_spec = waveform_to_stft(clean_wave, n_fft, hop_length, win_length, window)

                feat, _, _, _ = stft_features(noisy_spec)
                crm = model(feat)
                enh_spec = apply_complex_ratio_mask(noisy_spec, crm)
                enh_wave = stft_to_waveform(
                    enh_spec,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    length=noisy_wave.shape[-1],
                )

                loss_dict = criterion(
                    enhanced_complex=enh_spec,
                    clean_complex=clean_spec,
                    enhanced_wave=enh_wave,
                    clean_wave=clean_wave,
                )
                total_loss = loss_dict['total']

        if train:
            if scaler is not None and device.type == 'cuda':
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        for key in stats:
            stats[key] += float(loss_dict[key].detach().cpu().item())

    return _mean_dict(stats, len(loader))


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_gpus = torch.cuda.device_count()
        print(f'GPU(s): {n_gpus} | {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}')
    else:
        device = torch.device('cpu')
        n_gpus = 0
        print('Device: CPU')

    num_workers = min(4, os.cpu_count() or 1) if n_gpus > 0 else 0
    train_loader, val_loader, _ = make_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    model = FullSubNetPlusLite(
        fb_channels=args.fb_channels,
        fb_hidden=args.fb_hidden,
        sb_hidden=args.sb_hidden,
        subband_size=args.subband_size,
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f'DataParallel enabled on {n_gpus} GPUs.')

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f'Model params: {raw_model.count_params():,}')

    weights = LossWeights(
        complex_mse=args.w_complex_mse,
        log_mag_l1=args.w_logmag,
        phase_consistency=args.w_phase,
        mrstft=args.w_mrstft,
    )
    criterion = CompositeComplexLoss(weights=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float('inf')
    patience = 0

    print('\nTraining configuration:')
    print(f'  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}')
    print(f'  n_fft={args.n_fft}, hop={args.hop_length}, win={args.win_length}')
    print(
        '  weights='
        f'(complex={args.w_complex_mse}, logmag={args.w_logmag}, '
        f'phase={args.w_phase}, mrstft={args.w_mrstft})'
    )
    print('\n' + f"{'Ep':>4} {'Train':>9} {'Val':>9} {'CMSE':>8} {'LMAG':>8} {'PHS':>8} {'MRSTFT':>8} {'SC':>8} {'LR':>9} {'Time':>6}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_or_eval_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            scaler=scaler,
            train=True,
        )

        with torch.no_grad():
            vl = train_or_eval_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                scaler=None,
                train=False,
            )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"{epoch:4d} {tr['total']:9.5f} {vl['total']:9.5f} "
            f"{vl['complex_mse']:8.4f} {vl['log_mag_l1']:8.4f} {vl['phase_consistency']:8.4f} "
            f"{vl['mrstft']:8.4f} {vl['spectral_convergence']:8.4f} {lr_now:9.2e} {elapsed:5.1f}s",
            flush=True,
        )

        ckpt = {
            'epoch': epoch,
            'model_state': raw_model.state_dict(),
            'val_loss': vl['total'],
            'args': vars(args),
        }

        if vl['total'] < best_val:
            best_val = vl['total']
            patience = 0
            torch.save(ckpt, os.path.join(args.out_dir, 'best_model.pt'))
        else:
            patience += 1

        if epoch % args.save_every == 0:
            torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_epoch_{epoch:04d}.pt'))

        if patience >= args.patience:
            print(f'Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs).')
            break

    torch.save(ckpt, os.path.join(args.out_dir, 'last_model.pt'))
    print(f'\nBest val loss: {best_val:.6f}')
    print(f'Checkpoints: {args.out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FullSubNet+-style complex STFT pipeline.')

    parser.add_argument('--data', default=DEFAULT_DATA_NPZ)
    parser.add_argument('--out_dir', default=DEFAULT_CHECKPOINT_DIR)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--save_every', type=int, default=20)

    parser.add_argument('--n_fft', type=int, default=N_FFT)
    parser.add_argument('--hop_length', type=int, default=HOP_LENGTH)
    parser.add_argument('--win_length', type=int, default=WIN_LENGTH)

    parser.add_argument('--fb_channels', type=int, default=48)
    parser.add_argument('--fb_hidden', type=int, default=64)
    parser.add_argument('--sb_hidden', type=int, default=96)
    parser.add_argument('--subband_size', type=int, default=7)

    parser.add_argument('--w_complex_mse', type=float, default=1.0)
    parser.add_argument('--w_logmag', type=float, default=0.5)
    parser.add_argument('--w_phase', type=float, default=0.2)
    parser.add_argument('--w_mrstft', type=float, default=1.0)

    main(parser.parse_args())
