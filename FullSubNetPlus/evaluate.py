"""
evaluate.py
-----------
Strict reconstruction evaluation on validation windows.

Reports only:
- LSD
- Spectral Convergence
- Multi-scale STFT loss
- Mel MSE
- Mel Correlation
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import AUDIO_SR, DEFAULT_CHECKPOINT_DIR, DEFAULT_DATA_NPZ, HOP_LENGTH, N_FFT, WIN_LENGTH
from losses import MultiResolutionSTFTLoss
from metrics import mel_mse_and_corr, reconstruction_metrics
from model import FullSubNetPlusLite
from stft_ops import apply_complex_ratio_mask, build_hann_window, stft_features, stft_to_waveform, waveform_to_stft


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get('args', {})

    model = FullSubNetPlusLite(
        fb_channels=ckpt_args.get('fb_channels', 48),
        fb_hidden=ckpt_args.get('fb_hidden', 64),
        sb_hidden=ckpt_args.get('sb_hidden', 96),
        subband_size=ckpt_args.get('subband_size', 7),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, 'best_model.pt')

    model, ckpt = load_model(ckpt_path, device)
    ckpt_args = ckpt.get('args', {})

    n_fft = args.n_fft if args.n_fft is not None else ckpt_args.get('n_fft', N_FFT)
    hop_length = args.hop_length if args.hop_length is not None else ckpt_args.get('hop_length', HOP_LENGTH)
    win_length = args.win_length if args.win_length is not None else ckpt_args.get('win_length', WIN_LENGTH)

    data = np.load(args.data)
    noisy_val = torch.from_numpy(data['noisy_val']).float()
    clean_val = torch.from_numpy(data['clean_val']).float()

    loader = DataLoader(
        TensorDataset(noisy_val, clean_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    mrstft_metric = MultiResolutionSTFTLoss()
    window = build_hann_window(win_length, device)

    totals = {
        'lsd': 0.0,
        'spectral_convergence': 0.0,
        'multi_scale_stft': 0.0,
        'mel_mse': 0.0,
        'mel_corr': 0.0,
    }
    n_items = 0

    with torch.no_grad():
        for noisy_wave, clean_wave in loader:
            noisy_wave = noisy_wave.to(device)
            clean_wave = clean_wave.to(device)

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

            batch_metrics = reconstruction_metrics(
                pred_complex=enh_spec,
                target_complex=clean_spec,
                pred_wave=enh_wave,
                target_wave=clean_wave,
                mrstft_metric=mrstft_metric,
            )

            bs = noisy_wave.shape[0]
            totals['lsd'] += batch_metrics['lsd'] * bs
            totals['spectral_convergence'] += batch_metrics['spectral_convergence'] * bs
            totals['multi_scale_stft'] += batch_metrics['multi_scale_stft'] * bs

            enh_np = enh_wave.detach().cpu().numpy()
            clean_np = clean_wave.detach().cpu().numpy()
            mel_mse_batch = 0.0
            mel_corr_batch = 0.0
            for i in range(bs):
                mel_mse, mel_corr = mel_mse_and_corr(
                    enh_np[i],
                    clean_np[i],
                    sr=AUDIO_SR,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=args.n_mels,
                )
                mel_mse_batch += mel_mse
                mel_corr_batch += mel_corr

            totals['mel_mse'] += mel_mse_batch
            totals['mel_corr'] += mel_corr_batch
            n_items += bs

    if n_items == 0:
        raise RuntimeError('Validation set is empty.')

    metrics = {
        'lsd': totals['lsd'] / n_items,
        'spectral_convergence': totals['spectral_convergence'] / n_items,
        'multi_scale_stft': totals['multi_scale_stft'] / n_items,
        'mel_mse': totals['mel_mse'] / n_items,
        'mel_corr': totals['mel_corr'] / n_items,
        'num_val_windows': int(n_items),
        'data': args.data,
        'checkpoint': ckpt_path,
    }

    print('Validation Reconstruction Metrics')
    print(f"  LSD                  : {metrics['lsd']:.6f}")
    print(f"  Spectral convergence : {metrics['spectral_convergence']:.6f}")
    print(f"  Multi-scale STFT     : {metrics['multi_scale_stft']:.6f}")
    print(f"  Mel MSE              : {metrics['mel_mse']:.6f}")
    print(f"  Mel Correlation      : {metrics['mel_corr']:.6f}")
    print(f"  Val windows          : {metrics['num_val_windows']}")

    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f'  JSON saved           : {args.out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FullSubNet+ model on validation set.')
    parser.add_argument('--data', default=DEFAULT_DATA_NPZ)
    parser.add_argument('--ckpt', default=os.path.join(DEFAULT_CHECKPOINT_DIR, 'best_model.pt'))
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--n_fft', type=int, default=None)
    parser.add_argument('--hop_length', type=int, default=None)
    parser.add_argument('--win_length', type=int, default=None)
    parser.add_argument('--n_mels', type=int, default=80)

    parser.add_argument('--out_json', default=None)

    main(parser.parse_args())
