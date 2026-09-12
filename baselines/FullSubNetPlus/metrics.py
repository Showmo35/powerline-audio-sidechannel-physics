from typing import Dict

import numpy as np
import torch

from audio_utils import mel_spectrogram_from_waveform
from losses import EPS, MultiResolutionSTFTLoss


def log_spectral_distance(pred_complex: torch.Tensor, target_complex: torch.Tensor):
    pred_mag = torch.abs(pred_complex)
    target_mag = torch.abs(target_complex)

    pred_db = 20.0 * torch.log10(pred_mag + EPS)
    target_db = 20.0 * torch.log10(target_mag + EPS)

    per_frame = torch.sqrt(torch.mean((pred_db - target_db) ** 2, dim=1))
    return per_frame.mean()


def spectral_convergence(pred_complex: torch.Tensor, target_complex: torch.Tensor):
    pred_mag = torch.abs(pred_complex)
    target_mag = torch.abs(target_complex)
    diff = target_mag - pred_mag
    num = torch.linalg.norm(diff, ord='fro', dim=(1, 2))
    den = torch.linalg.norm(target_mag, ord='fro', dim=(1, 2)) + EPS
    return (num / den).mean()


def mel_mse_and_corr(
    pred_wave: np.ndarray,
    target_wave: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int = 80,
):
    pred_mel = mel_spectrogram_from_waveform(
        pred_wave,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    target_mel = mel_spectrogram_from_waveform(
        target_wave,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    common_t = min(pred_mel.shape[1], target_mel.shape[1])
    pred_mel = pred_mel[:, :common_t]
    target_mel = target_mel[:, :common_t]

    mse = float(np.mean((pred_mel - target_mel) ** 2))

    x = pred_mel.reshape(-1)
    y = target_mel.reshape(-1)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, y)[0, 1])

    return mse, corr


def reconstruction_metrics(
    pred_complex: torch.Tensor,
    target_complex: torch.Tensor,
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    mrstft_metric: MultiResolutionSTFTLoss,
) -> Dict[str, float]:
    lsd = log_spectral_distance(pred_complex, target_complex)
    sc = spectral_convergence(pred_complex, target_complex)
    mrstft, _ = mrstft_metric(pred_wave, target_wave)

    return {
        'lsd': float(lsd.detach().cpu().item()),
        'spectral_convergence': float(sc.detach().cpu().item()),
        'multi_scale_stft': float(mrstft.detach().cpu().item()),
    }
