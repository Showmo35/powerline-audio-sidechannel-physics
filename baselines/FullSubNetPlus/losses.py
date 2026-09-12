from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass
class LossWeights:
    complex_mse: float = 1.0
    log_mag_l1: float = 0.5
    phase_consistency: float = 0.2
    mrstft: float = 1.0


def complex_mse_loss(enhanced: torch.Tensor, clean: torch.Tensor):
    return F.mse_loss(enhanced.real, clean.real) + F.mse_loss(enhanced.imag, clean.imag)


def log_magnitude_l1_loss(enhanced: torch.Tensor, clean: torch.Tensor):
    enh_mag = torch.abs(enhanced)
    clean_mag = torch.abs(clean)
    return F.l1_loss(torch.log1p(enh_mag), torch.log1p(clean_mag))


def phase_consistency_loss(enhanced: torch.Tensor, clean: torch.Tensor):
    enh_real = enhanced.real
    enh_imag = enhanced.imag
    clean_real = clean.real
    clean_imag = clean.imag

    dot = enh_real * clean_real + enh_imag * clean_imag
    denom = (torch.abs(enhanced) * torch.abs(clean)) + EPS
    cos = torch.clamp(dot / denom, min=-1.0, max=1.0)

    weight = torch.abs(clean)
    weight = weight / (weight.mean() + EPS)
    return ((1.0 - cos) * weight).mean()


def _stft_mag(x: torch.Tensor, n_fft: int, hop_length: int):
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    return torch.abs(spec)


def spectral_convergence_loss(pred_mag: torch.Tensor, target_mag: torch.Tensor):
    # pred_mag/target_mag: (B, F, T)
    diff = target_mag - pred_mag
    num = torch.linalg.norm(diff, ord='fro', dim=(1, 2))
    den = torch.linalg.norm(target_mag, ord='fro', dim=(1, 2)) + EPS
    return (num / den).mean()


def single_resolution_stft_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    n_fft: int,
    hop_length: int,
):
    pred_mag = _stft_mag(pred_wave, n_fft=n_fft, hop_length=hop_length)
    target_mag = _stft_mag(target_wave, n_fft=n_fft, hop_length=hop_length)

    sc = spectral_convergence_loss(pred_mag, target_mag)
    log_l1 = F.l1_loss(torch.log(pred_mag + EPS), torch.log(target_mag + EPS))
    return sc + log_l1, sc


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes: Iterable[int] = (256, 512, 1024),
        hop_sizes: Iterable[int] = (64, 128, 256),
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes)
        if len(self.fft_sizes) != len(self.hop_sizes):
            raise ValueError('fft_sizes and hop_sizes must have equal length.')

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor):
        losses = []
        sc_losses = []
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            val, sc = single_resolution_stft_loss(
                pred_wave,
                target_wave,
                n_fft=n_fft,
                hop_length=hop,
            )
            losses.append(val)
            sc_losses.append(sc)
        return torch.stack(losses).mean(), torch.stack(sc_losses).mean()


class CompositeComplexLoss(torch.nn.Module):
    def __init__(
        self,
        weights: LossWeights,
        mrstft_fft_sizes: Tuple[int, ...] = (256, 512, 1024),
        mrstft_hop_sizes: Tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        self.weights = weights
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=mrstft_fft_sizes,
            hop_sizes=mrstft_hop_sizes,
        )

    def forward(
        self,
        enhanced_complex: torch.Tensor,
        clean_complex: torch.Tensor,
        enhanced_wave: torch.Tensor,
        clean_wave: torch.Tensor,
    ):
        c_mse = complex_mse_loss(enhanced_complex, clean_complex)
        lmag = log_magnitude_l1_loss(enhanced_complex, clean_complex)
        phs = phase_consistency_loss(enhanced_complex, clean_complex)
        mrstft, sc = self.mrstft(enhanced_wave, clean_wave)

        total = (
            self.weights.complex_mse * c_mse
            + self.weights.log_mag_l1 * lmag
            + self.weights.phase_consistency * phs
            + self.weights.mrstft * mrstft
        )

        return {
            'total': total,
            'complex_mse': c_mse,
            'log_mag_l1': lmag,
            'phase_consistency': phs,
            'mrstft': mrstft,
            'spectral_convergence': sc,
        }
