import torch


EPS = 1e-8


def build_hann_window(win_length: int, device: torch.device):
    # Pure Hann has exact zeros at its edges, which can trigger PyTorch's
    # NOLA check in istft(center=False). Floor the window very slightly.
    return torch.hann_window(win_length, device=device).clamp_min(1e-5)


def waveform_to_stft(
    wave: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
):
    return torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )


def stft_to_waveform(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    length: int,
):
    # For center=False, the maximum reconstructable length from T frames is:
    # n_fft + hop_length * (T - 1). If caller asks longer, reconstruct the
    # valid part then right-pad to requested length.
    max_len = n_fft + hop_length * (spec.shape[-1] - 1)
    recon_len = min(length, max_len)

    wave = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        length=recon_len,
    )
    if recon_len < length:
        wave = torch.nn.functional.pad(wave, (0, length - recon_len))
    return wave


def stft_features(noisy_complex: torch.Tensor):
    noisy_real = noisy_complex.real
    noisy_imag = noisy_complex.imag
    noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + EPS)
    log_mag = torch.log1p(noisy_mag)
    feat = torch.stack([noisy_real, noisy_imag, log_mag], dim=1)
    return feat, noisy_real, noisy_imag, noisy_mag


def apply_complex_ratio_mask(noisy_complex: torch.Tensor, crm: torch.Tensor):
    mask_real = crm[:, 0]
    mask_imag = crm[:, 1]

    noisy_real = noisy_complex.real
    noisy_imag = noisy_complex.imag

    enh_real = mask_real * noisy_real - mask_imag * noisy_imag
    enh_imag = mask_real * noisy_imag + mask_imag * noisy_real

    return torch.complex(enh_real, enh_imag)
