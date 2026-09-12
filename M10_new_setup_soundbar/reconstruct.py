#!/usr/bin/env python3
"""
reconstruct.py — mel → listenable audio, WITHOUT a neural vocoder.

Why Griffin-Lim instead of WaveRNN: WaveRNN is autoregressive (~9× slower than
real-time → ~180 GPU-h for the full 20 h corpus) and only needed for human
listening.  Griffin-Lim is ~real-time, deterministic, and good enough to feed an
ASR model.  This keeps the pipeline scalable to the whole dataset.

We invert the mel filterbank with a non-negative least-squares-ish pseudo-inverse
(clamped pinv) to get a linear magnitude spectrogram, then run Griffin-Lim.
"""

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as TT

from config import SetupConfig


def mel_to_audio(mel: np.ndarray, cfg: SetupConfig, device: str = 'cpu') -> np.ndarray:
    """Linear-magnitude mel [n_mels, T] → waveform at cfg.aud_sr via Griffin-Lim."""
    n_f = cfg.mel_n_fft // 2 + 1
    fb = AF.melscale_fbanks(n_freqs=n_f, f_min=cfg.mel_fmin, f_max=cfg.mel_fmax,
                            n_mels=cfg.mel_n_mels, sample_rate=cfg.aud_sr,
                            norm=None, mel_scale='htk').to(device)   # [n_f, n_mels]

    mel_t = torch.from_numpy(mel).float().to(device)                 # [n_mels, T]
    # invert filterbank: lin ≈ pinv(fb) @ mel, clamp ≥ 0 (energy is non-negative)
    fb_pinv = torch.linalg.pinv(fb)                                  # [n_mels, n_f]
    lin = (fb_pinv.T @ mel_t).clamp(min=0.0)                         # [n_f, T]

    gl = TT.GriffinLim(n_fft=cfg.mel_n_fft, win_length=cfg.mel_win,
                       hop_length=cfg.mel_hop, power=1.0,
                       n_iter=cfg.gl_iters).to(device)
    wav = gl(lin).cpu().numpy().astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak
    print(f'[gl] {cfg.gl_iters} iters → {len(wav)/cfg.aud_sr:.2f}s wav')
    return wav


def save_wav(path: str, wav: np.ndarray, sr: int) -> None:
    import wave
    x = wav / (np.abs(wav).max() + 1e-9) * 0.9
    pcm = (x * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
