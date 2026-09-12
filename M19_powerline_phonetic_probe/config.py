#!/usr/bin/env python3
"""
config.py — M19: does ANY powerline feature carry PHONETIC (spectral-shape) info?

Not another end-to-end ASR. A direct regression probe: predict the true audio
log-mel from powerline features, and report the ENVELOPE-REMOVED correlation —
after subtracting each frame's mean energy, does the powerline predict the SHAPE
of the audio spectrum (which distinguishes phonemes), or only its loudness?

Every prior number (mel_r 0.73, env_r 0.88) folds in the envelope. This isolates
the phonetic part. Feature families compared:
  env  — per-frame RMS of the 200 kHz signal (loudness only) → baseline; its
         envelope-removed correlation is ~0 by construction.
  wide — wideband log-STFT of the FULL 200 kHz (0-100 kHz, 48.8 Hz bins): contains
         every mains harmonic and its AM sidebands n·60±f — the only place audio
         spectrum could hide. If phonetics are anywhere, a probe on `wide` finds them.

Read: wide envelope-removed r ≈ 0 (≈ env) ⇒ no phonetic info in the powerline,
capture-side wall is physical. wide envelope-removed r ≫ env ⇒ spectral/phonetic
signal IS present (M15 just didn't extract it) → pursue that representation.
"""

from dataclasses import dataclass
import os

PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')
MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class ProbeConfig:
    cap_sr: int = 200_000
    ref_sr: int = 16_000

    win_s: float = 4.0
    n_frames: int = 400            # 4 s * 100 fps
    windows_per_chunk: int = 48

    # target audio mel (same as M14)
    n_fft: int = 1024
    hop:   int = 160               # 100 fps
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # wideband STFT of the 200 kHz powerline (resolves 60 Hz harmonics: 48.8 Hz bins)
    w_nfft: int = 4096
    w_hop:  int = 2000             # 200000/2000 = 100 fps → aligns to n_frames
    w_win:  int = 4096             # → n_wbins = 2049 over 0-100 kHz

    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, c): return os.path.join(BIN_DIR, f'{c}.bin')
    def wav_path(self, c): return os.path.join(WAV_DIR, f'{c}.wav')
    def lag_path(self, c): return os.path.join(BIN_DIR, f'{c}.lag')

    @property
    def n_wbins(self):
        return self.w_nfft // 2 + 1

    def all_chunks(self):
        return [f'chunk_{n:03d}' for n in range(1, self.n_chunks + 1)]

    def test_chunks(self):
        return {f'chunk_{n:03d}' for n in range(self.test_every, self.n_chunks + 1, self.test_every)}


CFG = ProbeConfig()
