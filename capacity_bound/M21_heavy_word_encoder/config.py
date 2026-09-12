#!/usr/bin/env python3
"""
config.py — M21: HEAVY, full-information word encoder from the 200 kHz powerline.

M20 said frequent words are separable (macro 31% / env-only 39% vs 3.3% chance) but
only via prosody. Critique: M20 used a compressed STFT view, a 12 M model, a light
run. M21 removes every excuse:
  * LOSSLESS raw 200 kHz (patchify, nothing thrown away) +
  * wideband STFT (all mains harmonics + AM sideband leakage) +
  * multi-scale envelope/prosody
  ... all three streams together, into a ~120 M transformer, trained hard.
Question: does it beat the envelope-only 39% ceiling? If yes -> real sub-envelope
word info. If it caps ~39% -> even maximal signal + capacity is only prosody.
"""
from dataclasses import dataclass
import os

ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX = ROOT + '/M20_powerline_word_classifier/word_index.json'    # reuse M20 alignment
OUT_DIR = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class M21Config:
    cap_sr: int = 200_000
    ref_sr: int = 16_000
    win_s: float = 1.5
    n_frames: int = 150             # 100 fps common grid

    # lossless raw patchify (zero-loss reshape of the full 200 kHz)
    patch_len: int = 2000           # 10 ms
    patch_hop: int = 1000           # 50 % overlap -> 299 patches, pooled to n_frames

    # wideband STFT (48.8 Hz bins -> resolves the 60 Hz comb + sidebands)
    w_nfft: int = 4096
    w_hop: int = 2000
    w_win: int = 4096
    log_eps: float = 1e-5

    # multi-scale envelope (prosody), RMS at several window sizes (ms)
    env_scales_ms: tuple = (4, 16, 64, 256)

    # vocabulary
    vocab_k: int = 30
    max_per_word: int = 800         # 2x M20 -> heavier data

    # HEAVY model
    d_model: int = 768
    n_heads: int = 12
    depth: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # split
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, c): return f'{BIN}/{c}.bin'
    def lag_path(self, c): return f'{BIN}/{c}.lag'
    def wav_path(self, c): return f'{WAVD}/{c}.wav'

    @property
    def a_len(self):
        return int(self.win_s * self.cap_sr)     # 300000

    @property
    def n_wbins(self):
        return self.w_nfft // 2 + 1

    def is_test(self, chunk):
        return int(chunk.split('_')[1]) % self.test_every == 0


CFG = M21Config()
