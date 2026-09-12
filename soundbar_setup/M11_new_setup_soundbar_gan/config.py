#!/usr/bin/env python3
"""
config.py — setup VARIANT: soundbar capture, CONDITIONAL GENERATIVE front-end.

Same soundbar data; method = a HiFi-GAN-style generator that maps the powerline
per-harmonic mel stack → CLEAN 16 kHz WAVEFORM, trained with adversarial +
feature-matching + mel + content(wav2vec2) losses.  Goal: avoid the L1-regression
blur of setup_soundbar_learned by using a generative objective with a content
loss that keeps output faithful (not just plausible).  Judged by Whisper WER.

Everything is at 16 kHz so the upsampling math is clean (hop 200 = product of the
upsample rates) and the content loss / WER see real audio in Whisper's rate.
"""

from dataclasses import dataclass, field
from typing import List
import os

PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')
MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')
FEAT_DIR     = os.path.join(OUT_DIR, 'dataset_gan')          # powerline stacks @16k
SHARED_MANIFEST = os.path.join(PROJECT_ROOT, 'setup_soundbar', 'outputs',
                               'dataset', 'manifest.jsonl')


@dataclass
class SetupConfig:
    cap_sr: int = 200_000
    sr: int = 16_000               # working + output + ASR rate

    mains_guess_hz: float = 60.0
    mains_search_hz: float = 5.0
    n_harmonics: int = 8           # → 2*n_harmonics input channels

    # mel / STFT (16 kHz, hop = product of upsample rates)
    n_fft: int = 1024
    win: int = 800
    hop: int = 200
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 8000.0

    # HiFi-GAN generator
    upsample_rates: List[int] = field(default_factory=lambda: [10, 5, 2, 2])     # prod=200=hop
    upsample_kernels: List[int] = field(default_factory=lambda: [20, 10, 4, 4])
    upsample_initial: int = 256
    resblock_kernels: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])

    # training
    seg_frames: int = 64           # 64*hop = 12800 samples (0.8 s) per segment
    w_mel: float = 45.0
    w_fm: float = 2.0
    w_content: float = 15.0
    content_model: str = 'facebook/wav2vec2-base-960h'

    def bin_path(self, c):  return os.path.join(BIN_DIR, f'{c}.bin')
    def wav_path(self, c):  return os.path.join(WAV_DIR, f'{c}.wav')
    def lag_path(self, c):  return os.path.join(BIN_DIR, f'{c}.lag')


CFG = SetupConfig()
