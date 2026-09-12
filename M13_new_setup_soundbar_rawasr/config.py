#!/usr/bin/env python3
"""
config.py — M13 raw-signal ASR sweep (setup variant of ../M10_new_setup_soundbar).

SAME soundbar data as M10/M11/M12.  DIFFERENT method: instead of a hand-designed
AM-sideband mel front-end, every model gets the *raw* 200 kSps powerline window and
a SHARED, learned strided-conv front-end (SincNet/wav2vec-style) downsamples it to a
frame-rate feature image [B,1,F,T].  Five recognizer/enhancer architectures ported
from Modules 3-7 then sit on top.  The question: with a *fully learned* front-end on
the raw capture, does word-level WER move off ~100%?

This module consumes only the base module's utterance MANIFEST (a data artifact:
utt_id, chunk, text, start_s/end_s timing, lag) and reads the raw .bin windows on the
fly; it owns its own front-end, models, and outputs.
"""

from dataclasses import dataclass
import os

# ── data roots (shared physical capture; same as M10) ──────────────────────────
PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')   # powerline .bin + .lag
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')            # clean ref .wav (16 kHz)

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')

# the base module's manifest is the only artifact we reuse
SHARED_MANIFEST = os.path.join(
    PROJECT_ROOT, 'M10_new_setup_soundbar', 'outputs', 'dataset', 'manifest.jsonl')


@dataclass
class RawConfig:
    # sample rates
    cap_sr: int = 200_000          # raw .bin capture rate (what the model ingests)
    ref_sr: int = 16_000           # clean reference rate (for enh-model mel targets)

    # how much of each utterance to feed (cap to bound memory/seq-length)
    max_dur_s: float = 16.0        # raw windows cropped/padded to this many seconds

    # shared learned front-end (raw 200 kHz -> [B, F, T])
    # strides multiply to the total decimation; cap_sr / prod(strides) = output fps.
    fe_strides: tuple = (10, 5, 5, 2, 2, 2)   # prod = 4000  ->  ~50 fps
    fe_kernels: tuple = (21, 11, 11, 5, 5, 5)
    fe_channels: tuple = (32, 64, 64, 80, 80, 80)
    fe_out_dim: int = 80           # F: feature bins handed to the models (mel-image height)

    # mel target for the enhancement archs (M4, M7), computed from the reference wav
    tgt_n_fft: int = 400
    tgt_hop: int = 160
    tgt_n_mels: int = 80

    # split
    test_chunks_spec: str = '41-46'

    def bin_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.bin')
    def wav_path(self, chunk): return os.path.join(WAV_DIR, f'{chunk}.wav')
    def lag_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.lag')

    @property
    def fe_total_stride(self):
        s = 1
        for k in self.fe_strides:
            s *= k
        return s

    @property
    def fps(self):
        return self.cap_sr / self.fe_total_stride


CFG = RawConfig()
