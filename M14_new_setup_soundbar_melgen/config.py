#!/usr/bin/env python3
"""
config.py — M14 PowerLine-Flow (PLF): conditional flow-matching mel generator.

GOAL (new, not M13's ASR).  Regenerate the audio log-mel spectrogram from the
powerline .bin capture.  The lag-aligned capture matches the audio ENERGY
ENVELOPE almost perfectly (median frame-RMS xcorr ≈ 0.88, see
../Capture_Analysis/estimate_lag.py) but loses fine phonetic detail — so the
map powerline→mel is one-to-many.  A regression model would average those
possibilities into a gray blur; we instead learn a CONDITIONAL GENERATIVE model
that samples a sharp, realistic mel consistent with the envelope.

METHOD.  Rectified flow matching (Voicebox / Stable-Audio family) with a DiT
backbone:
  * a powerline encoder turns the raw 0-16 kHz window into frame-aligned
    condition tokens c[T, d];
  * a flow-transformer (DiT, adaLN-zero on the flow-time t, RoPE attention)
    learns the velocity field of the probability-flow ODE from noise→mel,
    conditioned on c;
  * inference integrates that ODE in a few Euler steps (classifier-free guided).

INPUT.  The raw 200 kSps capture, lag-shifted by its per-chunk .lag sidecar and
band-limited to 0-16 kHz (decimated 200 kHz → 32 kHz).  Fed as a raw waveform.
ALL 202 chunks are used (windowed on the fly).
"""

from dataclasses import dataclass, field
import os

# ── data roots (shared physical capture; same as M10–M13) ──────────────────────
PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')   # powerline .bin + .lag
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')            # clean ref .wav (16 kHz)

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class PLFConfig:
    # ── sample rates ──────────────────────────────────────────────────────────
    cap_sr: int = 200_000          # raw .bin capture rate on disk
    in_sr:  int = 32_000           # model input rate → band 0-16 kHz (decimated)
    ref_sr: int = 16_000           # clean reference rate (mel target source)

    # ── windowing (applied to every one of the 202 chunks) ────────────────────
    win_s: float = 4.0             # seconds per training window
    n_frames: int = 400            # fixed model sequence length T (mel + condition)
    windows_per_chunk: int = 64    # random windows sampled per chunk per split build

    # ── mel target (from the 16 kHz reference) ────────────────────────────────
    n_fft: int = 1024
    hop:   int = 160               # 100 fps  → win_s*ref_sr/hop = 400 frames
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # ── powerline encoder (raw 32 kHz → condition tokens at frame rate) ────────
    # strided 1-D conv stem; prod(strides) = in_sr / fps = 32000 / 100 = 320
    enc_strides: tuple = (8, 5, 4, 2)
    enc_kernels: tuple = (33, 17, 9, 5)
    enc_channels: tuple = (128, 192, 256, 384)
    enc_tf_layers: int = 4         # transformer blocks after the conv stem

    # ── DiT flow-transformer ──────────────────────────────────────────────────
    d_model: int = 384
    n_heads: int = 6
    dit_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ── flow matching / training ──────────────────────────────────────────────
    sigma_min: float = 1e-4        # tiny noise floor on the OT path endpoint
    p_uncond: float = 0.1          # classifier-free-guidance condition dropout
    cfg_scale: float = 2.0         # guidance weight at sampling time
    sample_steps: int = 32         # few-step Euler ODE (user choice: few-step)

    # ── split ─────────────────────────────────────────────────────────────────
    # hold out every 12th chunk (16 chunks, spread across the dataset)
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.bin')
    def wav_path(self, chunk): return os.path.join(WAV_DIR, f'{chunk}.wav')
    def lag_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.lag')

    @property
    def in_len(self):   # raw input samples per window at in_sr
        return int(round(self.win_s * self.in_sr))

    @property
    def fps(self):
        return self.ref_sr / self.hop

    def all_chunks(self):
        return [f'chunk_{n:03d}' for n in range(1, self.n_chunks + 1)]

    def test_chunks(self):
        return {f'chunk_{n:03d}' for n in range(self.test_every, self.n_chunks + 1, self.test_every)}


CFG = PLFConfig()
