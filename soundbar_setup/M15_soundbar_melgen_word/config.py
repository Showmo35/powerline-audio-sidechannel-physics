#!/usr/bin/env python3
"""
config.py — M15 PowerLine-Flow-Word (PLF-W).

Extends M14 (PowerLine-Flow) with two ideas aimed at recovering WORDS, not just
the envelope:

  1. MULTI-STREAM input from the FULL 200 kHz capture (M14 only saw 0-16 kHz):
       * stream A — the raw 200 kHz waveform (a learned conv front-end; this
         alone already contains every mains-harmonic AM sideband n·60±f, i.e. the
         high-frequency detail M14 threw away);
       * stream B — a wideband log-mel of the same 200 kHz window (0-100 kHz),
         which exposes the harmonic/sideband structure explicitly as a feature.
  2. A CTC WORD-LOSS on the shared encoder: an auxiliary recognizer is trained on
     the true LibriSpeech text (full 202-chunk manifest, exact utterance timing),
     so the encoder is PUNISHED for not carrying phonetic content.  Total loss =
     flow_matching + lambda_ctc · CTC.  Sweeping lambda_ctc trades mel_r for WER —
     the tradeoff the capture allows.

Windows are UTTERANCE-aligned (from full_manifest.json) so each item has exact
text; audio is cropped/padded to max_dur.
"""

from dataclasses import dataclass
import os

PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')
MANIFEST     = os.path.join(MODULE_DIR, 'full_manifest.json')


@dataclass
class PLFWConfig:
    # ── sample rates ──────────────────────────────────────────────────────────
    cap_sr: int = 200_000          # FULL raw capture rate the model now ingests
    ref_sr: int = 16_000           # reference / mel-target rate

    # ── utterance windowing ───────────────────────────────────────────────────
    max_dur_s: float = 16.0        # utterances cropped/padded to this (covers most)
    min_dur_s: float = 2.0         # skip utterances shorter than this
    n_frames: int = 1600           # T = max_dur_s * fps  (16 s * 100 fps)

    # ── mel target (from 16 kHz reference) ────────────────────────────────────
    n_fft: int = 1024
    hop:   int = 160               # 100 fps
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # ── stream A: raw-200 kHz conv front-end ──────────────────────────────────
    # prod(strides) = cap_sr / fps = 200000 / 100 = 2000
    aStrides: tuple = (10, 10, 5, 2, 2)
    aKernels: tuple = (41, 21, 11, 5, 5)
    aChannels: tuple = (64, 128, 192, 256, 384)

    # ── stream B: wideband log-mel of the 200 kHz (0-100 kHz) ──────────────────
    b_n_fft: int = 2048
    b_hop:   int = 2000            # 200000/2000 = 100 fps → aligns to n_frames
    b_n_mels: int = 128
    b_fmax: float = 100_000.0

    # ── shared encoder + DiT + CTC head ───────────────────────────────────────
    d_model: int = 384
    n_heads: int = 6
    enc_tf_layers: int = 4
    dit_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ctc_vocab: int = 29            # text.VOCAB_SIZE

    # ── flow matching ─────────────────────────────────────────────────────────
    p_uncond: float = 0.1
    cfg_scale: float = 3.0         # M14 sweep found ~3.0 optimal
    sample_steps: int = 32
    lambda_ctc: float = 1.0        # word-loss weight (sweepable)

    # ── split (by chunk, every 12th held out) ─────────────────────────────────
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.bin')
    def wav_path(self, chunk): return os.path.join(WAV_DIR, f'{chunk}.wav')
    def lag_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.lag')

    @property
    def a_len(self):
        return int(round(self.max_dur_s * self.cap_sr))

    @property
    def fps(self):
        return self.ref_sr / self.hop

    def test_chunks(self):
        return {f'chunk_{n:03d}' for n in range(self.test_every, self.n_chunks + 1, self.test_every)}


CFG = PLFWConfig()
