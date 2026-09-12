#!/usr/bin/env python3
"""
config.py — M17 PLF-W on a LOSSLESS full-200 kHz input, high-res spectrogram target.

Recreates M15 (flow-matching DiT + CTC word-loss, utterance-aligned, same chunk
split) with the last model-side loophole closed:

  1. INPUT: the full 200 kHz raw capture enters via a ZERO-LOSS reshape — 10 ms
     patches of 2000 samples with 50 % overlap, then a learned Linear. No
     decimation, no STFT, no fixed filterbank: the only compression is learned,
     end-to-end, punished by the CTC word-loss if it discards phonetics.
     (M14 threw away everything >16 kHz; M15's conv stem funneled 200 k samples/s
     into 38.4 k dims/s and its wideband mel was lossy. M17 discards nothing.)
  2. TARGET: 513-bin linear-frequency log-STFT of the 16 kHz reference (6.4× the
     80-mel resolution; Griffin-Lim inverts it directly, no mel pseudo-inverse).

If WER still pins at ~100 % here, preprocessing is exonerated and the capture-wall
verdict (M13/M14/M15/M16) is final. If WER moves, the high-rate detail mattered.
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
class PLFW17Config:
    # ── sample rates ──────────────────────────────────────────────────────────
    cap_sr: int = 200_000          # FULL raw capture rate — ingested losslessly
    ref_sr: int = 16_000           # reference / spectrogram-target rate

    # ── utterance windowing (same as M15) ─────────────────────────────────────
    max_dur_s: float = 16.0
    min_dur_s: float = 2.0
    n_frames: int = 1600           # 16 s * 100 fps

    # ── target: high-res linear log-STFT of the 16 kHz reference ──────────────
    n_fft: int = 1024
    hop:   int = 160               # 100 fps
    win_length: int = 640
    log_eps: float = 1e-5

    # ── lossless patchify front-end (replaces M15's streams A+B) ──────────────
    patch_len: int = 2000          # 10 ms of 200 kHz — pure reshape, every sample enters
    patch_hop: int = 1000          # 50 % overlap → 2·n_frames tokens before pooling
    d_patch:   int = 1024          # learned Linear 2000→1024 (the ONLY compression)

    # ── shared encoder + DiT + CTC head (scaled up vs M15) ─────────────────────
    d_model: int = 512
    n_heads: int = 8
    enc_tf_layers: int = 6
    dit_layers: int = 10
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ctc_vocab: int = 29            # text.VOCAB_SIZE

    # ── flow matching (unchanged from M15) ────────────────────────────────────
    p_uncond: float = 0.1
    cfg_scale: float = 3.0
    sample_steps: int = 32
    lambda_ctc: float = 1.0

    # ── split (identical to M14/M15/M16: every 12th chunk held out) ───────────
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.bin')
    def wav_path(self, chunk): return os.path.join(WAV_DIR, f'{chunk}.wav')
    def lag_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.lag')

    @property
    def n_bins(self):              # spectrogram target height
        return self.n_fft // 2 + 1

    @property
    def a_len(self):               # raw samples per (padded) utterance window
        return int(round(self.max_dur_s * self.cap_sr))

    @property
    def fps(self):
        return self.ref_sr / self.hop

    def test_chunks(self):
        return {f'chunk_{n:03d}' for n in range(self.test_every, self.n_chunks + 1, self.test_every)}


CFG = PLFW17Config()
