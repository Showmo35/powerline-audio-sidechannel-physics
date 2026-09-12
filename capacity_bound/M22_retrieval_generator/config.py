#!/usr/bin/env python3
"""
config.py — M22: retrieval-optimized powerline→mel generator, trained FRESH from
scratch over the FULL open vocabulary.

GOAL.  Learn a generator that, from a powerline .bin window, produces a mel whose
generated word region has the HIGHEST correlation with the correct word's REAL mel
and is separated from other words — so kNN over a real-mel gallery returns the
correct word, for ANY word in the dataset (open vocabulary, ~25k word types).

FRESH.  No M14 checkpoint, no warm start.  The DiT architecture (models.py) and the
IO readers (data_io.py) are vendored into this module; weights are RANDOM-initialized
and the log-mel mean/std are estimated from the data at the start of training.

OBJECTIVE.  Flow matching learns to generate mels first; a retrieval loss (InfoNCE
over generated↔real word mels + a correlation term) is RAMPED IN afterwards and
shapes generations to be retrieval-optimal.  All retrieval terms live in the same
flattened-mel space the kNN uses (time-norm → flatten → mean-center → L2-norm, so
cosine == Pearson r).

Data: the shared 202-chunk soundbar capture; word occurrences come from M20's
word_index.json (a data artifact — {word: [[chunk, start, end], ...]}).
"""

import os
import json
from dataclasses import dataclass

# ── data roots (shared physical capture; same as M10–M23) ─────────────────────
PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
# BIN_DIR / OUT_DIR are env-overridable so the SAME module can train on a different
# device capture (e.g. monitor_bin_captures) into a separate output dir. Defaults =
# soundbar, so existing behavior is unchanged.
BIN_DIR      = os.environ.get('PLF_BIN_DIR',
                              os.path.join(DATA_ROOT, 'soundbar_bin_captures'))  # .bin + .lag
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')            # clean ref .wav (16 kHz, shared)

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.environ.get('PLF_OUT_DIR', os.path.join(MODULE_DIR, 'outputs'))
WORD_INDEX   = os.path.join(PROJECT_ROOT, 'M20_powerline_word_classifier', 'word_index.json')


# ══════════════════════════════════════════════════════════════════════════════
# Generator / mel config  (PowerLine-Flow architecture — vendored, trained fresh)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PLFConfig:
    # ── sample rates ──────────────────────────────────────────────────────────
    cap_sr: int = 200_000          # raw .bin capture rate on disk
    in_sr:  int = 32_000           # model input rate → band 0-16 kHz (decimated)
    ref_sr: int = 16_000           # clean reference rate (mel target source)

    # ── windowing ─────────────────────────────────────────────────────────────
    win_s: float = 4.0             # seconds per generator window
    n_frames: int = 400            # fixed model sequence length T (mel + condition)

    # ── mel target (from the 16 kHz reference) ────────────────────────────────
    n_fft: int = 1024
    hop:   int = 160               # 100 fps
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # ── powerline encoder (raw 32 kHz → condition tokens at frame rate) ────────
    enc_strides: tuple = (8, 5, 4, 2)     # prod = 320 = in_sr / fps
    enc_kernels: tuple = (33, 17, 9, 5)
    enc_channels: tuple = (128, 192, 256, 384)
    enc_tf_layers: int = 4

    # ── DiT flow-transformer ──────────────────────────────────────────────────
    d_model: int = 384
    n_heads: int = 6
    dit_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ── flow matching / sampling ──────────────────────────────────────────────
    sigma_min: float = 1e-4
    p_uncond: float = 0.1          # classifier-free-guidance condition dropout
    cfg_scale: float = 2.0         # guidance weight at sampling time
    sample_steps: int = 32         # full-quality Euler ODE (eval)

    # ── split ─────────────────────────────────────────────────────────────────
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.bin')
    def wav_path(self, chunk): return os.path.join(WAV_DIR, f'{chunk}.wav')
    def lag_path(self, chunk): return os.path.join(BIN_DIR, f'{chunk}.lag')

    @property
    def in_len(self):
        return int(round(self.win_s * self.in_sr))

    @property
    def fps(self):
        return self.ref_sr / self.hop

    def is_test(self, chunk):
        return int(chunk.split('_')[1]) % self.test_every == 0


CFG = PLFConfig()


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval / training config
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RCFG:
    # ── word framing (matches M20/M23 so the gallery space is comparable) ──────
    ctx_s: float = 0.10            # seconds of context each side of a word
    T: int = 64                    # time-normalized frames per word (kNN space)
    min_dur_s: float = 0.08        # drop degenerate occurrences
    max_dur_s: float = 2.00

    # ── OPEN VOCABULARY — ALL words in the dataset ────────────────────────────
    # min_occ = 1 keeps every word type (~25k); raise it only to trim the tail.
    min_occ: int = 1
    max_per_word: int = 0          # 0 = no per-word cap in TRAINING (use all occ)

    # ── retrieval objective ────────────────────────────────────────────────────
    tau: float = 0.10              # InfoNCE temperature
    lambda_con: float = 1.0        # InfoNCE (instance/supervised contrastive) weight
    lambda_corr: float = 1.0       # direct correlation term weight
    # flow matching is the base loss (weight 1.0); retrieval is ramped in on top.
    retr_start: int = 8000         # step to BEGIN adding the retrieval loss
    retr_ramp: int = 4000          # linear ramp 0→1 over this many steps

    # ── differentiable few-step sampling (train) vs full ODE (eval) ───────────
    train_steps: int = 8           # Euler steps we BACKPROP through (retrieval phase)
    eval_steps: int = 32
    cfg_scale_eval: float = 2.0

    # ── eval kNN sizing (open vocab is large — bound gallery & queries) ───────
    gallery_per_word: int = 50     # real train mels per word in the eval gallery
    eval_queries: int = 2000       # random test occurrences scored per eval


RC = RCFG()


def build_vocab():
    """Return (words, occ) over the FULL vocabulary.
    words = every word type kept by RC.min_occ (default: all ~25k);
    occ   = the full word_index.json map {word: [[chunk, s, e], ...]}."""
    occ = json.load(open(WORD_INDEX))
    words = sorted((w for w in occ if len(occ[w]) >= RC.min_occ),
                   key=lambda w: (-len(occ[w]), w))
    return words, occ
