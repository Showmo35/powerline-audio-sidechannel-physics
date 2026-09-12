#!/usr/bin/env python3
"""
config.py — M31: does the MAINS COMB carry more than prosody? A full-scale,
open-vocabulary A/B, trained from scratch.

WHY THIS MODULE EXISTS
----------------------
M28 tested the harmonics on a single confusable pair (which/this) and the result was
muddled: harmonic features beat the envelope only modestly (0.672 vs 0.600) and a
deliberately WRONG harmonic grid did better still (0.735) — so that experiment could
not support the claim. But one word pair is far too thin to conclude from, and one
M28 finding is solid and unexplained: the comb is **NOT rank-1** (PC1 explains only
~0.26 of the per-harmonic AM gram; harmonics track the envelope with OPPOSITE signs).
If the K harmonics were redundant copies of one envelope, PC1 would be ~1.0. They are
not — so they should carry information beyond the single broadband envelope.

THE REAL TEST: put the comb through the full open-vocab retrieval generator (M22) at
scale and see whether it beats the raw-waveform front-end, judged by M30's learned
deep-metric retrieval. Same generator, same objective, same metric, same data, same
schedule — ONLY the front-end differs:

    ARM raw   : 0-16 kHz decimated waveform      (M22's PowerlineEncoder)   <- baseline
    ARM comb  : per-harmonic coherent gram       (CombEncoder, ALL harmonics)

METRIC (from M30): retrieval runs in a LEARNED deep-metric space (Projector head +
SupCon + ProxyAnchor prototypes), not M22's raw-pixel Pearson embedding. M30 showed
the learned metric is worth ~+6/+4 points, so the basic embedding was under-reading
the channel.

Trained FRESH (no warm start, no M14/M22 checkpoint).
"""
import os
import json
from dataclasses import dataclass

PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.environ.get('PLF_BIN_DIR', os.path.join(DATA_ROOT, 'soundbar_bin_captures'))
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')
MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.environ.get('PLF_OUT_DIR', os.path.join(MODULE_DIR, 'outputs'))
WORD_INDEX   = os.path.join(PROJECT_ROOT, 'M20_powerline_word_classifier', 'word_index.json')


@dataclass
class PLFConfig:
    # ── rates ──
    cap_sr: int = 200_000          # raw .bin (the comb needs the FULL rate)
    in_sr:  int = 32_000           # ARM raw only: decimated 0-16 kHz input
    ref_sr: int = 16_000

    # ── windowing ──
    win_s: float = 4.0
    n_frames: int = 400            # DiT sequence length (100 fps)

    # ── mel target ──
    n_fft: int = 1024
    hop:   int = 160
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # ── ARM raw: waveform encoder ──
    enc_strides: tuple = (8, 5, 4, 2)
    enc_kernels: tuple = (33, 17, 9, 5)
    enc_channels: tuple = (128, 192, 256, 384)
    enc_tf_layers: int = 4

    # ── DiT ──
    d_model: int = 384
    n_heads: int = 6
    dit_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ── flow matching / sampling ──
    sigma_min: float = 1e-4
    p_uncond: float = 0.1
    cfg_scale: float = 2.0
    sample_steps: int = 32

    # ── split ──
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, c): return os.path.join(BIN_DIR, f'{c}.bin')
    def wav_path(self, c): return os.path.join(WAV_DIR, f'{c}.wav')
    def lag_path(self, c): return os.path.join(BIN_DIR, f'{c}.lag')

    # pad the captured window so the comb's Tukey flat-centre crop lands EXACTLY on
    # win_s (otherwise the condition is time-stretched ~11% vs the mel target).
    pad_s: float = 0.40

    @property
    def in_len(self): return int(round(self.win_s * self.in_sr))
    @property
    def tot_s(self): return self.win_s + 2 * self.pad_s
    @property
    def raw_len(self): return int(round(self.tot_s * self.cap_sr))
    @property
    def pad_frac(self): return self.pad_s / self.tot_s
    @property
    def fps(self): return self.ref_sr / self.hop

    def is_test(self, chunk):
        return int(chunk.split('_')[1]) % self.test_every == 0


CFG = PLFConfig()


@dataclass
class RCFG:
    # ── word framing (matches M22/M23 so the gallery space is comparable) ──
    ctx_s: float = 0.10
    T: int = 64                    # time-normalized frames per word
    min_dur_s: float = 0.08
    max_dur_s: float = 2.00

    # ── OPEN vocabulary — every word type ──
    min_occ: int = 1
    max_per_word: int = 0

    # ── COMB front-end ──
    # K=1600 covers the ENTIRE comb to Nyquist (96 kHz). bw must be < f0/2 = 30 Hz so
    # adjacent harmonics never mix. Tg ~ 50 fps is the real bandwidth limit (+-25 Hz).
    K: int = 1600
    bw: float = 25.0
    Tg: int = 200                  # gram frames over win_s (50 fps; DiT interpolates up)

    # ── retrieval objective ──
    tau: float = 0.10
    lambda_con: float = 1.0        # SupCon
    lambda_proxy: float = 1.0      # ProxyAnchor (M30)
    lambda_corr: float = 1.0       # direct correlation term
    proj_hidden: int = 1024
    proj_dim: int = 128
    proxy_alpha: float = 32.0
    proxy_delta: float = 0.10
    retr_start: int = 8000         # flow-matching only before this
    retr_ramp: int = 4000

    # ── sampling ──
    train_steps: int = 8
    eval_steps: int = 32
    cfg_scale_eval: float = 2.0

    # ── eval ──
    gallery_per_word: int = 50
    eval_queries: int = 2000


RC = RCFG()


def build_vocab():
    occ = json.load(open(WORD_INDEX))
    words = sorted((w for w in occ if len(occ[w]) >= RC.min_occ),
                   key=lambda w: (-len(occ[w]), w))
    return words, occ
