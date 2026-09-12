#!/usr/bin/env python3
"""
config.py — M16: from-scratch char-CTC ASR on M14 PowerLine-Flow generated mels.

QUESTION.  M14's generated mels look close to the real mels (mel_r ≈ 0.73,
env_r ≈ 0.87) yet Whisper reads ~100 % WER off them.  Two explanations:
  (a) the lexical content is absent (capture-side wall, M13/M14/M15 verdict), or
  (b) the content is present but systematically distorted — a "PLF accent"
      Whisper has never seen and cannot adapt to zero-shot.
M16 separates them: train an ASR FROM SCRATCH on (PLF-generated mel → true
LibriSpeech text) pairs.  Such a model learns whatever consistent phonetic cues
exist in the generated mels, however distorted.  If it too stays ~100 % WER on
held-out chunks, (a) is confirmed with no remaining excuse.

CONTROL.  A twin model with identical architecture / hyperparameters / split is
trained on the REAL reference mels.  Its (low) WER is the pipeline upper bound;
the gap between the twins is exactly the lexical information the PLF mels lack.

SPLIT.  Strictly the SAME chunk split as M14/M15: every 12th chunk held out.
The flow model may memorise its training chunks, so only test-chunk WER counts.
"""

from dataclasses import dataclass
import os

PROJECT_ROOT = '<REPO_ROOT>'
M14_DIR      = os.path.join(PROJECT_ROOT, 'M14_new_setup_soundbar_melgen')
M14_CKPT     = os.path.join(M14_DIR, 'outputs', 'cluster_a100', 'last.pt')  # EMA inside
MANIFEST     = os.path.join(PROJECT_ROOT, 'M15_soundbar_melgen_word', 'full_manifest.json')
WAV_DIR      = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures', 'audio_chunks')

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
GEN_DIR      = os.path.join(MODULE_DIR, 'gen_mels')      # {utt_id}.npy float16 [80, T]
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class ASRConfig:
    # ── mel geometry (MUST match M14's target mel) ─────────────────────────────
    ref_sr: int = 16_000
    n_fft: int = 1024
    hop:   int = 160               # 100 fps
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # ── utterances (same filter as M15) ────────────────────────────────────────
    min_dur_s: float = 2.0
    max_dur_s: float = 16.0

    # ── split (identical to M14/M15) ───────────────────────────────────────────
    test_every: int = 12
    n_chunks: int = 202

    # ── ASR encoder (conv ×4 subsample + pre-norm transformer + CTC) ───────────
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    vocab_size: int = 29           # text.py char CTC vocab (blank=0)

    # ── SpecAugment (train only) ───────────────────────────────────────────────
    n_freq_masks: int = 2
    freq_mask: int = 10
    n_time_masks: int = 2
    time_mask_frac: float = 0.05

    @property
    def fps(self):
        return self.ref_sr / self.hop

    def wav_path(self, chunk):
        return os.path.join(WAV_DIR, f'{chunk}.wav')

    def all_chunks(self):
        return [f'chunk_{n:03d}' for n in range(1, self.n_chunks + 1)]

    def test_chunks(self):
        return {f'chunk_{n:03d}' for n in range(self.test_every, self.n_chunks + 1, self.test_every)}


CFG = ASRConfig()
