#!/usr/bin/env python3
"""
config.py — M18: ViT reads the mel as an image → CTC emissions → LM assembles words.

The last test. M16 showed a small conv-CTC on M14's generated mels stays ~100 %
WER while its real-mel twin reaches 44.6 %. M18 asks whether a STRONGER reader
(a from-scratch ViT over the mel image) plus a LANGUAGE-MODEL back-end can pull
words out that M16's tiny reader missed — WITHOUT fooling ourselves.

The danger: an LM on a contentless signal invents fluent words rather than
recovering them (and a pretrained LLM has memorised LibriSpeech's public-domain
texts). So M18 is a NULL-SUBTRACTED trio, same ViT/hparams/split throughout:

  gen  — M14-generated mels (reused from M16/gen_mels)      [the experiment]
  real — reference mels                                      [ceiling]
  null — real mels with per-frame frequency SHUFFLE          [hallucination floor]
         (energy envelope preserved exactly, phonetics destroyed)

Read: WER(gen) ≈ WER(null) < WER(real) ⇒ the LM hallucinates, capture-wall final.
      WER(gen) ≪ WER(null)             ⇒ real residual signal, pursue.
"""

from dataclasses import dataclass
import os

PROJECT_ROOT = '<REPO_ROOT>'
M16_DIR      = os.path.join(PROJECT_ROOT, 'M16_asr_on_plf_mels')
GEN_DIR      = os.path.join(M16_DIR, 'gen_mels')                 # reuse M16's generated mels
MANIFEST     = os.path.join(PROJECT_ROOT, 'M15_soundbar_melgen_word', 'full_manifest.json')
WAV_DIR      = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures', 'audio_chunks')

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class ViTConfig:
    # ── mel geometry (must match M14/M16 generated mels) ───────────────────────
    ref_sr: int = 16_000
    n_fft: int = 1024
    hop:   int = 160               # 100 fps
    win_length: int = 640
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    log_eps: float = 1e-5

    # ── utterances (same filter + split as M15/M16) ────────────────────────────
    min_dur_s: float = 2.0
    max_dur_s: float = 16.0
    test_every: int = 12
    n_chunks: int = 202

    # ── ViT patchify: read the [80, T] mel as an image ─────────────────────────
    # patch (freq 16 × time 4): freq → 5 tokens (80/16), time → T/4 (25 fps, ok for CTC)
    patch_f: int = 16
    patch_t: int = 4
    max_time_patches: int = 400    # 16 s * 100 fps / 4
    # ── ViT trunk (from scratch — no ImageNet; spectrograms aren't natural images)
    d_model: int = 512
    n_heads: int = 8
    depth: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    vocab_size: int = 29           # text.py char CTC vocab (blank=0)

    # ── SpecAugment (train only) ───────────────────────────────────────────────
    n_freq_masks: int = 2
    freq_mask: int = 10
    n_time_masks: int = 2
    time_mask_frac: float = 0.05

    @property
    def n_freq_patches(self):
        return self.n_mels // self.patch_f     # 5

    @property
    def fps(self):
        return self.ref_sr / self.hop

    def wav_path(self, chunk):
        return os.path.join(WAV_DIR, f'{chunk}.wav')

    def test_chunks(self):
        return {f'chunk_{n:03d}' for n in range(self.test_every, self.n_chunks + 1, self.test_every)}


CFG = ViTConfig()
