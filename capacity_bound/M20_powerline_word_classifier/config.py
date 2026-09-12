#!/usr/bin/env python3
"""
config.py — M20: closed-set frequent-word classifier from the 200 kHz powerline.

Reframe: not open-vocab ASR (which fails) but "can the powerline reliably tell a
FIXED set of frequent words apart?" Each word occurrence -> full 200 kHz signal
(word + context) -> wideband STFT (resolves the 60 Hz harmonic comb and its AM
subband leakage) -> conv+transformer -> K-way word label. Chunk split; per-word
accuracy vs chance; wav2vec2 audio linear probe as the ceiling.
"""
from dataclasses import dataclass
import os

ROOT = '<REPO_ROOT>'
BIN = ROOT + '/Powerline_Data_Captures/soundbar_bin_captures'
WAVD = ROOT + '/Powerline_Data_Captures/audio_chunks'
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(MODULE_DIR, 'word_index.json')
OUT_DIR = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class M20Config:
    cap_sr: int = 200_000
    ref_sr: int = 16_000

    # word window: center on the word, take a fixed context span (long-term dependency)
    win_s: float = 1.5
    n_frames: int = 150            # win_s * 100 fps

    # wideband STFT of the 200 kHz signal (48.8 Hz bins -> resolves 60 Hz comb + sidebands)
    w_nfft: int = 4096
    w_hop: int = 2000              # 200000/2000 = 100 fps
    w_win: int = 4096
    log_eps: float = 1e-5

    # vocabulary: top-K frequent words
    vocab_k: int = 30
    max_per_word: int = 400        # cap to limit imbalance from ultra-common words

    # model
    d_model: int = 384
    n_heads: int = 6
    depth: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    freq_pool: int = 8             # conv stride over frequency

    # split (every 12th chunk held out)
    test_every: int = 12
    n_chunks: int = 202

    def bin_path(self, c): return f'{BIN}/{c}.bin'
    def wav_path(self, c): return f'{WAVD}/{c}.wav'
    def lag_path(self, c): return f'{BIN}/{c}.lag'

    @property
    def n_wbins(self):
        return self.w_nfft // 2 + 1

    def is_test(self, chunk):
        n = int(chunk.split('_')[1])
        return n % self.test_every == 0


CFG = M20Config()
