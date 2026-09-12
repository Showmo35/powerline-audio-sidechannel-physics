#!/usr/bin/env python3
"""
config.py — setup VARIANT: soundbar capture, LEARNED front-end.

Same physical setup as ../setup_soundbar (LibriSpeech through a soundbar, USRP
@200 kSps on the AC cord) but a different *method*: instead of the fixed
8-harmonic AM-sideband sum, a U-Net learns powerline→clean-mel enhancement from
the paired data.  This module is self-contained; it only consumes the sibling's
utterance manifest (a data artifact: utt_id, chunk, text, timing) and otherwise
owns its own features, model, and outputs.
"""

from dataclasses import dataclass
import os

PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')
FEAT_DIR     = os.path.join(OUT_DIR, 'dataset_enh')          # stacks + clean targets
# utterance list/timing/text reused from the fixed-front-end module (data only):
SHARED_MANIFEST = os.path.join(PROJECT_ROOT, 'setup_soundbar', 'outputs',
                               'dataset', 'manifest.jsonl')


@dataclass
class SetupConfig:
    cap_sr: int = 200_000
    aud_sr: int = 22_050
    asr_sr: int = 16_000

    mains_guess_hz: float = 60.0
    mains_search_hz: float = 5.0
    n_harmonics: int = 8           # → 2*n_harmonics input channels

    mel_n_fft: int = 2048
    mel_win:   int = 1100
    mel_hop:   int = 275
    mel_n_mels: int = 80
    mel_fmin: float = 40.0
    mel_fmax: float = 11025.0

    # learned-model training
    crop_frames: int = 512         # ~6.4 s joint random crop for batching
    base_ch: int = 48              # U-Net base width

    def bin_path(self, chunk):  return os.path.join(BIN_DIR, f'{chunk}.bin')
    def wav_path(self, chunk):  return os.path.join(WAV_DIR, f'{chunk}.wav')
    def lag_path(self, chunk):  return os.path.join(BIN_DIR, f'{chunk}.lag')


CFG = SetupConfig()
