#!/usr/bin/env python3
"""
config.py — Setup-specific constants for the `setup_soundbar` module.

One module folder == one physical capture setup.  This module describes the
SOUNDBAR setup: LibriSpeech train-clean-100 played through a soundbar, with a
USRP sampling the AC-cord current at 200 kSps (real).  Future setups (different
appliance / probe / rate) get their own sibling module folder with their own
config.py; nothing here is imported across setups.

Everything that another setup might change lives in this file so the rest of the
pipeline is setup-agnostic.
"""

from dataclasses import dataclass, field
import os

# ── data roots ────────────────────────────────────────────────────────────────
PROJECT_ROOT = '<REPO_ROOT>'
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Powerline_Data_Captures')
BIN_DIR      = os.path.join(DATA_ROOT, 'soundbar_bin_captures')   # powerline .bin + .lag
WAV_DIR      = os.path.join(DATA_ROOT, 'audio_chunks')            # clean reference .wav (16 kHz)

MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(MODULE_DIR, 'outputs')


@dataclass
class SetupConfig:
    """All knobs for the soundbar setup. Pass an instance through the pipeline."""
    # sample rates
    cap_sr: int = 200_000          # USRP real-sampling rate of the .bin captures
    aud_sr: int = 22_050           # working rate for the AM-sideband mel front-end
    asr_sr: int = 16_000           # Whisper input rate (also the native reference rate)

    # mains / AM carrier
    mains_guess_hz: float = 60.0   # US mains; detect_mains refines within ±search
    mains_search_hz: float = 5.0
    n_harmonics: int = 8           # mains harmonics summed in the sideband front-end

    # mel front-end (kept compatible with the proven Capture_Analysis params)
    mel_n_fft: int = 2048
    mel_win:   int = 1100
    mel_hop:   int = 275
    mel_n_mels: int = 80
    mel_fmin: float = 40.0
    mel_fmax: float = 11025.0

    # framing for VAD / alignment
    frame_s: float = 0.05

    # reconstruction
    gl_iters: int = 64             # Griffin-Lim iterations (fast, no neural vocoder)

    # default evaluation window
    default_chunk: str = 'chunk_002'
    default_start_s: float = 0.0
    default_dur_s: float = 60.0

    def bin_path(self, chunk: str) -> str:
        return os.path.join(BIN_DIR, f'{chunk}.bin')

    def wav_path(self, chunk: str) -> str:
        return os.path.join(WAV_DIR, f'{chunk}.wav')

    def lag_path(self, chunk: str) -> str:
        return os.path.join(BIN_DIR, f'{chunk}.lag')


CFG = SetupConfig()
