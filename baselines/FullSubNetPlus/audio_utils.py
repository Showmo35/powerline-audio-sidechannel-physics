import os
from math import gcd

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

from config import (
    AUDIO_SR,
    BP_HIGH,
    BP_LOW,
    CHAPTER_TIMING,
    FS_CAPTURE,
    MP3_DIR,
)


def match_chapter_name(file_id: str):
    fid = file_id.lower()
    for chapter_name in CHAPTER_TIMING:
        num = chapter_name.split('_')[1]
        chapter_aliases = {
            f'chapter_{num}',
            f'chap_{num}',
            f'chap_{int(num)}',
            f'chap{int(num)}',
        }
        if any(alias in fid for alias in chapter_aliases):
            return chapter_name
    return None


def bandpass_filter(x: np.ndarray, fs: int, lo: float, hi: float, order: int = 5):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def lowpass_filter(x: np.ndarray, fs: int, cutoff: float, order: int = 5):
    sos = butter(order, cutoff / (fs / 2), btype='low', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def highpass_filter(x: np.ndarray, fs: int, cutoff: float, order: int = 4):
    sos = butter(order, cutoff / (fs / 2), btype='high', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def iq_demodulate(sig: np.ndarray, fs: int, fc: float, audio_bw: float = 8000):
    t = np.arange(len(sig), dtype=np.float64) / fs
    sig = sig.astype(np.float64)
    i_sig = lowpass_filter(sig * np.cos(2 * np.pi * fc * t), fs, audio_bw)
    q_sig = lowpass_filter(sig * np.sin(2 * np.pi * fc * t), fs, audio_bw)
    env = np.abs(i_sig + 1j * q_sig).astype(np.float32)
    return highpass_filter(env, fs, cutoff=30.0)


def resample_to(x: np.ndarray, fs_in: int, fs_out: int):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x: np.ndarray, target_rms: float = 0.05):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x * (target_rms / rms)).astype(np.float32)


def preprocess_noisy_waveform(
    pl_raw: np.ndarray,
    preprocess_mode: str = 'bandpass',
    carrier_hz: float = 20433.35,
):
    if preprocess_mode == 'bandpass':
        noisy = bandpass_filter(pl_raw, FS_CAPTURE, BP_LOW, BP_HIGH)
    elif preprocess_mode == 'iq':
        noisy = iq_demodulate(pl_raw, FS_CAPTURE, carrier_hz)
    else:
        raise ValueError(f'Unsupported preprocess_mode={preprocess_mode}')

    noisy = resample_to(noisy, FS_CAPTURE, AUDIO_SR)
    return noisy


def load_aligned_noisy_clean(
    pl_path: str,
    chapter_name: str,
    preprocess_mode: str = 'bandpass',
    carrier_hz: float = 20433.35,
):
    timing = CHAPTER_TIMING[chapter_name]
    offset_sec = timing['start']
    mp3_duration = timing['mp3_duration']

    pl_raw = np.fromfile(pl_path, dtype=np.float32).astype(np.float64)
    start = int(offset_sec * FS_CAPTURE)
    length = int(mp3_duration * FS_CAPTURE)
    pl_raw = pl_raw[start:start + length]

    noisy = preprocess_noisy_waveform(
        pl_raw,
        preprocess_mode=preprocess_mode,
        carrier_hz=carrier_hz,
    )

    ch_num = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f'Alice_In_Wonderland_ch_{ch_num}.mp3')
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f'Missing MP3 for {chapter_name}: {mp3_path}')

    clean, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        clean = resample_to(clean, mp3_sr, AUDIO_SR)

    common = min(len(noisy), len(clean))
    noisy = noisy[:common]
    clean = clean[:common]

    noisy = rms_norm(noisy)
    clean = rms_norm(clean)
    return noisy, clean, mp3_path


def temporal_partition_indices(
    n_samples: int,
    sr: int,
    train_ratio: float,
    guard_sec: float,
    window_sec: float,
):
    win = int(window_sec * sr)
    guard = int(guard_sec * sr)
    split = int(n_samples * train_ratio)

    train_end = max(win, split - guard)
    val_start = min(n_samples - win, split + guard)

    return train_end, val_start


def make_window_starts(start: int, end: int, win: int, hop: int):
    if end - start < win:
        return np.array([], dtype=np.int64)
    return np.arange(start, end - win + 1, hop, dtype=np.int64)


def mel_spectrogram_from_waveform(
    wave: np.ndarray,
    sr: int = AUDIO_SR,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    power: float = 2.0,
):
    mel = librosa.feature.melspectrogram(
        y=wave,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann',
        center=False,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )
    return mel.astype(np.float32)
