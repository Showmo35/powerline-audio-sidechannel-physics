"""
prepare_data.py
---------------
Soundbar adaptation of Module 4 data preparation.

Builds paired (soundbar leakage, clean audio) mel-spectrogram windows from:
- Soundbar/LibriSpeech/chunk_*.bin
- audio_chunks/chunk_*.wav

The mapping is sequential and anchored so the last full audio chunk is aligned
with the last soundbar chunk, per SOUNDBAR_ALIGNMENT_SUMMARY.md.

Usage:
    python prepare_data.py
    python prepare_data.py --out_dir data --out_name train_data.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from math import gcd

import librosa
import numpy as np
import torch
import whisper
from scipy.signal import butter, resample_poly, sosfiltfilt

SCRATCH = "<REPO_ROOT>"
CAPTURE_ROOT = os.path.join(SCRATCH, "Powerline_Data_Captures")
SOUNDBAR_DIR = os.path.join(CAPTURE_ROOT, "Soundbar", "LibriSpeech")
AUDIO_DIR = os.path.join(CAPTURE_ROOT, "audio_chunks")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FS = 200_000
AUDIO_SR = 16_000
BP_LOW = 50
BP_HIGH = 4000
N_FFT = 400
HOP = 160
N_MELS = 80
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_HOP_SEC = 0.25
DEFAULT_ANCHOR_AUDIO_IDX = 201
DEFAULT_EXCLUDE = {24, 34, 55, 56}
DEFAULT_MAX_WINDOWS_PER_PAIR = 0

_WHISPER_FB = None


def bandpass_filter(x: np.ndarray, fs: int, lo: int, hi: int, order: int = 5) -> np.ndarray:
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x: np.ndarray, target: float = 0.05) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


def _get_whisper_filterbank() -> np.ndarray:
    global _WHISPER_FB
    if _WHISPER_FB is None:
        _WHISPER_FB = whisper.audio.mel_filters(torch.device("cpu"), N_MELS).numpy()
    return _WHISPER_FB


def log_mel_spectrogram(x: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    fb = _get_whisper_filterbank()
    win = np.hanning(N_FFT)
    nf = (len(x) - N_FFT) // HOP + 1
    if nf <= 0:
        raise ValueError("Input segment is too short for the mel spectrogram window")
    frames = np.stack([x[i * HOP : i * HOP + N_FFT] * win for i in range(nf)])
    mag_sq = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    mel = fb @ mag_sq.T
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def list_chunks(folder: str, ext: str):
    pattern = re.compile(rf"chunk_(\d{{3}})\.{re.escape(ext)}$")
    out = []
    for name in os.listdir(folder):
        match = pattern.match(name)
        if match:
            out.append((int(match.group(1)), os.path.join(folder, name)))
    return sorted(out)


def align_audio_index(soundbar_idx: int, soundbar_max_idx: int, anchor_audio_idx: int) -> int:
    # Direct 1:1 mapping: chunk_N.bin pairs with chunk_N.wav.
    return soundbar_idx


def read_soundbar_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def read_audio_wav(path: str) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio.astype(np.float64), int(sr)


def make_windows(
    noisy: np.ndarray,
    clean: np.ndarray,
    win_sec: float,
    hop_sec: float,
    max_windows: int,
):
    win_samp = int(win_sec * AUDIO_SR)
    hop_samp = int(hop_sec * AUDIO_SR)
    if len(noisy) < win_samp or len(clean) < win_samp:
        return [], []

    starts = list(range(0, min(len(noisy), len(clean)) - win_samp + 1, hop_samp))
    if max_windows > 0 and len(starts) > max_windows:
        pick = np.linspace(0, len(starts) - 1, max_windows, dtype=int)
        starts = [starts[i] for i in pick]

    noisy_list, clean_list = [], []
    for start in starts:
        end = start + win_samp
        noisy_list.append(log_mel_spectrogram(noisy[start:end])[np.newaxis])
        clean_list.append(log_mel_spectrogram(clean[start:end])[np.newaxis])
    return noisy_list, clean_list


def process_pair(
    soundbar_path: str,
    audio_path: str,
    win_sec: float,
    hop_sec: float,
    max_windows: int,
):
    noisy_raw = read_soundbar_bin(soundbar_path)
    clean_raw, clean_sr = read_audio_wav(audio_path)

    if clean_sr != AUDIO_SR:
        clean_raw = resample_to(clean_raw, clean_sr, AUDIO_SR)

    # Match durations before filtering/resampling.
    duration = min(len(noisy_raw) / FS, len(clean_raw) / AUDIO_SR)
    noisy_raw = noisy_raw[: int(duration * FS)]
    clean_raw = clean_raw[: int(duration * AUDIO_SR)]

    noisy_bp = bandpass_filter(noisy_raw, FS, BP_LOW, BP_HIGH)
    noisy_audio = resample_to(noisy_bp, FS, AUDIO_SR)

    noisy_audio = rms_norm(noisy_audio)
    clean_raw = rms_norm(clean_raw)

    noisy_list, clean_list = make_windows(
        noisy_audio, clean_raw, win_sec, hop_sec, max_windows
    )
    if not noisy_list:
        return None, None

    return (
        np.stack(noisy_list).astype(np.float32),
        np.stack(clean_list).astype(np.float32),
    )


def prepare_aligned_audio(soundbar_path: str, audio_path: str):
    """Load, align, filter, resample, and normalise one pair into audio-rate arrays."""
    noisy_raw = read_soundbar_bin(soundbar_path)
    clean_raw, clean_sr = read_audio_wav(audio_path)

    if clean_sr != AUDIO_SR:
        clean_raw = resample_to(clean_raw, clean_sr, AUDIO_SR)

    # Match durations before filtering/resampling.
    duration = min(len(noisy_raw) / FS, len(clean_raw) / AUDIO_SR)
    noisy_raw = noisy_raw[: int(duration * FS)]
    clean_raw = clean_raw[: int(duration * AUDIO_SR)]

    noisy_bp = bandpass_filter(noisy_raw, FS, BP_LOW, BP_HIGH)
    noisy_audio = resample_to(noisy_bp, FS, AUDIO_SR)

    noisy_audio = rms_norm(noisy_audio)
    clean_raw = rms_norm(clean_raw)
    return noisy_audio, clean_raw


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "alignment_manifest.csv")
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    # Clear existing shards so each run is self-consistent.
    for name in os.listdir(shard_dir):
        if name.startswith("train_") or name.startswith("val_"):
            if name.endswith(".npz"):
                os.remove(os.path.join(shard_dir, name))

    sb_files = list_chunks(args.soundbar_dir, "bin")
    audio_files = list_chunks(args.audio_dir, "wav")
    if not sb_files:
        raise RuntimeError(f"No soundbar .bin files found in {args.soundbar_dir}")
    if not audio_files:
        raise RuntimeError(f"No audio .wav files found in {args.audio_dir}")

    sb_indices = [idx for idx, _ in sb_files]
    audio_indices = [idx for idx, _ in audio_files]
    sb_max = max(sb_indices)
    audio_max = max(audio_indices)

    exclude = set(args.exclude)
    manifest_rows = []
    n_train_total = 0
    n_val_total = 0
    train_shard_idx = 0
    val_shard_idx = 0

    print(f"\nSoundbar dir : {args.soundbar_dir}")
    print(f"Audio dir    : {args.audio_dir}")
    print(f"Output dir   : {args.out_dir}")
    print(f"Window / Hop : {args.win_sec}s / {args.hop_sec}s")
    max_windows_desc = "ALL" if args.max_windows_per_pair <= 0 else str(args.max_windows_per_pair)
    print(f"Max windows  : {max_windows_desc} per aligned pair")
    print(f"Bandpass     : {BP_LOW}-{BP_HIGH} Hz")
    print(f"Anchor audio : {args.anchor_audio_idx}")
    print(f"Excluded bins: {sorted(exclude)}\n")
    print(f"Segment size : {args.segment_sec}s\n")

    segment_samp = int(args.segment_sec * AUDIO_SR) if args.segment_sec > 0 else 0

    for sb_idx, sb_path in sb_files:
        if sb_idx in exclude:
            continue

        audio_idx = align_audio_index(sb_idx, sb_max, args.anchor_audio_idx)
        if audio_idx < 1 or audio_idx > audio_max:
            continue

        audio_path = os.path.join(args.audio_dir, f"chunk_{audio_idx:03d}.wav")
        if not os.path.exists(audio_path):
            continue

        t0 = time.time()
        noisy_audio, clean_audio = prepare_aligned_audio(sb_path, audio_path)
        min_len = min(len(noisy_audio), len(clean_audio))
        if min_len <= 0:
            print(f"  SKIP chunk_{sb_idx:03d}: too short after alignment")
            continue

        if segment_samp <= 0:
            segment_samp = min_len

        pair_windows = 0
        pair_train = 0
        pair_val = 0
        remaining_cap = args.max_windows_per_pair

        for start in range(0, min_len, segment_samp):
            end = min(start + segment_samp, min_len)
            noisy_seg = noisy_audio[start:end]
            clean_seg = clean_audio[start:end]

            cap_this = remaining_cap if remaining_cap > 0 else 0
            noisy_list, clean_list = make_windows(
                noisy_seg, clean_seg, args.win_sec, args.hop_sec, cap_this
            )
            if not noisy_list:
                continue

            noisy = np.stack(noisy_list).astype(np.float32)
            clean = np.stack(clean_list).astype(np.float32)
            n_win = len(noisy)
            pair_windows += n_win

            if remaining_cap > 0:
                remaining_cap -= n_win

            split_idx = int(0.9 * n_win)
            noisy_train = noisy[:split_idx]
            clean_train = clean[:split_idx]
            noisy_val = noisy[split_idx:]
            clean_val = clean[split_idx:]

            if len(noisy_train) > 0:
                np.savez_compressed(
                    os.path.join(shard_dir, f"train_{train_shard_idx:06d}.npz"),
                    noisy=noisy_train,
                    clean=clean_train,
                )
                pair_train += len(noisy_train)
                n_train_total += len(noisy_train)
                train_shard_idx += 1

            if len(noisy_val) > 0:
                np.savez_compressed(
                    os.path.join(shard_dir, f"val_{val_shard_idx:06d}.npz"),
                    noisy=noisy_val,
                    clean=clean_val,
                )
                pair_val += len(noisy_val)
                n_val_total += len(noisy_val)
                val_shard_idx += 1

            del noisy, clean, noisy_train, clean_train, noisy_val, clean_val

            if remaining_cap == 0 and args.max_windows_per_pair > 0:
                break

        if pair_windows == 0:
            print(f"  SKIP chunk_{sb_idx:03d}: too short after alignment")
            continue

        manifest_rows.append({
            "soundbar_chunk": sb_idx,
            "audio_chunk": audio_idx,
            "n_windows": pair_windows,
            "train_windows": pair_train,
            "val_windows": pair_val,
        })
        print(f"  chunk_{sb_idx:03d}.bin -> chunk_{audio_idx:03d}.wav  | windows={pair_windows}  | {time.time()-t0:.1f}s")

    if n_train_total == 0:
        raise RuntimeError("No aligned pairs were collected. Check paths and exclusions.")

    out_file = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_file,
        format=np.array("sharded"),
        shard_dir=np.array(shard_dir),
        n_train=np.array(n_train_total),
        n_val=np.array(n_val_total),
    )

    metadata = {
        "soundbar_dir": args.soundbar_dir,
        "audio_dir": args.audio_dir,
        "anchor_audio_idx": args.anchor_audio_idx,
        "exclude": sorted(exclude),
        "soundbar_max_idx": sb_max,
        "audio_max_idx": audio_max,
        "window_sec": args.win_sec,
        "hop_sec": args.hop_sec,
        "bandpass_hz": [BP_LOW, BP_HIGH],
        "fs_soundbar": FS,
        "fs_audio": AUDIO_SR,
        "n_train": int(n_train_total),
        "n_val": int(n_val_total),
        "spec_shape": [1, N_MELS, 98],
        "data_format": "sharded",
        "segment_sec": args.segment_sec,
        "shard_dir": shard_dir,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["soundbar_chunk", "audio_chunk", "n_windows", "train_windows", "val_windows"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved data   : {out_file}")
    print(f"Saved meta   : {os.path.join(args.out_dir, 'metadata.json')}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved shards : {shard_dir}")
    print(f"Train samples: {n_train_total:,}")
    print(f"Val samples  : {n_val_total:,}")
    print(f"Spec shape   : {(1, N_MELS, 98)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Soundbar leakage data for Module 4")
    parser.add_argument("--soundbar_dir", default=SOUNDBAR_DIR)
    parser.add_argument("--audio_dir", default=AUDIO_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_name", default="train_data.npz")
    parser.add_argument("--win_sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--hop_sec", type=float, default=DEFAULT_HOP_SEC)
    parser.add_argument("--anchor_audio_idx", type=int, default=DEFAULT_ANCHOR_AUDIO_IDX)
    parser.add_argument("--exclude", type=int, nargs="*", default=sorted(DEFAULT_EXCLUDE))
    parser.add_argument(
        "--max_windows_per_pair",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_PAIR,
        help="Cap windows kept from each aligned chunk pair (<=0 keeps all windows)",
    )
    parser.add_argument(
        "--segment_sec",
        type=float,
        default=300.0,
        help="Process each aligned pair in fixed-length segments (seconds)",
    )
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
