"""
prepare_data.py
---------------
Build sharded training data for the Soundbar UNet pipeline.

The prep step is OOM-safe:
- raw powerline `.bin` files are accessed with `np.memmap`
- only the last 30 minutes of each pair are used
- audio is processed in smaller segments
- windows are written to compact train/val shards as they are produced

Default pairing is sequential and end-anchored:
- soundbar chunks are sorted by chunk id
- audio chunks are sorted by chunk id
- the soundbar list is paired with the last N audio chunks

If the numbering is direct, use `--pair_mode direct`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from math import gcd

import numpy as np
import soundfile as sf
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
DEFAULT_TAIL_SEC = 30 * 60
DEFAULT_SEGMENT_SEC = 300.0
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_HOP_SEC = 0.25
DEFAULT_PAIR_MODE = "sequential"
DEFAULT_SHARD_WINDOWS = 512
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


def log_mel_spectrogram(x: np.ndarray) -> np.ndarray:
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


def resolve_pairs(soundbar_files, audio_files, pair_mode: str):
    if pair_mode == "direct":
        audio_map = {idx: path for idx, path in audio_files}
        pairs = []
        for sb_idx, sb_path in soundbar_files:
            audio_path = audio_map.get(sb_idx)
            if audio_path is not None:
                pairs.append((sb_idx, sb_path, sb_idx, audio_path))
        return pairs

    pair_count = min(len(soundbar_files), len(audio_files))
    if pair_count <= 0:
        return []

    sb_slice = soundbar_files[:pair_count]
    audio_slice = audio_files[-pair_count:]
    pairs = []
    for (sb_idx, sb_path), (audio_idx, audio_path) in zip(sb_slice, audio_slice):
        pairs.append((sb_idx, sb_path, audio_idx, audio_path))
    return pairs


def read_soundbar_tail(path: str, tail_sec: float) -> np.ndarray:
    mm = np.memmap(path, dtype=np.float32, mode="r")
    tail_samp = int(tail_sec * FS)
    start = max(0, len(mm) - tail_samp)
    return np.asarray(mm[start:], dtype=np.float32)


def read_audio_tail(path: str, tail_sec: float) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    tail_frames = int(tail_sec * info.samplerate)
    start = max(0, info.frames - tail_frames)
    audio, sr = sf.read(path, start=start, frames=tail_frames, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def prepare_aligned_pair(soundbar_path: str, audio_path: str, tail_sec: float):
    noisy_raw = read_soundbar_tail(soundbar_path, tail_sec).astype(np.float64)
    clean_raw, clean_sr = read_audio_tail(audio_path, tail_sec)
    clean_raw = clean_raw.astype(np.float64)

    if clean_sr != AUDIO_SR:
        clean_raw = resample_to(clean_raw, clean_sr, AUDIO_SR).astype(np.float64)

    noisy_len_sec = len(noisy_raw) / FS
    clean_len_sec = len(clean_raw) / AUDIO_SR
    duration_sec = min(noisy_len_sec, clean_len_sec, tail_sec)

    noisy_raw = noisy_raw[: int(duration_sec * FS)]
    clean_raw = clean_raw[: int(duration_sec * AUDIO_SR)]

    noisy_bp = bandpass_filter(noisy_raw, FS, BP_LOW, BP_HIGH)
    noisy_audio = resample_to(noisy_bp, FS, AUDIO_SR)

    noisy_audio = rms_norm(noisy_audio)
    clean_raw = rms_norm(clean_raw)
    return noisy_audio, clean_raw


def total_window_count(length: int, win_samp: int, hop_samp: int) -> int:
    if length < win_samp:
        return 0
    return ((length - win_samp) // hop_samp) + 1


class ShardWriter:
    def __init__(self, shard_dir: str, split: str, shard_window_limit: int):
        self.shard_dir = shard_dir
        self.split = split
        self.shard_window_limit = shard_window_limit
        self.shard_idx = 0
        self.buffer_noisy = []
        self.buffer_clean = []
        self.total_windows = 0

    def add(self, noisy_spec: np.ndarray, clean_spec: np.ndarray):
        self.buffer_noisy.append(noisy_spec[np.newaxis])
        self.buffer_clean.append(clean_spec[np.newaxis])
        self.total_windows += 1
        if len(self.buffer_noisy) >= self.shard_window_limit:
            self.flush()

    def flush(self):
        if not self.buffer_noisy:
            return None
        noisy = np.stack(self.buffer_noisy).astype(np.float32)
        clean = np.stack(self.buffer_clean).astype(np.float32)
        shard_name = f"{self.split}_{self.shard_idx:06d}.npz"
        shard_path = os.path.join(self.shard_dir, shard_name)
        np.savez_compressed(shard_path, noisy=noisy, clean=clean)
        self.shard_idx += 1
        self.buffer_noisy.clear()
        self.buffer_clean.clear()
        return shard_path


def remove_previous_shards(shard_dir: str):
    if not os.path.isdir(shard_dir):
        return
    for name in os.listdir(shard_dir):
        if name.endswith(".npz") and (name.startswith("train_") or name.startswith("val_")):
            os.remove(os.path.join(shard_dir, name))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    remove_previous_shards(shard_dir)

    soundbar_files = list_chunks(args.soundbar_dir, "bin")
    audio_files = list_chunks(args.audio_dir, "wav")
    if not soundbar_files:
        raise RuntimeError(f"No soundbar .bin files found in {args.soundbar_dir}")
    if not audio_files:
        raise RuntimeError(f"No audio .wav files found in {args.audio_dir}")

    pairs = resolve_pairs(soundbar_files, audio_files, args.pair_mode)
    if not pairs:
        raise RuntimeError("No aligned chunk pairs could be resolved")

    print(f"\nSoundbar dir : {args.soundbar_dir}")
    print(f"Audio dir    : {args.audio_dir}")
    print(f"Output dir   : {args.out_dir}")
    print(f"Pair mode    : {args.pair_mode}")
    print(f"Tail / segment: {args.tail_sec}s / {args.segment_sec}s")
    print(f"Window / hop : {args.win_sec}s / {args.hop_sec}s")
    print(f"Shard size   : {args.shard_windows} windows per file")
    print(f"Bandpass     : {BP_LOW}-{BP_HIGH} Hz")
    print(f"Pairs        : {len(pairs)}\n")

    win_samp = int(args.win_sec * AUDIO_SR)
    hop_samp = int(args.hop_sec * AUDIO_SR)
    segment_samp = int(args.segment_sec * AUDIO_SR)
    segment_step_samp = max(hop_samp, segment_samp - (win_samp - hop_samp))

    train_writer = ShardWriter(shard_dir, "train", args.shard_windows)
    val_writer = ShardWriter(shard_dir, "val", args.shard_windows)

    manifest_rows = []
    n_train_total = 0
    n_val_total = 0
    total_pairs = 0

    for sb_idx, sb_path, audio_idx, audio_path in pairs:
        t0 = time.time()
        noisy_audio, clean_audio = prepare_aligned_pair(sb_path, audio_path, args.tail_sec)
        min_len = min(len(noisy_audio), len(clean_audio))
        if min_len < win_samp:
            print(f"  SKIP chunk_{sb_idx:03d}: too short after alignment")
            continue

        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]

        total_windows = total_window_count(min_len, win_samp, hop_samp)
        if total_windows == 0:
            print(f"  SKIP chunk_{sb_idx:03d}: no windows")
            continue

        split_idx = int(0.9 * total_windows)
        windows_written = 0
        train_windows = 0
        val_windows = 0
        global_window_idx = 0

        seg_start = 0
        while seg_start < min_len:
            seg_end = min(seg_start + segment_samp, min_len)
            noisy_seg = noisy_audio[seg_start:seg_end]
            clean_seg = clean_audio[seg_start:seg_end]
            if len(noisy_seg) < win_samp:
                break

            if seg_end < min_len:
                max_start = min(len(noisy_seg) - win_samp, segment_step_samp - 1)
            else:
                max_start = len(noisy_seg) - win_samp

            if max_start < 0:
                break

            for start in range(0, max_start + 1, hop_samp):
                end = start + win_samp
                noisy_spec = log_mel_spectrogram(noisy_seg[start:end])
                clean_spec = log_mel_spectrogram(clean_seg[start:end])

                if global_window_idx < split_idx:
                    train_writer.add(noisy_spec, clean_spec)
                    train_windows += 1
                    n_train_total += 1
                else:
                    val_writer.add(noisy_spec, clean_spec)
                    val_windows += 1
                    n_val_total += 1

                windows_written += 1
                global_window_idx += 1

                if args.max_windows_per_pair > 0 and windows_written >= args.max_windows_per_pair:
                    break

            if args.max_windows_per_pair > 0 and windows_written >= args.max_windows_per_pair:
                break

            seg_start += segment_step_samp

        total_pairs += 1
        manifest_rows.append(
            {
                "soundbar_chunk": sb_idx,
                "audio_chunk": audio_idx,
                "windows": windows_written,
                "train_windows": train_windows,
                "val_windows": val_windows,
                "duration_sec": round(min_len / AUDIO_SR, 3),
            }
        )
        print(
            f"  chunk_{sb_idx:03d}.bin -> chunk_{audio_idx:03d}.wav"
            f"  | windows={windows_written}  | {time.time() - t0:.1f}s"
        )

    train_writer.flush()
    val_writer.flush()

    if not manifest_rows:
        raise RuntimeError("No aligned pairs were written. Check alignment and file availability.")

    if n_train_total == 0 or n_val_total == 0:
        raise RuntimeError("No train/val samples were written. Check alignment and window settings.")

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
        "pair_mode": args.pair_mode,
        "tail_sec": args.tail_sec,
        "segment_sec": args.segment_sec,
        "window_sec": args.win_sec,
        "hop_sec": args.hop_sec,
        "bandpass_hz": [BP_LOW, BP_HIGH],
        "fs_soundbar": FS,
        "fs_audio": AUDIO_SR,
        "n_pairs": total_pairs,
        "n_train": int(n_train_total),
        "n_val": int(n_val_total),
        "spec_shape": [1, N_MELS, 98],
        "shard_dir": shard_dir,
        "shard_windows": args.shard_windows,
        "max_windows_per_pair": args.max_windows_per_pair,
    }

    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(args.out_dir, "alignment_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved data    : {out_file}")
    print(f"Saved meta    : {os.path.join(args.out_dir, 'metadata.json')}")
    print(f"Saved manifest: {os.path.join(args.out_dir, 'alignment_manifest.csv')}")
    print(f"Train samples : {n_train_total:,}")
    print(f"Val samples   : {n_val_total:,}")
    print(f"Shards        : {shard_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundbar_dir", default=SOUNDBAR_DIR)
    parser.add_argument("--audio_dir", default=AUDIO_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_name", default="train_data.npz")
    parser.add_argument("--pair_mode", choices=["sequential", "direct"], default=DEFAULT_PAIR_MODE)
    parser.add_argument("--tail_sec", type=float, default=DEFAULT_TAIL_SEC)
    parser.add_argument("--segment_sec", type=float, default=DEFAULT_SEGMENT_SEC)
    parser.add_argument("--win_sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--hop_sec", type=float, default=DEFAULT_HOP_SEC)
    parser.add_argument("--shard_windows", type=int, default=DEFAULT_SHARD_WINDOWS)
    parser.add_argument("--max_windows_per_pair", type=int, default=DEFAULT_MAX_WINDOWS_PER_PAIR)
    args = parser.parse_args()
    main(args)