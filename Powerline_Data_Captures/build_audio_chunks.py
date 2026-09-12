#!/usr/bin/env python3
"""
Concatenate LibriSpeech FLACs into fixed-length WAV chunks.

Scans train-clean-100 recursively, sorts all FLACs, and concatenates
them sequentially into 30-minute WAV files. Picks up where it left off
if interrupted (tracks progress in a state file).

Usage:
    python3 build_audio_chunks.py \
        --flac-root ~/Downloads/LibriSpeech/train-clean-100 \
        --output-dir ~/Documents/power/audio_chunks

Options:
    --flac-root     Root of LibriSpeech split (required)
    --output-dir    Where to save chunk WAV files (required)
    --chunk-mins    Length of each output chunk in minutes (default: 30)
    --sample-rate   Output sample rate in Hz (default: 16000)
    --reset         Ignore existing progress and start from the first FLAC

Output:
    <output-dir>/chunk_001.wav
    <output-dir>/chunk_002.wav
    ...
    <output-dir>/build_progress.json   <- state file for resume

Requirements:
    pip install soundfile numpy
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("[error] soundfile not installed. Run:  pip install soundfile")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_MINS = 30
DEFAULT_SAMP_RATE  = 16_000   # 16 kHz is standard for LibriSpeech / ASR


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_flacs(root):
    files = sorted(root.rglob("*.flac"))
    if not files:
        print(f"[error] No .flac files found under {root}")
        sys.exit(1)
    return files


def load_state(state_path):
    if state_path.exists():
        try:
            s = json.loads(state_path.read_text())
            return s.get("flac_index", 0), s.get("chunk_index", 1)
        except Exception:
            pass
    return 0, 1


def save_state(state_path, flac_index, chunk_index):
    state_path.write_text(json.dumps({
        "flac_index":  flac_index,
        "chunk_index": chunk_index,
    }, indent=2))


def load_flac_mono(path, target_sr):
    """Load a FLAC file, mix to mono, resample to target_sr."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # mix to mono
    data = data.mean(axis=1)
    # resample if needed
    if sr != target_sr:
        from fractions import Fraction
        ratio = Fraction(target_sr, sr).limit_denominator(100)
        up, down = ratio.numerator, ratio.denominator
        # simple linear resample via numpy
        old_len = len(data)
        new_len = int(old_len * target_sr / sr)
        data = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            data
        ).astype(np.float32)
    return data


def progress_bar(current, total, width=40):
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total
    return f"[{bar}] {pct:.1f}%  ({current}/{total} FLACs)"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate LibriSpeech FLACs into 30-minute WAV chunks."
    )
    parser.add_argument("--flac-root",   required=True,
                        help="Root of LibriSpeech split e.g. .../train-clean-100")
    parser.add_argument("--output-dir",  required=True,
                        help="Directory to write chunk WAV files")
    parser.add_argument("--chunk-mins",  type=float, default=DEFAULT_CHUNK_MINS,
                        help=f"Chunk length in minutes (default: {DEFAULT_CHUNK_MINS})")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMP_RATE,
                        help=f"Output sample rate Hz (default: {DEFAULT_SAMP_RATE})")
    parser.add_argument("--reset",       action="store_true",
                        help="Ignore existing progress and start from the first FLAC")
    args = parser.parse_args()

    flac_root  = Path(args.flac_root).expanduser().resolve()
    out_dir    = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path    = out_dir / "build_progress.json"
    chunk_samples = int(args.chunk_mins * 60 * args.sample_rate)

    all_flacs = collect_flacs(flac_root)
    total     = len(all_flacs)

    print(f"\n{'='*62}")
    print(f"  FLAC root    : {flac_root}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Total FLACs  : {total}")
    print(f"  Chunk length : {args.chunk_mins} min  ({chunk_samples:,} samples @ {args.sample_rate} Hz)")
    print(f"  Sample rate  : {args.sample_rate} Hz")
    print(f"{'='*62}\n")

    if args.reset:
        state_path.unlink(missing_ok=True)
        print("[info] Progress reset.\n")

    flac_idx, chunk_idx = load_state(state_path)
    if flac_idx > 0:
        print(f"[info] Resuming from FLAC {flac_idx}/{total}, chunk {chunk_idx}\n")

    # buffer accumulates samples until we have a full chunk
    buffer = np.zeros(0, dtype=np.float32)

    # fast-forward buffer with leftover from previous run
    # (we just start fresh from the saved flac_idx — partial chunk is discarded)

    while flac_idx < total:
        flac_path = all_flacs[flac_idx]

        # ── load and append to buffer ─────────────────────────────────────────
        try:
            audio = load_flac_mono(flac_path, args.sample_rate)
        except Exception as e:
            print(f"\n[warn] Skipping {flac_path.name}: {e}")
            flac_idx += 1
            continue

        buffer = np.concatenate([buffer, audio])

        # ── flush complete chunks from buffer ─────────────────────────────────
        while len(buffer) >= chunk_samples:
            chunk_data = buffer[:chunk_samples]
            buffer     = buffer[chunk_samples:]

            out_path = out_dir / f"chunk_{chunk_idx:03d}.wav"
            sf.write(str(out_path), chunk_data, args.sample_rate, subtype="PCM_16")
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"\n[chunk {chunk_idx:03d}] Saved -> {out_path.name}  ({size_mb:.1f} MB)")

            chunk_idx += 1
            save_state(state_path, flac_idx, chunk_idx)

        # ── progress bar ──────────────────────────────────────────────────────
        flac_idx += 1
        print(f"  {progress_bar(flac_idx, total)}  buf={len(buffer)/args.sample_rate:.1f}s",
              end="\r", flush=True)
        save_state(state_path, flac_idx, chunk_idx)

    # ── flush final partial chunk if any ──────────────────────────────────────
    if len(buffer) > args.sample_rate:  # only save if > 1 second of audio
        out_path = out_dir / f"chunk_{chunk_idx:03d}.wav"
        sf.write(str(out_path), buffer, args.sample_rate, subtype="PCM_16")
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\n[chunk {chunk_idx:03d}] Final partial chunk saved -> {out_path.name}  ({size_mb:.1f} MB)")
        chunk_idx += 1

    print(f"\n\n[done] {chunk_idx - 1} chunk(s) written to {out_dir}")
    state_path.unlink(missing_ok=True)  # clean up state on successful completion


if __name__ == "__main__":
    main()
