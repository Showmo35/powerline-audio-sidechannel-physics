#!/usr/bin/env python3
"""
Play audio chunks sequentially and record USRP capture for each one.

For each chunk_XXX.wav in the chunks directory:
  1. GNURadio starts  (USRP capture begins)
  2. Wait GAP seconds (default: 3)
  3. Play chunk_XXX.wav through laptop speakers (always via HDMI)
  4. Stop GNURadio when playback ends
  5. Move .bin to output dir as chunk_XXX.bin
  6. Move to next chunk

Output .bin files are named to match the chunk WAVs:
    chunk_001.wav  ->  chunk_001.bin
    chunk_002.wav  ->  chunk_002.bin
    ...

Progress is saved in capture_progress.json so runs can be resumed.

Usage:
    python3 play_and_capture.py \
        --chunks-dir ~/Documents/power/audio_chunks \
        --output-dir ~/Documents/power/bin_captures

Options:
    --chunks-dir    Directory containing chunk_XXX.wav files (required)
    --output-dir    Directory to save chunk_XXX.bin files (required)
    --grc           Path to simple_rx.grc (default: next to this script)
    --gap           Seconds between GRC start and playback start (default: 3)
    --sink          PulseAudio sink name for audio output
                    (default: alsa_output.pci-0000_00_1f.3.hdmi-stereo)
    --reset         Restart from chunk_001 ignoring saved progress

Requirements:
    sudo apt install ffmpeg
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
GRC_FILE           = Path(__file__).parent / "simple_rx.grc"
DEFAULT_GAP        = 3
DEFAULT_SINK       = "alsa_output.pci-0000_00_1f.3.hdmi-stereo"
EXPECTED_SAMP_RATE = 200_000

# Must match the file sink path in simple_rx.grc
BIN_SRC = Path("/home/user/Documents/power/soundbar_capture_1.bin")


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_chunks(chunks_dir):
    files = sorted(chunks_dir.glob("chunk_*.wav"))
    if not files:
        print(f"[error] No chunk_*.wav files found in {chunks_dir}")
        sys.exit(1)
    return files


def load_progress(state_path):
    if state_path.exists():
        try:
            return int(json.loads(state_path.read_text()).get("next_index", 0))
        except Exception:
            pass
    return 0


def save_progress(state_path, index):
    state_path.write_text(json.dumps({"next_index": index}, indent=2))


def compile_grc(grc_path):
    py_path = grc_path.with_suffix(".py")
    if py_path.exists():
        print(f"[info] Using pre-compiled flowgraph: {py_path.name}")
        return py_path
    print(f"[info] Compiling {grc_path.name} with grcc ...")
    r = subprocess.run(["grcc", str(grc_path), "-o", str(grc_path.parent)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("[error] grcc failed:\n" + r.stderr)
        sys.exit(1)
    print(f"[info] Compiled -> {py_path.name}")
    return py_path


def verify_samp_rate(py_path, expected=EXPECTED_SAMP_RATE):
    src = py_path.read_text()
    m = re.search(r"samp_rate\s*=\s*([0-9e.+]+)", src)
    if m:
        actual = int(float(m.group(1)))
        if actual != expected:
            print(f"[warn] GRC samp_rate is {actual} Hz (expected {expected} Hz) — continuing anyway.")
        else:
            print(f"[info] samp_rate confirmed: {actual} Hz")
    else:
        print("[warn] Could not verify samp_rate.")


def stop_process(proc, name, timeout=5):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def move_bin(chunk_stem, out_dir):
    for _ in range(10):
        if BIN_SRC.exists() and BIN_SRC.stat().st_size > 0:
            break
        time.sleep(0.5)
    if BIN_SRC.exists() and BIN_SRC.stat().st_size > 0:
        dst = out_dir / f"{chunk_stem}.bin"
        shutil.move(str(BIN_SRC), dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"  [info] Saved -> {dst.name}  ({size_mb:.1f} MB)")
        return str(dst)
    print(f"  [warn] USRP bin not found or empty: {BIN_SRC}")
    return None


def get_wav_duration(wav_path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Play audio chunks and capture USRP signal for each one."
    )
    parser.add_argument("--chunks-dir", required=True,
                        help="Directory containing chunk_XXX.wav files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save chunk_XXX.bin files")
    parser.add_argument("--grc", default=str(GRC_FILE),
                        help="Path to simple_rx.grc (default: next to this script)")
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP,
                        help=f"Seconds between GRC start and playback (default: {DEFAULT_GAP})")
    parser.add_argument("--sink", default=DEFAULT_SINK,
                        help=f"PulseAudio sink for audio output (default: {DEFAULT_SINK})")
    parser.add_argument("--reset", action="store_true",
                        help="Restart from chunk_001 ignoring saved progress")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir).expanduser().resolve()
    out_dir    = Path(args.output_dir).expanduser().resolve()
    grc_path   = Path(args.grc).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "capture_progress.json"

    if not grc_path.exists():
        print(f"[error] GRC file not found: {grc_path}")
        sys.exit(1)
    if not shutil.which("ffplay"):
        print("[error] ffplay not found. Install:  sudo apt install ffmpeg")
        sys.exit(1)

    py_path = compile_grc(grc_path)
    verify_samp_rate(py_path)

    all_chunks = collect_chunks(chunks_dir)
    total      = len(all_chunks)

    if args.reset:
        save_progress(state_path, 0)
        print("[info] Progress reset to chunk 0.\n")

    start_idx = load_progress(state_path)
    if start_idx >= total:
        print("[info] All chunks already captured. Use --reset to redo.")
        sys.exit(0)

    # Build environment with PULSE_SINK locked to HDMI
    # This ensures ffplay always outputs to HDMI regardless of system default
    audio_env = {**os.environ, "PULSE_SINK": args.sink}

    print(f"\n{'='*62}")
    print(f"  Chunks dir   : {chunks_dir}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Total chunks : {total}")
    print(f"  Starting at  : chunk {start_idx + 1}/{total}")
    print(f"  Gap          : {args.gap}s")
    print(f"  Audio sink   : {args.sink}")
    print(f"  GRC bin src  : {BIN_SRC}")
    print(f"{'='*62}\n")

    for idx in range(start_idx, total):
        chunk_path = all_chunks[idx]
        chunk_stem = chunk_path.stem   # e.g. "chunk_001"
        dst_bin    = out_dir / f"{chunk_stem}.bin"

        # skip already captured
        if dst_bin.exists():
            print(f"[skip] {chunk_stem}.bin already exists.")
            save_progress(state_path, idx + 1)
            continue

        duration = get_wav_duration(chunk_path)
        dur_str  = f"{duration/60:.1f} min" if duration else "unknown"

        print(f"\n── Chunk {idx+1}/{total}  ({chunk_stem})  {dur_str} ──")

        grc_proc   = None
        audio_proc = None

        def on_sigint(signum, frame):
            print(f"\n[info] Ctrl+C — saving progress at chunk {idx+1} ...")
            stop_process(audio_proc, "ffplay")
            stop_process(grc_proc,   "GNURadio")
            move_bin(chunk_stem, out_dir)
            save_progress(state_path, idx)   # retry this chunk next run
            print(f"[info] Resume from: {chunk_stem}")
            sys.exit(0)

        signal.signal(signal.SIGINT,  on_sigint)
        signal.signal(signal.SIGTERM, on_sigint)

        # ── 1. start GNURadio ─────────────────────────────────────────────────
        grc_proc = subprocess.Popen(
            [sys.executable, str(py_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"  [1/3] GNURadio started  (PID {grc_proc.pid})")

        # ── 2. gap ────────────────────────────────────────────────────────────
        print(f"  [2/3] Waiting {args.gap}s ...")
        time.sleep(args.gap)

        if grc_proc.poll() is not None:
            err = grc_proc.stderr.read().decode(errors="replace")
            print(f"  [error] GNURadio crashed during gap:\n{err}")
            save_progress(state_path, idx)
            sys.exit(1)

        # ── 3. play chunk via HDMI ────────────────────────────────────────────
        print(f"  [3/3] Playing {chunk_path.name} -> {args.sink}")
        audio_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(chunk_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=audio_env,   # PULSE_SINK locks output to HDMI
        )

        # ── wait for playback ─────────────────────────────────────────────────
        start = time.time()
        while True:
            if audio_proc.poll() is not None:
                elapsed = time.time() - start
                print(f"\n  [info] Playback finished ({elapsed:.1f}s)")
                break

            if grc_proc.poll() is not None:
                err = grc_proc.stderr.read().decode(errors="replace")
                print(f"\n  [error] GNURadio crashed during playback:\n{err}")
                stop_process(audio_proc, "ffplay")
                move_bin(chunk_stem, out_dir)
                save_progress(state_path, idx)
                sys.exit(1)

            elapsed = time.time() - start
            if duration:
                pct = min(100.0, 100.0 * elapsed / duration)
                filled = int(pct / 2)
                bar = "█" * filled + "░" * (50 - filled)
                print(f"  [{bar}] {pct:.0f}%  ({elapsed:.0f}/{duration:.0f}s)",
                      end="\r", flush=True)
            time.sleep(0.5)

        # ── stop GNURadio and save bin ────────────────────────────────────────
        stop_process(grc_proc, "GNURadio")
        move_bin(chunk_stem, out_dir)
        save_progress(state_path, idx + 1)

        remaining = total - (idx + 1)
        print(f"  [info] {idx+1}/{total} done  |  {remaining} remaining\n")

    print(f"\n[done] All {total} chunks captured.")
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
