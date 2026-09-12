"""
prepare_data_alice.py
---------------------
Build training data for Module4_UNet_Soundbar using the May29_Alice
powerline captures and Alice_In_Wonderland MP3 ground truth.

Per-chapter timing offsets are used for precise alignment — each .bin file
has a known startup delay before the audio begins.

Output: dense npz with noisy_train / clean_train / noisy_val / clean_val
        compatible with Module4's dataset.py and inference.py.

Usage:
    python prepare_data_alice.py
    python prepare_data_alice.py --split temporal --out_name train_data_alice_temporal.npz
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import torch
import whisper
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH      = '<REPO_ROOT>'
MAY29_DIR    = os.path.join(SCRATCH, 'May29_Alice')
MP3_DIR      = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
DEFAULT_OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ── Per-chapter timing offsets (seconds from start of .bin to audio start) ────
CHAPTER_TIMING = {
    'chapter_01': {'start': 1.38,  'mp3_duration': 631.95},
    'chapter_02': {'start': 2.04,  'mp3_duration': 724.14},
    'chapter_04': {'start': 1.95,  'mp3_duration': 1168.38},
    'chapter_05': {'start': 2.17,  'mp3_duration': 794.54},
    'chapter_06': {'start': 2.09,  'mp3_duration': 765.18},
    'chapter_07': {'start': 2.05,  'mp3_duration': 1027.74},
    'chapter_08': {'start': 1.95,  'mp3_duration': 789.39},
    'chapter_10': {'start': 1.63,  'mp3_duration': 1345.72},
    'chapter_11': {'start': 1.61,  'mp3_duration': 601.23},
}

# ── Signal parameters ─────────────────────────────────────────────────────────
FS       = 200_000   # SDR capture sample rate (Hz)
AUDIO_SR = 16_000    # Whisper sample rate (Hz)
BP_LOW   = 50        # Bandpass low cutoff (Hz)
BP_HIGH  = 4000      # Bandpass high cutoff (Hz)

# ── Mel spectrogram parameters ────────────────────────────────────────────────
N_FFT  = 400
HOP    = 160
N_MELS = 80

_WHISPER_FB = None


def _get_whisper_filterbank():
    global _WHISPER_FB
    if _WHISPER_FB is None:
        _WHISPER_FB = whisper.audio.mel_filters(torch.device('cpu'), N_MELS).numpy()
    return _WHISPER_FB


def bandpass_filter(x, fs, lo, hi, order=5):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x * target / rms).astype(np.float32)


def log_mel_spectrogram(x):
    fb  = _get_whisper_filterbank()
    win = np.hanning(N_FFT)
    nf  = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * win for i in range(nf)])
    mag_sq = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    mel    = fb @ mag_sq.T
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def match_chapter_name(file_id):
    fid = file_id.lower()
    for ch in CHAPTER_TIMING:
        num = ch.split('_')[1]
        if (f'chapter_{num}' in fid or f'chap_{num}' in fid
                or f'chap_{int(num)}' in fid or f'chap{int(num)}' in fid):
            return ch
    return None


def process_file(pl_path, chapter_name, win_sec, hop_sec):
    timing       = CHAPTER_TIMING[chapter_name]
    offset_sec   = timing['start']
    mp3_duration = timing['mp3_duration']

    print(f"  Loading {os.path.basename(pl_path)} ...")
    pl_raw = np.fromfile(pl_path, dtype=np.float32).astype(np.float64)

    start_samp = int(offset_sec * FS)
    chap_samps = int(mp3_duration * FS)
    pl_raw = pl_raw[start_samp:start_samp + chap_samps]
    print(f"    Chapter: {chapter_name} | offset {offset_sec}s | duration {len(pl_raw)/FS:.1f}s")

    pl_bp    = bandpass_filter(pl_raw, FS, BP_LOW, BP_HIGH)
    pl_audio = resample_to(pl_bp, FS, AUDIO_SR)

    ch_num   = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f"Alice_In_Wonderland_ch_{ch_num}.mp3")
    if not os.path.exists(mp3_path):
        print(f"    WARNING: MP3 not found: {mp3_path} -- skipping")
        return None, None

    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

    L         = min(len(pl_audio), len(mp3_audio))
    pl_audio  = rms_norm(pl_audio[:L])
    mp3_audio = rms_norm(mp3_audio[:L])

    win_samp = int(win_sec * AUDIO_SR)
    hop_samp = int(hop_sec * AUDIO_SR)

    noisy_list, clean_list = [], []
    for s in range(0, L - win_samp, hop_samp):
        e = s + win_samp
        noisy_list.append(log_mel_spectrogram(pl_audio[s:e])[np.newaxis])
        clean_list.append(log_mel_spectrogram(mp3_audio[s:e])[np.newaxis])

    print(f"    Windows: {len(noisy_list)}  spec shape (1,{N_MELS},{noisy_list[0].shape[-1]})")
    return (np.stack(noisy_list).astype(np.float32),
            np.stack(clean_list).astype(np.float32))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nOutput dir : {args.out_dir}")
    print(f"Window     : {args.win_sec}s  Hop: {args.hop_sec}s")
    print(f"Bandpass   : {BP_LOW}-{BP_HIGH} Hz")
    print(f"Split      : {args.split}\n")

    pl_files = sorted(glob.glob(os.path.join(MAY29_DIR, '*_real.bin')))
    print(f"Found {len(pl_files)} powerline files in {MAY29_DIR}\n")

    all_train_noisy, all_train_clean = [], []
    all_val_noisy,   all_val_clean   = [], []

    for pl_path in pl_files:
        file_id      = os.path.basename(pl_path).replace('_real.bin', '')
        chapter_name = match_chapter_name(file_id)
        if chapter_name is None:
            print(f"  SKIP {file_id}: no chapter timing found")
            continue

        t0 = time.time()
        noisy, clean = process_file(pl_path, chapter_name, args.win_sec, args.hop_sec)
        if noisy is None:
            continue

        if args.split == 'temporal':
            n      = len(noisy)
            split  = int(0.9 * n)
            all_train_noisy.append(noisy[:split])
            all_train_clean.append(clean[:split])
            all_val_noisy.append(noisy[split:])
            all_val_clean.append(clean[split:])
            print(f"    Done {time.time()-t0:.1f}s  (train: {split}, val: {n-split})\n")
        else:
            all_train_noisy.append(noisy)
            all_train_clean.append(clean)
            print(f"    Done {time.time()-t0:.1f}s  ({len(noisy)} windows)\n")

    if not all_train_noisy:
        print("ERROR: no data collected. Check paths.")
        sys.exit(1)

    rng = np.random.default_rng(42)

    if args.split == 'temporal':
        noisy_train = np.concatenate(all_train_noisy)
        clean_train = np.concatenate(all_train_clean)
        noisy_val   = np.concatenate(all_val_noisy)
        clean_val   = np.concatenate(all_val_clean)
        perm        = rng.permutation(len(noisy_train))
        noisy_train = noisy_train[perm]
        clean_train = clean_train[perm]
    else:
        all_noisy = np.concatenate(all_train_noisy)
        all_clean = np.concatenate(all_train_clean)
        perm      = rng.permutation(len(all_noisy))
        all_noisy = all_noisy[perm]
        all_clean = all_clean[perm]
        split     = int(0.9 * len(all_noisy))
        noisy_train, noisy_val = all_noisy[:split], all_noisy[split:]
        clean_train, clean_val = all_clean[:split], all_clean[split:]

    print(f"Total  : train={len(noisy_train):,}  val={len(noisy_val):,}  "
          f"shape={noisy_train.shape[1:]}")
    print(f"noisy  : mean={noisy_train.mean():.4f}  std={noisy_train.std():.4f}")
    print(f"clean  : mean={clean_train.mean():.4f}  std={clean_train.std():.4f}")

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        noisy_train=noisy_train, clean_train=clean_train,
        noisy_val=noisy_val,     clean_val=clean_val,
    )
    print(f"\nSaved -> {out_path}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--win_sec',  type=float, default=1.0)
    parser.add_argument('--hop_sec',  type=float, default=0.25)
    parser.add_argument('--out_dir',  default=DEFAULT_OUT)
    parser.add_argument('--out_name', default='train_data_alice.npz')
    parser.add_argument('--split',    default='temporal', choices=['temporal', 'random'])
    args = parser.parse_args()
    main(args)
