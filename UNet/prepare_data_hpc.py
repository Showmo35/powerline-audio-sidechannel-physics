"""
prepare_data_hpc.py
-------------------
Runs on an HPC cluster. Reads all chapter *_real.bin (powerline) and *_img.bin (USB)
files, aligns them with the corresponding MP3 ground truth using known chapter
offsets, extracts overlapping windows, and computes log-mel spectrograms in
Whisper's exact format (16 kHz, 80 mel bins, Whisper filterbank + log scaling).

This allows UNet predictions to be fed directly into Whisper's encoder.

Usage (on HPC login/compute node):
    python prepare_data_hpc.py
    python prepare_data_hpc.py --folders May29_Alice July10_Podcasts
    python prepare_data_hpc.py --win_sec 1.0 --hop_sec 0.25 --out_dir /fs/scratch/<allocation>/train_data
"""

import os, sys, glob, argparse, time
import numpy as np
import torch
import whisper
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

# ── HPC paths ─────────────────────────────────────────────────────────────────
SCRATCH         = '<REPO_ROOT>'
DATA_FOLDERS    = {
    'May29_Alice'     : os.path.join(SCRATCH, 'May29_Alice'),
    'July10_Podcasts' : os.path.join(SCRATCH, 'July10_Podcasts'),
}
MP3_DIR         = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ── Chapter timing from alice_timing_analysis.txt ────────────────────────────
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
FS          = 200_000     # SDR capture sample rate (Hz)
AUDIO_SR    = 16_000      # Whisper's sample rate (Hz)
FC_CARRIER  = 20_433.35   # SMPS carrier frequency (Hz)
AUDIO_BW    = 8_000       # Audio bandwidth after demodulation (Hz)

# ── Whisper-compatible STFT / mel parameters ─────────────────────────────────
N_FFT   = 400
HOP     = 160
N_MELS  = 80

# ── helpers ───────────────────────────────────────────────────────────────────

def butter_lowpass(x, fs, cutoff, order=5):
    nyq = fs / 2.0
    sos = butter(order, cutoff/nyq, btype='low', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def iq_demodulate(sig, fs=FS, fc=FC_CARRIER, audio_bw=AUDIO_BW):
    """IQ (AM) demodulation with SOS filters for numerical stability."""
    t   = np.arange(len(sig), dtype=np.float64) / fs
    sig = sig.astype(np.float64)
    I   = butter_lowpass(sig * np.cos(2*np.pi*fc*t), fs, audio_bw)
    Q   = butter_lowpass(sig * np.sin(2*np.pi*fc*t), fs, audio_bw)
    env = np.abs(I + 1j*Q).astype(np.float32)
    sos_hp = butter(4, 30/(fs/2), btype='high', output='sos')
    return sosfiltfilt(sos_hp, env).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out//g, fs_in//g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


# ── Whisper-format mel spectrogram ───────────────────────────────────────────

_WHISPER_FB = None

def _get_whisper_filterbank():
    """Load Whisper's mel filterbank (80, 201) — cached."""
    global _WHISPER_FB
    if _WHISPER_FB is None:
        _WHISPER_FB = whisper.audio.mel_filters(
            torch.device('cpu'), N_MELS).numpy()   # (80, 201)
    return _WHISPER_FB


def log_mel_spectrogram(x, sr=AUDIO_SR):
    """
    Compute log-mel spectrogram in Whisper's exact format.

    Returns (N_MELS=80, T) float32 in Whisper's normalised log scale.
    """
    fb  = _get_whisper_filterbank()                             # (80, 201)
    win = np.hanning(N_FFT)
    nf  = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i*HOP:i*HOP+N_FFT] * win for i in range(nf)])
    mag_sq = np.abs(np.fft.rfft(frames, axis=-1))**2           # (T, 201)
    mel    = (fb @ mag_sq.T)                                    # (80, T)

    # Whisper's log normalisation
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)                          # (80, T)


def match_chapter_name(file_id):
    """Map file basename to a chapter_timing key."""
    fid = file_id.lower()
    for ch in CHAPTER_TIMING:
        num = ch.split('_')[1]
        if (f'chapter_{num}' in fid or f'chap_{num}' in fid
                or f'chap_{int(num)}' in fid or f'chap{int(num)}' in fid):
            return ch
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def process_file(pl_path, usb_path, chapter_name, win_sec, hop_sec, verbose=True):
    """
    Load one chapter pair, demodulate, align with MP3, window and compute
    Whisper-format log-mel spectrograms.

    Returns
    -------
    noisy_specs : np.ndarray  (N, 1, 80, T)  powerline demod
    clean_specs : np.ndarray  (N, 1, 80, T)  MP3 ground truth
    usb_specs   : np.ndarray  (N, 1, 80, T)  USB demod (soft reference)
    """
    timing = CHAPTER_TIMING[chapter_name]
    offset_sec   = timing['start']
    mp3_duration = timing['mp3_duration']

    if verbose:
        print(f"  Loading {os.path.basename(pl_path)} ...")

    pl_raw  = np.fromfile(pl_path,  dtype=np.float32)
    usb_raw = np.fromfile(usb_path, dtype=np.float32)

    start_samp  = int(offset_sec * FS)
    chap_samps  = int(mp3_duration * FS)
    pl_raw  = pl_raw [start_samp : start_samp + chap_samps]
    usb_raw = usb_raw[start_samp : start_samp + chap_samps]

    if verbose:
        print(f"    Chapter  : {chapter_name} | offset {offset_sec}s | "
              f"duration {len(pl_raw)/FS:.1f}s")

    pl_demod  = iq_demodulate(pl_raw)
    usb_demod = iq_demodulate(usb_raw)

    pl_audio  = resample_to(pl_demod,  FS, AUDIO_SR)
    usb_audio = resample_to(usb_demod, FS, AUDIO_SR)

    ch_num   = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f"Alice_In_Wonderland_ch_{ch_num}.mp3")
    if not os.path.exists(mp3_path):
        print(f"    WARNING: MP3 not found: {mp3_path}  -- skipping")
        return None, None, None

    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

    L = min(len(pl_audio), len(usb_audio), len(mp3_audio))
    pl_audio  = pl_audio[:L]
    usb_audio = usb_audio[:L]
    mp3_audio = mp3_audio[:L]

    pl_audio  = rms_norm(pl_audio)
    usb_audio = rms_norm(usb_audio)
    mp3_audio = rms_norm(mp3_audio)

    win_samp = int(win_sec * AUDIO_SR)
    hop_samp = int(hop_sec * AUDIO_SR)
    starts   = range(0, L - win_samp, hop_samp)

    noisy_list, clean_list, usb_list = [], [], []
    for s in starts:
        e = s + win_samp
        pl_lmel  = log_mel_spectrogram(pl_audio[s:e])
        usb_lmel = log_mel_spectrogram(usb_audio[s:e])
        mp3_lmel = log_mel_spectrogram(mp3_audio[s:e])
        noisy_list.append(pl_lmel[np.newaxis])
        clean_list.append(mp3_lmel[np.newaxis])
        usb_list.append(usb_lmel[np.newaxis])

    if verbose:
        print(f"    Windows  : {len(noisy_list)}  "
              f"spec shape (1,{N_MELS},{noisy_list[0].shape[-1]})")

    return (np.stack(noisy_list).astype(np.float32),
            np.stack(clean_list).astype(np.float32),
            np.stack(usb_list).astype(np.float32))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nOutput directory : {args.out_dir}")
    print(f"Window : {args.win_sec}s  Hop : {args.hop_sec}s")
    print(f"Folders: {args.folders}")
    print(f"Spectrogram format: Whisper-compatible "
          f"(SR={AUDIO_SR}, N_FFT={N_FFT}, HOP={HOP}, N_MELS={N_MELS})\n")

    all_train_noisy, all_train_clean, all_train_usb = [], [], []
    all_val_noisy,   all_val_clean,   all_val_usb   = [], [], []

    for folder_name in args.folders:
        if folder_name not in DATA_FOLDERS:
            print(f"WARNING: unknown folder '{folder_name}' -- skipping")
            continue
        data_dir = DATA_FOLDERS[folder_name]
        pl_files = sorted(glob.glob(os.path.join(data_dir, '*_real.bin')))
        print(f"Folder: {folder_name}  ({len(pl_files)} files)")

        for pl_path in pl_files:
            usb_path = pl_path.replace('_real.bin', '_img.bin')
            if not os.path.exists(usb_path):
                print(f"  SKIP {pl_path}: no matching _img.bin")
                continue

            file_id      = os.path.basename(pl_path).replace('_real.bin', '')
            chapter_name = match_chapter_name(file_id)

            if chapter_name is None:
                print(f"  SKIP {file_id}: no chapter timing found")
                continue

            t0 = time.time()
            noisy, clean, usb = process_file(
                pl_path, usb_path, chapter_name,
                args.win_sec, args.hop_sec)

            if noisy is None:
                continue

            if args.split == 'temporal':
                # Temporal split: first 90% of each chapter → train, last 10% → val
                n_win = len(noisy)
                split_idx = int(0.9 * n_win)
                all_train_noisy.append(noisy[:split_idx])
                all_train_clean.append(clean[:split_idx])
                all_train_usb.append(usb[:split_idx])
                all_val_noisy.append(noisy[split_idx:])
                all_val_clean.append(clean[split_idx:])
                all_val_usb.append(usb[split_idx:])
                print(f"    Done in {time.time()-t0:.1f}s  "
                      f"(train: {split_idx}, val: {n_win - split_idx})")
            else:
                # Random split: collect all windows, split later
                all_train_noisy.append(noisy)
                all_train_clean.append(clean)
                all_train_usb.append(usb)
                print(f"    Done in {time.time()-t0:.1f}s  ({len(noisy)} windows)")

        print()

    if not all_train_noisy:
        print("ERROR: no data collected. Check paths and folder names.")
        sys.exit(1)

    rng = np.random.default_rng(42)

    if args.split == 'temporal':
        # Concatenate per-chapter splits
        noisy_train = np.concatenate(all_train_noisy, axis=0)
        clean_train = np.concatenate(all_train_clean, axis=0)
        usb_train   = np.concatenate(all_train_usb,   axis=0)
        noisy_val   = np.concatenate(all_val_noisy,   axis=0)
        clean_val   = np.concatenate(all_val_clean,   axis=0)
        usb_val     = np.concatenate(all_val_usb,     axis=0)
        # Shuffle train set
        perm = rng.permutation(len(noisy_train))
        noisy_train = noisy_train[perm]
        clean_train = clean_train[perm]
        usb_train   = usb_train[perm]
        print(f"Total: train={len(noisy_train):,}  val={len(noisy_val):,}  "
              f"shape: {noisy_train.shape[1:]}")
        print(f"Split: temporal (first 90% of each chapter = train, last 10% = val)")
    else:
        # Pool all windows, then random 90:10 split
        all_noisy = np.concatenate(all_train_noisy, axis=0)
        all_clean = np.concatenate(all_train_clean, axis=0)
        all_usb   = np.concatenate(all_train_usb,   axis=0)
        perm = rng.permutation(len(all_noisy))
        all_noisy = all_noisy[perm]
        all_clean = all_clean[perm]
        all_usb   = all_usb[perm]
        split_idx   = int(0.9 * len(all_noisy))
        noisy_train = all_noisy[:split_idx]
        clean_train = all_clean[:split_idx]
        usb_train   = all_usb[:split_idx]
        noisy_val   = all_noisy[split_idx:]
        clean_val   = all_clean[split_idx:]
        usb_val     = all_usb[split_idx:]
        print(f"Total: train={len(noisy_train):,}  val={len(noisy_val):,}  "
              f"shape: {noisy_train.shape[1:]}")
        print(f"Split: random shuffle (90% train, 10% val)")

    # Diagnostics
    print(f"\nData stats (Whisper log-mel scale, no z-norm):")
    print(f"  noisy train: mean={noisy_train.mean():.4f}  std={noisy_train.std():.4f}  "
          f"min={noisy_train.min():.4f}  max={noisy_train.max():.4f}")
    print(f"  clean train: mean={clean_train.mean():.4f}  std={clean_train.std():.4f}  "
          f"min={clean_train.min():.4f}  max={clean_train.max():.4f}")

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        noisy_train=noisy_train, clean_train=clean_train, usb_train=usb_train,
        noisy_val=noisy_val,     clean_val=clean_val,     usb_val=usb_val,
    )

    print(f"\nSaved -> {out_path}")
    print(f"  Train : {len(noisy_train):,} windows")
    print(f"  Val   : {len(noisy_val):,} windows")
    print(f"  Spec shape (1,M,T) : {noisy_train.shape[1:]}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', nargs='+', default=['May29_Alice'],
                        help='Which dataset folders to use')
    parser.add_argument('--win_sec', type=float, default=1.0,
                        help='Spectrogram window length in seconds')
    parser.add_argument('--hop_sec', type=float, default=0.25,
                        help='Window hop in seconds')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help='Output directory for .npz files')
    parser.add_argument('--split', default='temporal', choices=['temporal', 'random'],
                        help='Split strategy: temporal (first 90%% train) or random shuffle')
    parser.add_argument('--out_name', default='train_data.npz',
                        help='Output filename within out_dir')
    args = parser.parse_args()
    main(args)
