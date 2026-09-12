"""
prepare_data_hpc.py
-------------------
Runs on an HPC cluster. Reads all chapter *_real.bin (powerline) and *_img.bin (USB)
files, aligns them with the corresponding MP3 ground truth using known chapter
offsets, extracts overlapping windows, computes log-mel spectrograms, and saves
sharded .npz files ready for training.

Usage (on HPC login/compute node):
    python prepare_data_hpc.py
    python prepare_data_hpc.py --folders May29_Alice July10_Podcasts
    python prepare_data_hpc.py --win_sec 1.0 --hop_sec 0.25 --out_dir /fs/scratch/<allocation>/train_data
"""

import os, sys, glob, argparse, time
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

# ── HPC paths ─────────────────────────────────────────────────────────────────
SCRATCH         = '/fs/scratch/<allocation>'
DATA_FOLDERS    = {
    'May29_Alice'     : os.path.join(SCRATCH, 'May29_Alice'),
    'July10_Podcasts' : os.path.join(SCRATCH, 'July10_Podcasts'),
}
MP3_DIR         = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
DEFAULT_OUT_DIR = os.path.join(SCRATCH, 'train_data')

# ── Chapter timing from alice_timing_analysis.txt ────────────────────────────
# start = seconds into capture where the chapter audio begins
# mp3_duration = length of the MP3 file in seconds
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
AUDIO_SR    = 22_050      # Target audio sample rate (Hz)
FC_CARRIER  = 20_433.35   # SMPS carrier frequency (Hz)
AUDIO_BW    = 8_000       # Audio bandwidth after demodulation (Hz)

# ── STFT / mel parameters ─────────────────────────────────────────────────────
N_FFT   = 512
HOP     = 128
N_MELS  = 64
F_MIN   = 80.0
F_MAX   = 8_000.0

# ── helpers ───────────────────────────────────────────────────────────────────

def butter_bandpass(x, fs, lo, hi, order=5):
    nyq = fs / 2.0
    sos = butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def butter_lowpass(x, fs, cutoff, order=5):
    nyq = fs / 2.0
    sos = butter(order, cutoff/nyq, btype='low', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def iq_demodulate(sig, fs=FS, fc=FC_CARRIER, audio_bw=AUDIO_BW):
    """
    IQ (AM) demodulation:
      1. Mix with carrier -> baseband I/Q
      2. Low-pass filter to audio bandwidth
      3. Compute complex envelope magnitude
      4. High-pass at 30 Hz to remove DC
    """
    t   = np.arange(len(sig), dtype=np.float64) / fs
    sig = sig.astype(np.float64)
    I   = butter_lowpass(sig * np.cos(2*np.pi*fc*t), fs, audio_bw)
    Q   = butter_lowpass(sig * np.sin(2*np.pi*fc*t), fs, audio_bw)
    env = np.abs(I + 1j*Q).astype(np.float32)
    # Remove DC drift
    nyq = fs / 2.0
    sos_hp = butter(4, 30/nyq, btype='high', output='sos')
    return sosfiltfilt(sos_hp, env).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out//g, fs_in//g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


# ── mel spectrogram (no librosa dependency for speed) ─────────────────────────

def _mel_filterbank(sr, n_fft, n_mels, f_min, f_max):
    def hz2mel(h): return 2595 * np.log10(1 + h/700)
    def mel2hz(m): return 700 * (10**(m/2595) - 1)
    mel_pts  = np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels+2)
    hz_pts   = mel2hz(mel_pts)
    bins     = np.floor((n_fft+1)*hz_pts/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1), np.float32)
    for m in range(1, n_mels+1):
        for k in range(bins[m-1], bins[m]):
            fb[m-1,k] = (k-bins[m-1]) / max(1, bins[m]-bins[m-1])
        for k in range(bins[m], bins[m+1]):
            fb[m-1,k] = (bins[m+1]-k) / max(1, bins[m+1]-bins[m])
    return fb

_FB_CACHE = {}

def log_mel_spectrogram(x, sr=AUDIO_SR):
    global _FB_CACHE
    key = (sr, N_FFT, N_MELS)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(sr, N_FFT, N_MELS, F_MIN, F_MAX)
    fb  = _FB_CACHE[key]
    win = np.hanning(N_FFT)
    nf  = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i*HOP:i*HOP+N_FFT]*win for i in range(nf)])
    mag    = np.abs(np.fft.rfft(frames, axis=-1)).astype(np.float32)
    mel    = mag @ fb.T                                   # (T, M)
    return np.log1p(mel * 1e4).T.astype(np.float32)      # (M, T)


def match_chapter_name(file_id):
    """Map file basename to a chapter_timing key."""
    fid = file_id.lower()
    for ch in CHAPTER_TIMING:
        num = ch.split('_')[1]                            # '01', '04', etc.
        if (f'chapter_{num}' in fid or f'chap_{num}' in fid
                or f'chap_{int(num)}' in fid or f'chap{int(num)}' in fid):
            return ch
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def process_file(pl_path, usb_path, chapter_name, win_sec, hop_sec, verbose=True):
    """
    Load one chapter pair, demodulate, align with MP3, window and compute
    log-mel spectrograms.

    Returns
    -------
    noisy_specs : np.ndarray  (N, 1, N_MELS, T)  powerline demod
    clean_specs : np.ndarray  (N, 1, N_MELS, T)  MP3 ground truth
    usb_specs   : np.ndarray  (N, 1, N_MELS, T)  USB demod (soft reference)
    """
    timing = CHAPTER_TIMING[chapter_name]
    offset_sec   = timing['start']
    mp3_duration = timing['mp3_duration']

    if verbose:
        print(f"  Loading {os.path.basename(pl_path)} ...")

    # Load raw .bin files (real float32 at 200 kHz)
    pl_raw  = np.fromfile(pl_path,  dtype=np.float32)
    usb_raw = np.fromfile(usb_path, dtype=np.float32)

    # Trim to aligned chapter content
    start_samp  = int(offset_sec * FS)
    chap_samps  = int(mp3_duration * FS)
    pl_raw  = pl_raw [start_samp : start_samp + chap_samps]
    usb_raw = usb_raw[start_samp : start_samp + chap_samps]

    if verbose:
        print(f"    Chapter  : {chapter_name} | offset {offset_sec}s | "
              f"duration {len(pl_raw)/FS:.1f}s")

    # IQ demodulate
    pl_demod  = iq_demodulate(pl_raw)
    usb_demod = iq_demodulate(usb_raw)

    # Resample to AUDIO_SR
    pl_audio  = resample_to(pl_demod,  FS, AUDIO_SR)
    usb_audio = resample_to(usb_demod, FS, AUDIO_SR)

    # Load matching MP3
    ch_num   = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f"Alice_In_Wonderland_ch_{ch_num}.mp3")
    if not os.path.exists(mp3_path):
        print(f"    WARNING: MP3 not found: {mp3_path}  -- skipping")
        return None, None, None

    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

    # Align lengths
    L = min(len(pl_audio), len(usb_audio), len(mp3_audio))
    pl_audio  = pl_audio[:L]
    usb_audio = usb_audio[:L]
    mp3_audio = mp3_audio[:L]

    # RMS normalise
    pl_audio  = rms_norm(pl_audio)
    usb_audio = rms_norm(usb_audio)
    mp3_audio = rms_norm(mp3_audio)

    # Sliding window -> log-mel spectrograms
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
    print(f"Folders: {args.folders}\n")

    all_noisy, all_clean, all_usb = [], [], []

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

            all_noisy.append(noisy)
            all_clean.append(clean)
            all_usb.append(usb)
            print(f"    Done in {time.time()-t0:.1f}s")

        print()

    if not all_noisy:
        print("ERROR: no data collected. Check paths and folder names.")
        sys.exit(1)

    # Concatenate all chapters
    noisy_all = np.concatenate(all_noisy, axis=0)
    clean_all = np.concatenate(all_clean, axis=0)
    usb_all   = np.concatenate(all_usb,   axis=0)
    print(f"Total windows : {len(noisy_all):,}  shape: {noisy_all.shape[1:]}")

    # Global statistics for normalisation (computed on training split only)
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(noisy_all))
    split = int(0.9 * len(noisy_all))

    train_idx = idx[:split]
    val_idx   = idx[split:]

    noisy_train = noisy_all[train_idx]
    clean_train = clean_all[train_idx]
    usb_train   = usb_all[train_idx]
    noisy_val   = noisy_all[val_idx]
    clean_val   = clean_all[val_idx]
    usb_val     = usb_all[val_idx]

    # Stats from training set
    noisy_mean = float(noisy_train.mean())
    noisy_std  = float(noisy_train.std()) + 1e-6
    clean_mean = float(clean_train.mean())
    clean_std  = float(clean_train.std()) + 1e-6

    print(f"\nNormalisation stats (from train split):")
    print(f"  noisy mean={noisy_mean:.4f}  std={noisy_std:.4f}")
    print(f"  clean mean={clean_mean:.4f}  std={clean_std:.4f}")

    # Normalise
    noisy_train = ((noisy_train - noisy_mean) / noisy_std).astype(np.float32)
    clean_train = ((clean_train - clean_mean) / clean_std).astype(np.float32)
    usb_train   = ((usb_train   - noisy_mean) / noisy_std).astype(np.float32)
    noisy_val   = ((noisy_val   - noisy_mean) / noisy_std).astype(np.float32)
    clean_val   = ((clean_val   - clean_mean) / clean_std).astype(np.float32)
    usb_val     = ((usb_val     - noisy_mean) / noisy_std).astype(np.float32)

    # Save
    out_path = os.path.join(args.out_dir, 'train_data.npz')
    np.savez_compressed(
        out_path,
        noisy_train=noisy_train, clean_train=clean_train, usb_train=usb_train,
        noisy_val=noisy_val,     clean_val=clean_val,     usb_val=usb_val,
        noisy_mean=np.array([noisy_mean]), noisy_std=np.array([noisy_std]),
        clean_mean=np.array([clean_mean]), clean_std=np.array([clean_std]),
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
    args = parser.parse_args()
    main(args)
