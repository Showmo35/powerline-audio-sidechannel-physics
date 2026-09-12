"""
prepare_data.py
---------------
Bandpass data preparation for UNet training with perceptual loss.
Produces paired noisy/clean mel spectrograms (no text labels needed).

Same pipeline as UNet/prepare_data_bandpass.py:
  - Bandpass 50-4000 Hz, downsample 200kHz -> 16kHz
  - Whisper-format log-mel spectrograms (80 bins)
  - 1.0s windows, 0.25s hop
  - Temporal split: 90/10 per chapter (no overlap)

Usage:
    python prepare_data.py
    python prepare_data.py --out_dir data --out_name train_data.npz
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
    'May29_Alice': os.path.join(SCRATCH, 'May29_Alice'),
}
MP3_DIR         = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ── Chapter timing ────────────────────────────────────────────────────────────
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
BP_LOW      = 50          # Bandpass low cutoff (Hz)
BP_HIGH     = 4000        # Bandpass high cutoff (Hz)

# ── Whisper STFT / mel parameters ─────────────────────────────────────────────
N_FFT   = 400
HOP     = 160
N_MELS  = 80

# ── helpers ───────────────────────────────────────────────────────────────────

def bandpass_filter(x, fs, lo, hi, order=5):
    """Bandpass filter using SOS for numerical stability."""
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


# ── Whisper-format mel spectrogram ───────────────────────────────────────────

_WHISPER_FB = None

def _get_whisper_filterbank():
    global _WHISPER_FB
    if _WHISPER_FB is None:
        _WHISPER_FB = whisper.audio.mel_filters(
            torch.device('cpu'), N_MELS).numpy()
    return _WHISPER_FB


def log_mel_spectrogram(x, sr=AUDIO_SR):
    """Whisper-format log-mel spectrogram. Returns (80, T) float32."""
    fb  = _get_whisper_filterbank()
    win = np.hanning(N_FFT)
    nf  = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * win for i in range(nf)])
    mag_sq = np.abs(np.fft.rfft(frames, axis=-1))**2
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


# ── main ─────────────────────────────────────────────────────────────────────

def process_file(pl_path, chapter_name, win_sec, hop_sec, verbose=True):
    """
    Bandpass the raw powerline signal at 50-4000 Hz, downsample to 16 kHz,
    align with MP3 ground truth, and compute paired mel spectrograms.

    Returns
    -------
    noisy_specs : np.ndarray  (N, 1, 80, T)
    clean_specs : np.ndarray  (N, 1, 80, T)
    """
    timing = CHAPTER_TIMING[chapter_name]
    offset_sec   = timing['start']
    mp3_duration = timing['mp3_duration']

    if verbose:
        print(f"  Loading {os.path.basename(pl_path)} ...")

    pl_raw = np.fromfile(pl_path, dtype=np.float32).astype(np.float64)

    start_samp = int(offset_sec * FS)
    chap_samps = int(mp3_duration * FS)
    pl_raw = pl_raw[start_samp:start_samp + chap_samps]

    if verbose:
        print(f"    Chapter  : {chapter_name} | offset {offset_sec}s | "
              f"duration {len(pl_raw)/FS:.1f}s")

    # Bandpass 50-4000 Hz (captures 60 Hz harmonic modulation by speech)
    if verbose:
        print(f"    Bandpass  : {BP_LOW}-{BP_HIGH} Hz")
    pl_bp = bandpass_filter(pl_raw, FS, BP_LOW, BP_HIGH)

    # Downsample to 16 kHz
    pl_audio = resample_to(pl_bp, FS, AUDIO_SR)

    # Load MP3 ground truth
    ch_num   = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f"Alice_In_Wonderland_ch_{ch_num}.mp3")
    if not os.path.exists(mp3_path):
        print(f"    WARNING: MP3 not found: {mp3_path}  -- skipping")
        return None, None

    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

    # Align lengths
    L = min(len(pl_audio), len(mp3_audio))
    pl_audio  = pl_audio[:L]
    mp3_audio = mp3_audio[:L]

    # RMS normalize
    pl_audio  = rms_norm(pl_audio)
    mp3_audio = rms_norm(mp3_audio)

    # Window and compute spectrograms
    win_samp = int(win_sec * AUDIO_SR)
    hop_samp = int(hop_sec * AUDIO_SR)
    starts   = range(0, L - win_samp, hop_samp)

    noisy_list, clean_list = [], []
    for s in starts:
        e = s + win_samp
        pl_lmel  = log_mel_spectrogram(pl_audio[s:e])
        mp3_lmel = log_mel_spectrogram(mp3_audio[s:e])
        noisy_list.append(pl_lmel[np.newaxis])
        clean_list.append(mp3_lmel[np.newaxis])

    if verbose:
        print(f"    Windows  : {len(noisy_list)}  "
              f"spec shape (1,{N_MELS},{noisy_list[0].shape[-1]})")

    return (np.stack(noisy_list).astype(np.float32),
            np.stack(clean_list).astype(np.float32))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nOutput directory : {args.out_dir}")
    print(f"Window : {args.win_sec}s  Hop : {args.hop_sec}s")
    print(f"Bandpass : {BP_LOW}-{BP_HIGH} Hz  (no IQ demod)")
    print(f"Split : temporal (90/10 per chapter)")
    print(f"Folders: {args.folders}\n")

    all_train_noisy, all_train_clean = [], []
    all_val_noisy,   all_val_clean   = [], []

    for folder_name in args.folders:
        if folder_name not in DATA_FOLDERS:
            print(f"WARNING: unknown folder '{folder_name}' -- skipping")
            continue
        data_dir = DATA_FOLDERS[folder_name]
        pl_files = sorted(glob.glob(os.path.join(data_dir, '*_real.bin')))
        print(f"Folder: {folder_name}  ({len(pl_files)} files)")

        for pl_path in pl_files:
            file_id      = os.path.basename(pl_path).replace('_real.bin', '')
            chapter_name = match_chapter_name(file_id)

            if chapter_name is None:
                print(f"  SKIP {file_id}: no chapter timing found")
                continue

            t0 = time.time()
            noisy, clean = process_file(
                pl_path, chapter_name, args.win_sec, args.hop_sec)

            if noisy is None:
                continue

            # Temporal split: first 90% train, last 10% val (no overlap)
            n_win = len(noisy)
            split_idx = int(0.9 * n_win)
            all_train_noisy.append(noisy[:split_idx])
            all_train_clean.append(clean[:split_idx])
            all_val_noisy.append(noisy[split_idx:])
            all_val_clean.append(clean[split_idx:])
            print(f"    Done in {time.time()-t0:.1f}s  "
                  f"(train: {split_idx}, val: {n_win - split_idx})")

        print()

    if not all_train_noisy:
        print("ERROR: no data collected. Check paths and folder names.")
        sys.exit(1)

    rng = np.random.default_rng(42)

    noisy_train = np.concatenate(all_train_noisy, axis=0)
    clean_train = np.concatenate(all_train_clean, axis=0)
    noisy_val   = np.concatenate(all_val_noisy,   axis=0)
    clean_val   = np.concatenate(all_val_clean,   axis=0)

    # Shuffle training data
    perm = rng.permutation(len(noisy_train))
    noisy_train = noisy_train[perm]
    clean_train = clean_train[perm]

    print(f"Total: train={len(noisy_train):,}  val={len(noisy_val):,}  "
          f"shape: {noisy_train.shape[1:]}")
    print(f"Split: temporal (first 90% of each chapter = train, last 10% = val)")

    print(f"\nData stats:")
    print(f"  noisy train: mean={noisy_train.mean():.4f}  std={noisy_train.std():.4f}  "
          f"min={noisy_train.min():.4f}  max={noisy_train.max():.4f}")
    print(f"  clean train: mean={clean_train.mean():.4f}  std={clean_train.std():.4f}  "
          f"min={clean_train.min():.4f}  max={clean_train.max():.4f}")

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        noisy_train=noisy_train, clean_train=clean_train,
        noisy_val=noisy_val,     clean_val=clean_val,
    )

    print(f"\nSaved -> {out_path}")
    print(f"  Train : {len(noisy_train):,} windows")
    print(f"  Val   : {len(noisy_val):,} windows")
    print(f"  Spec shape (1,M,T) : {noisy_train.shape[1:]}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare bandpass data for UNet + perceptual loss training")
    parser.add_argument('--folders', nargs='+', default=['May29_Alice'])
    parser.add_argument('--win_sec', type=float, default=1.0)
    parser.add_argument('--hop_sec', type=float, default=0.25)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--out_name', default='train_data.npz')
    args = parser.parse_args()
    main(args)
