"""
prepare_data.py
---------------
Loads Chap_4_20s_segment.mat, IQ-demodulates the powerline and USB signals,
aligns them with the MP3 ground truth, and saves windowed spectrogram pairs
as a .npz file for training.

Output: data/train_data.npz
  - noisy_specs   : (N, 1, F, T)  - powerline demodulated mel-spectrograms
  - clean_specs   : (N, 1, F, T)  - MP3 ground truth mel-spectrograms
  - usb_specs     : (N, 1, F, T)  - USB demodulated (soft GT, for reference)
"""

import os, numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, resample_poly
from math import gcd
import warnings; warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
BASE   = r"C:\Users\showm\OneDrive\Desktop\Power_Test"
MAT    = os.path.join(BASE, "Chap_4_20s_segment.mat")
OUT    = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)

# ── signal params ─────────────────────────────────────────────────────────────
FS          = 200_000        # capture sample rate
MP3_SR      = 22_050         # target audio sample rate
FC_CARRIER  = 20_433.35      # confirmed SMPS carrier (Hz)
OFFSET_SEC  = 1.95           # USB capture start offset into MP3

# ── STFT / mel params ─────────────────────────────────────────────────────────
N_FFT       = 512
HOP         = 128
N_MELS      = 64
F_MIN       = 80.0
F_MAX       = 8_000.0
WIN_SEC     = 1.0            # spectrogram window length (seconds)
HOP_SEC     = 0.25           # spectrogram window hop (seconds)

# ── helpers ───────────────────────────────────────────────────────────────────

def butter_filter(x, fs, lo=None, hi=None, order=6):
    nyq = fs / 2
    if lo and hi:
        b, a = butter(order, [lo/nyq, hi/nyq], 'band')
    elif lo:
        b, a = butter(order, lo/nyq, 'high')
    elif hi:
        b, a = butter(order, hi/nyq, 'low')
    return filtfilt(b, a, x)


def iq_demodulate(sig, fs, fc, audio_bw=8000):
    """AM demodulation via IQ (complex envelope)."""
    t   = np.arange(len(sig)) / fs
    I   = butter_filter(sig * np.cos(2*np.pi*fc*t), fs, hi=audio_bw)
    Q   = butter_filter(sig * np.sin(2*np.pi*fc*t), fs, hi=audio_bw)
    env = np.abs(I + 1j*Q)
    env = butter_filter(env, fs, lo=30)        # remove DC
    return env.astype(np.float32)


def resample_to_mp3sr(x, fs_in, fs_out):
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out//g, fs_in//g).astype(np.float32)


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels, f_min, f_max):
    """Returns (n_mels, n_fft//2+1) mel filterbank matrix."""
    f_min_mel = hz_to_mel(f_min)
    f_max_mel = hz_to_mel(f_max)
    mel_points = np.linspace(f_min_mel, f_max_mel, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m       = bin_points[m]
        f_m_plus  = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / max(1, f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / max(1, f_m_plus - f_m)
    return fbank


def stft_magnitude(x, n_fft=N_FFT, hop=HOP):
    """Compute STFT magnitude (frames, freq_bins)."""
    window  = np.hanning(n_fft)
    n_frames = (len(x) - n_fft) // hop + 1
    frames  = np.stack([x[i*hop:i*hop+n_fft] * window
                        for i in range(n_frames)])                # (T, n_fft)
    spec    = np.abs(np.fft.rfft(frames, axis=-1))                # (T, F)
    return spec.astype(np.float32)                                 # (T, F)


def log_mel_spectrogram(x, sr=MP3_SR, n_fft=N_FFT, hop=HOP,
                        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
    """Returns (n_mels, T) log mel-spectrogram."""
    mag  = stft_magnitude(x, n_fft, hop)                          # (T, F)
    fb   = mel_filterbank(sr, n_fft, n_mels, f_min, f_max)        # (M, F)
    mel  = (mag @ fb.T)                                            # (T, M)
    lmel = np.log1p(mel * 1e4).T                                   # (M, T)
    return lmel.astype(np.float32)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading .mat file...")
    mat = sio.loadmat(MAT)
    pl  = mat['powerline'].flatten().astype(np.float64)
    usb = mat['usb'].flatten().astype(np.float64)
    mp3 = mat['mp3'].flatten().astype(np.float32)

    print("IQ-demodulating signals...")
    pl_demod  = iq_demodulate(pl,  FS, FC_CARRIER)
    usb_demod = iq_demodulate(usb, FS, FC_CARRIER)

    print("Resampling to 22050 Hz...")
    pl_22k  = resample_to_mp3sr(pl_demod,  FS, MP3_SR)
    usb_22k = resample_to_mp3sr(usb_demod, FS, MP3_SR)

    # Align to MP3
    off     = int(OFFSET_SEC * MP3_SR)
    mp3_al  = mp3[off : off + len(pl_22k)]
    L       = min(len(pl_22k), len(usb_22k), len(mp3_al))
    pl_22k  = pl_22k[:L];  usb_22k = usb_22k[:L];  mp3_al = mp3_al[:L]

    # ── Normalize each signal independently ──────────────────────────────────
    def rms_norm(x, target_rms=0.05):
        r = np.sqrt(np.mean(x**2)) + 1e-9
        return x * (target_rms / r)

    pl_22k  = rms_norm(pl_22k)
    usb_22k = rms_norm(usb_22k)
    mp3_al  = rms_norm(mp3_al)

    # ── Window into overlapping segments ──────────────────────────────────────
    win_samples = int(WIN_SEC * MP3_SR)
    hop_samples = int(HOP_SEC * MP3_SR)
    starts      = range(0, L - win_samples, hop_samples)
    print(f"Creating {len(list(starts))} windows  "
          f"({WIN_SEC}s, hop={HOP_SEC}s)...")

    noisy_list, clean_list, usb_list = [], [], []

    for s in starts:
        e = s + win_samples
        pl_seg  = pl_22k[s:e]
        usb_seg = usb_22k[s:e]
        mp3_seg = mp3_al[s:e]

        # Log mel spectrograms
        pl_lmel  = log_mel_spectrogram(pl_seg)
        usb_lmel = log_mel_spectrogram(usb_seg)
        mp3_lmel = log_mel_spectrogram(mp3_seg)

        noisy_list.append(pl_lmel[np.newaxis])   # (1, M, T)
        clean_list.append(mp3_lmel[np.newaxis])
        usb_list.append(usb_lmel[np.newaxis])

    noisy_specs = np.stack(noisy_list)   # (N, 1, M, T)
    clean_specs = np.stack(clean_list)
    usb_specs   = np.stack(usb_list)

    print(f"noisy_specs shape : {noisy_specs.shape}")
    print(f"clean_specs shape : {clean_specs.shape}")

    # Global normalisation  (zero-mean / unit-variance per channel across dataset)
    noisy_mean = noisy_specs.mean(); noisy_std = noisy_specs.std() + 1e-6
    clean_mean = clean_specs.mean(); clean_std = clean_specs.std() + 1e-6

    noisy_norm = (noisy_specs - noisy_mean) / noisy_std
    clean_norm = (clean_specs - clean_mean) / clean_std
    usb_norm   = (usb_specs   - noisy_mean) / noisy_std   # same scale as noisy

    # Data augmentation: gain jitter + time-shift copies
    aug_noisy, aug_clean, aug_usb = [noisy_norm], [clean_norm], [usb_norm]
    rng = np.random.default_rng(42)
    for gain in [0.8, 1.2]:
        aug_noisy.append(noisy_norm * gain)
        aug_clean.append(clean_norm * gain)
        aug_usb.append(usb_norm * gain)

    noisy_all = np.concatenate(aug_noisy, axis=0)
    clean_all = np.concatenate(aug_clean, axis=0)
    usb_all   = np.concatenate(aug_usb,   axis=0)

    # Shuffle
    idx = rng.permutation(len(noisy_all))
    noisy_all = noisy_all[idx]; clean_all = clean_all[idx]; usb_all = usb_all[idx]

    # Train / val split 80/20
    split = int(0.8 * len(noisy_all))
    out_path = os.path.join(OUT, "train_data.npz")
    np.savez_compressed(
        out_path,
        noisy_train   = noisy_all[:split].astype(np.float32),
        clean_train   = clean_all[:split].astype(np.float32),
        usb_train     = usb_all[:split].astype(np.float32),
        noisy_val     = noisy_all[split:].astype(np.float32),
        clean_val     = clean_all[split:].astype(np.float32),
        usb_val       = usb_all[split:].astype(np.float32),
        noisy_mean    = np.array([noisy_mean]),
        noisy_std     = np.array([noisy_std]),
        clean_mean    = np.array([clean_mean]),
        clean_std     = np.array([clean_std]),
    )
    print(f"\nSaved -> {out_path}")
    print(f"  Train: {noisy_all[:split].shape[0]} windows")
    print(f"  Val  : {noisy_all[split:].shape[0]} windows")
    print(f"  Spec shape (1, M, T): {noisy_all.shape[1:]}")
    print("Done.")


if __name__ == "__main__":
    main()
