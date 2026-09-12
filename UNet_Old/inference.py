"""
inference.py
------------
Load a trained PowerlineUNet checkpoint and reconstruct audio
from a powerline IQ-demodulated signal.

Usage:
    python inference.py                           # uses default .mat file
    python inference.py --mat path/to/file.mat    # custom file
"""

import os, sys, argparse
import numpy as np
import torch
import scipy.io as sio
import soundfile as sf
from scipy.signal import butter, filtfilt, hilbert, resample_poly
from math import gcd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import PowerlineUNet

# ── constants ─────────────────────────────────────────────────────────────────
BASE        = r"C:\Users\showm\OneDrive\Desktop\Power_Test"
CKPT_PATH   = os.path.join(BASE, "checkpoints", "best_model.pt")
MAT_FILE    = os.path.join(BASE, "Chap_4_20s_segment.mat")
OUT_DIR     = os.path.join(BASE, "inference_output")
os.makedirs(OUT_DIR, exist_ok=True)

FS          = 200_000
MP3_SR      = 22_050
FC_CARRIER  = 20_433.35
OFFSET_SEC  = 1.95
N_FFT       = 512
HOP         = 128
N_MELS      = 64
F_MIN       = 80.0
F_MAX       = 8_000.0
WIN_SEC     = 1.0
HOP_SEC     = 0.125    # 50% denser hop for overlap-add reconstruction


# ── signal processing helpers ─────────────────────────────────────────────────

def butter_filter(x, fs, lo=None, hi=None, order=6):
    nyq = fs / 2
    if lo and hi:  b, a = butter(order, [lo/nyq, hi/nyq], 'band')
    elif lo:       b, a = butter(order, lo/nyq, 'high')
    else:          b, a = butter(order, hi/nyq, 'low')
    return filtfilt(b, a, x)


def iq_demodulate(sig, fs, fc, audio_bw=8000):
    t = np.arange(len(sig)) / fs
    I = butter_filter(sig * np.cos(2*np.pi*fc*t), fs, hi=audio_bw)
    Q = butter_filter(sig * np.sin(2*np.pi*fc*t), fs, hi=audio_bw)
    env = np.abs(I + 1j*Q)
    return butter_filter(env, fs, lo=30).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out//g, fs_in//g).astype(np.float32)


def hz_to_mel(hz):  return 2595 * np.log10(1 + hz / 700)
def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels, f_min, f_max):
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1), np.float32)
    for m in range(1, n_mels+1):
        for k in range(bins[m-1], bins[m]):
            fb[m-1,k] = (k - bins[m-1]) / max(1, bins[m] - bins[m-1])
        for k in range(bins[m], bins[m+1]):
            fb[m-1,k] = (bins[m+1] - k) / max(1, bins[m+1] - bins[m])
    return fb

_FB = None
def log_mel_spec(x, sr=MP3_SR):
    global _FB
    if _FB is None:
        _FB = mel_filterbank(sr, N_FFT, N_MELS, F_MIN, F_MAX)
    win    = np.hanning(N_FFT)
    nf     = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i*HOP:i*HOP+N_FFT] * win for i in range(nf)])
    mag    = np.abs(np.fft.rfft(frames, axis=-1))
    mel    = mag @ _FB.T
    return np.log1p(mel * 1e4).T.astype(np.float32)   # (M, T)


def spec_to_audio_griffin_lim(spec_db, sr=MP3_SR, n_iter=60):
    """
    Approximate inversion of log-mel spectrogram via Griffin-Lim on STFT.
    spec_db : (M, T) log-mel spectrogram
    """
    # Un-log mel
    mel_mag = (np.expm1(spec_db) / 1e4).clip(0)            # (M, T)

    # Pseudo-inverse of mel filterbank to get linear spectrogram estimate
    fb = mel_filterbank(sr, N_FFT, N_MELS, F_MIN, F_MAX)   # (M, F)
    # pinv approach
    fb_t = fb.T                                              # (F, M)
    lin_mag = mel_mag.T @ fb.T                               # (T, F) approximate
    lin_mag = lin_mag.clip(0)

    # Griffin-Lim
    n_frames, n_freqs = lin_mag.shape
    # Init random phase
    angles = np.exp(1j * 2 * np.pi * np.random.rand(*lin_mag.shape))
    for _ in range(n_iter):
        stft = lin_mag * angles
        # iSTFT
        sig  = _istft(stft)
        # re-STFT
        stft2 = _stft(sig, n_frames)
        angles = np.exp(1j * np.angle(stft2))

    return _istft(lin_mag * angles)


def _stft(x, n_frames):
    win    = np.hanning(N_FFT)
    result = np.zeros((n_frames, N_FFT//2+1), dtype=complex)
    for i in range(n_frames):
        frame = x[i*HOP:i*HOP+N_FFT]
        if len(frame) < N_FFT:
            frame = np.pad(frame, (0, N_FFT-len(frame)))
        result[i] = np.fft.rfft(frame * win)
    return result

def _istft(stft):
    n_frames = stft.shape[0]
    sig_len  = (n_frames - 1) * HOP + N_FFT
    sig      = np.zeros(sig_len)
    win      = np.hanning(N_FFT)
    win_sum  = np.zeros(sig_len)
    for i in range(n_frames):
        frame = np.fft.irfft(stft[i])[:N_FFT]
        sig[i*HOP:i*HOP+N_FFT]     += frame * win
        win_sum[i*HOP:i*HOP+N_FFT] += win**2
    win_sum = np.maximum(win_sum, 1e-8)
    return (sig / win_sum).astype(np.float32)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(mat_path, ckpt_path, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    stats = ckpt['data_stats']
    noisy_mean = stats['noisy_mean']
    noisy_std  = stats['noisy_std']
    clean_mean = stats['clean_mean']
    clean_std  = stats['clean_std']
    print(f"  Best val loss : {ckpt['val_loss']:.5f}  (epoch {ckpt['epoch']})")

    model = PowerlineUNet(base_ch=args.get('base_ch', 32)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  ({model.count_params():,} params)")

    # Load and preprocess signal
    print(f"\nLoading: {mat_path}")
    mat = sio.loadmat(mat_path)
    pl  = mat['powerline'].flatten().astype(np.float64)
    mp3 = mat['mp3'].flatten().astype(np.float32)

    print("IQ demodulating powerline...")
    pl_demod = iq_demodulate(pl, FS, FC_CARRIER)
    pl_22k   = resample_to(pl_demod, FS, MP3_SR)
    off      = int(OFFSET_SEC * MP3_SR)
    mp3_al   = mp3[off:off+len(pl_22k)]
    L        = min(len(pl_22k), len(mp3_al))
    pl_22k   = pl_22k[:L];  mp3_al = mp3_al[:L]

    # RMS normalize
    def rms_norm(x, r=0.05): return x * r / (np.sqrt(np.mean(x**2)) + 1e-9)
    pl_22k = rms_norm(pl_22k); mp3_al = rms_norm(mp3_al)

    # Slide windows and run model
    win_s = int(WIN_SEC * MP3_SR)
    hop_s = int(HOP_SEC * MP3_SR)
    starts = list(range(0, L - win_s, hop_s))

    pred_specs = []
    print(f"Running model on {len(starts)} windows...")
    with torch.no_grad():
        for s in starts:
            seg  = pl_22k[s:s+win_s]
            spec = log_mel_spec(seg)                          # (M, T)
            spec_norm = (spec - noisy_mean) / noisy_std
            inp  = torch.from_numpy(spec_norm[np.newaxis, np.newaxis]).to(device)
            out  = model(inp).squeeze().cpu().numpy()         # (M, T)
            # De-normalize
            out  = out * clean_std + clean_mean
            pred_specs.append(out)

    print("Inverting spectrograms to audio (Griffin-Lim)...")
    reconstructed_segs = []
    for spec in pred_specs:
        audio = spec_to_audio_griffin_lim(spec, sr=MP3_SR, n_iter=50)
        reconstructed_segs.append(audio[:win_s])

    # Overlap-add
    out_sig = np.zeros(L)
    win_count = np.zeros(L)
    for i, (s, seg) in enumerate(zip(starts, reconstructed_segs)):
        e = min(s + len(seg), L)
        out_sig[s:e]   += seg[:e-s]
        win_count[s:e] += 1
    win_count = np.maximum(win_count, 1)
    out_sig /= win_count

    # Normalize output
    out_sig = rms_norm(out_sig.astype(np.float32))

    # Save
    recon_path = os.path.join(out_dir, "reconstructed_audio.wav")
    mp3_path   = os.path.join(out_dir, "mp3_ground_truth.wav")
    noisy_path = os.path.join(out_dir, "powerline_demodulated.wav")

    sf.write(recon_path, out_sig.astype(np.float32), MP3_SR)
    sf.write(mp3_path,   mp3_al.astype(np.float32), MP3_SR)
    sf.write(noisy_path, pl_22k.astype(np.float32), MP3_SR)
    print(f"\nSaved: {recon_path}")
    print(f"Saved: {mp3_path}")
    print(f"Saved: {noisy_path}")

    # Quality metrics
    from scipy.signal import correlate
    def snr_db(clean, noisy):
        s_pwr = np.mean(clean**2)
        n_pwr = np.mean((clean - noisy)**2) + 1e-10
        return 10 * np.log10(s_pwr / n_pwr + 1e-10)

    n = min(len(out_sig), len(mp3_al))
    snr_recon  = snr_db(mp3_al[:n], out_sig[:n])
    snr_noisy  = snr_db(mp3_al[:n], pl_22k[:n])
    r_recon    = np.corrcoef(mp3_al[:n], out_sig[:n])[0,1]
    r_noisy    = np.corrcoef(mp3_al[:n], pl_22k[:n])[0,1]

    print(f"\n{'Metric':<28} {'Noisy PL':>12}  {'Reconstructed':>15}")
    print("─" * 58)
    print(f"{'SNR vs MP3 (dB)':<28} {snr_noisy:>12.2f}  {snr_recon:>15.2f}")
    print(f"{'Pearson r vs MP3':<28} {r_noisy:>12.4f}  {r_recon:>15.4f}")

    # Comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    t = np.arange(n) / MP3_SR
    axes[0].plot(t, mp3_al[:n], color='black', lw=0.5)
    axes[0].set_title('MP3 Ground Truth')
    axes[1].plot(t, out_sig[:n], color='blue',  lw=0.5)
    axes[1].set_title(f'Reconstructed  (r={r_recon:.3f}, SNR={snr_recon:.1f}dB)')
    axes[2].plot(t, pl_22k[:n],  color='red',   lw=0.5)
    axes[2].set_title(f'Noisy Powerline IQ  (r={r_noisy:.3f}, SNR={snr_noisy:.1f}dB)')
    for ax in axes:
        ax.set_ylabel('Amplitude'); ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inference_comparison.png"), dpi=130)
    plt.close()
    print(f"\nPlot saved -> {out_dir}/inference_comparison.png")
    print("Inference complete.")
    return out_sig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat',  default=MAT_FILE)
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument('--out',  default=OUT_DIR)
    args = parser.parse_args()
    run_inference(args.mat, args.ckpt, args.out)
