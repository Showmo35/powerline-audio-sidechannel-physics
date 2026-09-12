"""
alignment_analysis.py
---------------------
Confirms cross-correlation alignment of .bin vs .wav pairs.
Produces spectrograms showing:
  - unaligned noisy (bin) vs clean (wav)
  - aligned noisy vs clean
  - the cross-correlation curve used to find the lag
"""

import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, resample_poly, correlate
from math import gcd

# ── constants (match prepare_data.py) ────────────────────────────────────────
BIN_DIR = "<REPO_ROOT>/Powerline_Data_Captures/Soundbar/LibriSpeech"
WAV_DIR = "<REPO_ROOT>/Powerline_Data_Captures/audio_chunks"
OUT_DIR = "<REPO_ROOT>/Module4_UNet_Soundbar_V2/alignment_analysis"

FS      = 200_000   # bin sample rate
SR      = 16_000    # target audio sample rate
BP_LOW  = 50
BP_HIGH = 4000

# chunks to analyse (pick a few good full-length ones + one failed one)
CHUNKS  = [1, 5, 10, 20, 34]   # 34 is the short/failed chunk

VIZ_SEC    = 5      # seconds to show in spectrogram panels
TAIL_SEC   = 60     # seconds of tail to load for cross-correlation (keep small for speed)
MAX_LAG_SEC = 10    # search range for lag


os.makedirs(OUT_DIR, exist_ok=True)


# ── signal processing helpers ─────────────────────────────────────────────────

def bandpass(x, fs, lo, hi, order=5):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def resample(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


def read_bin_tail(path, tail_sec):
    mm = np.memmap(path, dtype=np.float32, mode="r")
    n  = int(tail_sec * FS)
    return np.array(mm[max(0, len(mm) - n):], dtype=np.float64)


def read_wav_tail(path, tail_sec):
    info = sf.info(path)
    n    = int(tail_sec * info.samplerate)
    start = max(0, info.frames - n)
    audio, sr = sf.read(path, start=start, frames=n, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64), int(sr)


def process_bin(raw, fs_in=FS, fs_out=SR):
    bp = bandpass(raw, fs_in, BP_LOW, BP_HIGH)
    rs = resample(bp, fs_in, fs_out)
    return rms_norm(rs)


def cross_corr_lag(noisy, clean, sr=SR, max_lag_sec=MAX_LAG_SEC):
    """
    Returns lag in samples such that clean[lag:] aligns with noisy[0:].
    Positive lag  → clean leads noisy (trim start of clean).
    Negative lag  → noisy leads clean (trim start of noisy).
    """
    max_lag = int(max_lag_sec * sr)
    n = min(len(noisy), len(clean))
    corr = correlate(clean[:n], noisy[:n], mode="full")
    lags  = np.arange(-(n - 1), n)
    mask  = np.abs(lags) <= max_lag
    best  = int(lags[mask][np.argmax(corr[mask])])
    return best, corr, lags, mask


def apply_lag(noisy, clean, lag):
    """Trim signals so they are aligned."""
    if lag > 0:
        clean = clean[lag:]
    elif lag < 0:
        noisy = noisy[-lag:]
    n = min(len(noisy), len(clean))
    return noisy[:n], clean[:n]


# ── spectrogram helper ────────────────────────────────────────────────────────

def spectrogram(x, sr=SR, n_fft=512, hop=128):
    win = np.hanning(n_fft)
    nf  = (len(x) - n_fft) // hop + 1
    frames = np.stack([x[i*hop: i*hop+n_fft] * win for i in range(nf)])
    mag = np.abs(np.fft.rfft(frames, axis=-1))
    return np.log1p(mag).T          # (freq, time)


def trim_viz(x, sr, sec):
    return x[:int(sec * sr)]


# ── main analysis loop ────────────────────────────────────────────────────────

summary_rows = []

for chunk_id in CHUNKS:
    name     = f"chunk_{chunk_id:03d}"
    bin_path = os.path.join(BIN_DIR, f"{name}.bin")
    wav_path = os.path.join(WAV_DIR, f"{name}.wav")

    if not os.path.exists(bin_path) or not os.path.exists(wav_path):
        print(f"[skip] {name}: file missing")
        continue

    print(f"\n── {name} ──")

    # ── load tails ────────────────────────────────────────────────────────────
    raw_bin = read_bin_tail(bin_path, TAIL_SEC)
    raw_wav, wav_sr = read_wav_tail(wav_path, TAIL_SEC)
    if wav_sr != SR:
        raw_wav = resample(raw_wav, wav_sr, SR)

    bin_dur = len(raw_bin) / FS
    wav_dur = len(raw_wav) / SR
    print(f"  bin tail loaded : {bin_dur:.2f}s  |  wav tail: {wav_dur:.2f}s")

    # ── process bin → 16 kHz ──────────────────────────────────────────────────
    noisy = process_bin(raw_bin)
    clean = rms_norm(raw_wav.astype(np.float32))

    # ── cross-correlation ─────────────────────────────────────────────────────
    lag, corr, lags, mask = cross_corr_lag(noisy, clean)
    lag_sec = lag / SR
    print(f"  cross-corr lag  : {lag} samples  =  {lag_sec:.3f}s")

    # aligned signals
    noisy_al, clean_al = apply_lag(noisy, clean, lag)
    print(f"  aligned length  : {len(noisy_al)/SR:.2f}s")

    summary_rows.append(dict(chunk=name, bin_dur=bin_dur, wav_dur=wav_dur,
                             lag_samples=lag, lag_sec=lag_sec))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"{name}  |  bin tail={bin_dur:.1f}s  wav tail={wav_dur:.1f}s"
                 f"  |  lag={lag_sec:.3f}s", fontsize=13, fontweight="bold")

    hop_v = 128

    # ── row 0: BEFORE alignment ───────────────────────────────────────────────
    noisy_v = trim_viz(noisy, SR, VIZ_SEC)
    clean_v = trim_viz(clean, SR, VIZ_SEC)
    S_n_b = spectrogram(noisy_v, hop=hop_v)
    S_c_b = spectrogram(clean_v, hop=hop_v)
    vmin = min(S_n_b.min(), S_c_b.min())
    vmax = max(S_n_b.max(), S_c_b.max())

    axes[0, 0].imshow(S_n_b, aspect="auto", origin="lower",
                      vmin=vmin, vmax=vmax, cmap="magma")
    axes[0, 0].set_title("BEFORE — Noisy (bin, first 5s of tail)")
    axes[0, 0].set_ylabel("Freq bin")

    axes[0, 1].imshow(S_c_b, aspect="auto", origin="lower",
                      vmin=vmin, vmax=vmax, cmap="magma")
    axes[0, 1].set_title("BEFORE — Clean (wav, first 5s of tail)")

    # ── row 1: AFTER alignment ────────────────────────────────────────────────
    noisy_av = trim_viz(noisy_al, SR, VIZ_SEC)
    clean_av = trim_viz(clean_al, SR, VIZ_SEC)
    S_n_a = spectrogram(noisy_av, hop=hop_v)
    S_c_a = spectrogram(clean_av, hop=hop_v)
    vmin2 = min(S_n_a.min(), S_c_a.min())
    vmax2 = max(S_n_a.max(), S_c_a.max())

    axes[1, 0].imshow(S_n_a, aspect="auto", origin="lower",
                      vmin=vmin2, vmax=vmax2, cmap="magma")
    axes[1, 0].set_title("AFTER  — Noisy (bin, aligned 5s)")
    axes[1, 0].set_ylabel("Freq bin")

    axes[1, 1].imshow(S_c_a, aspect="auto", origin="lower",
                      vmin=vmin2, vmax=vmax2, cmap="magma")
    axes[1, 1].set_title("AFTER  — Clean (wav, aligned 5s)")

    # ── row 2: waveform overlay + cross-correlation ───────────────────────────
    t_n = np.linspace(0, VIZ_SEC, int(VIZ_SEC * SR))
    t_len = min(len(t_n), len(noisy_av), len(clean_av))
    axes[2, 0].plot(t_n[:t_len], noisy_av[:t_len], alpha=0.7, label="noisy", lw=0.6)
    axes[2, 0].plot(t_n[:t_len], clean_av[:t_len], alpha=0.7, label="clean", lw=0.6)
    axes[2, 0].set_title("Waveform overlay (aligned)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].legend(fontsize=8)

    # cross-correlation curve (zoom around peak)
    lags_m  = lags[mask]
    corr_m  = corr[mask]
    peak_i  = np.argmax(corr_m)
    zoom_w  = int(MAX_LAG_SEC * SR)
    axes[2, 1].plot(lags_m / SR, corr_m, lw=0.8)
    axes[2, 1].axvline(lags_m[peak_i] / SR, color="red", lw=1.5,
                       label=f"peak lag={lag_sec:.3f}s")
    axes[2, 1].set_title("Cross-correlation (noisy vs clean)")
    axes[2, 1].set_xlabel("Lag (s)")
    axes[2, 1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{name}_alignment.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── summary table ─────────────────────────────────────────────────────────────
print("\n\n══ SUMMARY ══")
print(f"{'chunk':<15} {'bin_tail':>10} {'wav_tail':>10} {'lag_s':>10} {'lag_samp':>10}")
print("-" * 60)
for r in summary_rows:
    print(f"{r['chunk']:<15} {r['bin_dur']:>10.2f} {r['wav_dur']:>10.2f} "
          f"{r['lag_sec']:>10.3f} {r['lag_samples']:>10}")

print(f"\nImages saved to: {OUT_DIR}")
