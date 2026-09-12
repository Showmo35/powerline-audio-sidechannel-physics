#!/usr/bin/env python3
"""
check_monitor_align.py — confirm powerline<->audio alignment for a new-device
capture (monitor, TV, ...) before any M22-style run.

New-device bins have NO .lag sidecars and their length offset vs the reference
wav is unknown a priori (soundbar was only 0.8 s longer, lag<=700 ms; monitor
~3.9 s longer). So we (1) verify the device leaks the loudness envelope at
all, and (2) find the true offset with a WIDE two-sided search.

For a sample of chunks: RMS-envelope of the device's 200 kHz float32 bin vs the
16 kHz reference wav, z-normalized cross-correlation over +-SEARCH_S, report
signed lag + peak. High peak (>~0.5) across chunks with a consistent lag =>
envelope leaks, alignment OK. Low/scattered peak => device does not carry the
envelope, an attack pipeline on it is pointless.

Usage:  python3 check_monitor_align.py [--device monitor|TV] [--n 12] [--dur 120] [--viz out.png] [--apply]
"""
import argparse, os, wave
import numpy as np
from scipy.signal import correlate as _xcorr

ROOT = "<REPO_ROOT>/Powerline_Data_Captures"
DEVICE_DIRS = {
    "monitor": "monitor_bin_captures",
    "soundbar": "soundbar_bin_captures",
    "TV": "TV",
}
AUD_DIR = os.path.join(ROOT, "audio_chunks")
CAP_SR = 200_000
FRAME_S = 0.05
SEARCH_S = 5.0          # wide two-sided search (device offset unknown, maybe > 700 ms)


def cap_env(bin_path, crop_s, frame_s=FRAME_S):
    frame = int(round(frame_s * CAP_SR))
    n_read = int(round(crop_s * CAP_SR))
    mm = np.memmap(bin_path, dtype=np.float32, mode="r")
    seg = mm[:n_read]
    nf = len(seg) // frame
    x = np.asarray(seg[:nf * frame], dtype=np.float64).reshape(nf, frame)
    e = np.sqrt(np.mean(x * x, axis=1)); del mm
    return e


def aud_env(wav_path, crop_s, frame_s=FRAME_S):
    w = wave.open(wav_path, "rb"); sr = w.getframerate()
    n = min(int(round(crop_s * sr)), w.getnframes())
    x = np.frombuffer(w.readframes(n), np.int16).astype(np.float64) / 32768.0; w.close()
    frame = int(round(frame_s * sr)); nf = len(x) // frame
    x = x[:nf * frame].reshape(nf, frame)
    return np.sqrt(np.mean(x * x, axis=1))


def xcorr_lag(ae, ce, frame_s=FRAME_S, search_s=SEARCH_S):
    n = min(len(ae), len(ce)); a = ae[:n]; c = ce[:n]
    a = (a - a.mean()) / (a.std() + 1e-12)
    c = (c - c.mean()) / (c.std() + 1e-12)
    xc = _xcorr(a, c, mode="full", method="fft") / n
    lags = np.arange(-(n - 1), n)
    mx = int(round(search_s / frame_s))
    valid = (np.abs(lags) <= mx)
    idx = np.flatnonzero(valid); k = int(idx[np.argmax(xc[idx])])
    # parabolic sub-frame refine
    if 0 < k < len(xc) - 1:
        y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]; den = y0 - 2 * y1 + y2
        delta = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den else 0.0
    else:
        delta = 0.0
    numpy_lag = lags[k] + delta
    phys_lag_ms = -numpy_lag * frame_s * 1000.0   # capture delayed behind audio -> positive
    return phys_lag_ms, float(xc[k]), (a, c, lags, xc, k)


def main():
    global SEARCH_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=sorted(DEVICE_DIRS), default="monitor",
                    help="which device's bin_captures dir to check (default monitor)")
    ap.add_argument("--n", type=int, default=12, help="# chunks to sample across the range")
    ap.add_argument("--dur", type=float, default=120.0, help="seconds read per chunk")
    ap.add_argument("--search", type=float, default=SEARCH_S, help="two-sided lag search window (s)")
    ap.add_argument("--viz", type=str, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="ALSO measure every chunk (dur-crop) and WRITE .lag sidecars")
    args = ap.parse_args()

    SEARCH_S = args.search
    bin_dir = os.path.join(ROOT, DEVICE_DIRS[args.device])

    chunks = sorted(f[:-4] for f in os.listdir(bin_dir)
                     if f.startswith("chunk_") and f.endswith(".bin"))
    if not chunks:
        raise SystemExit(f"No complete chunk_*.bin found in {bin_dir}")
    n_sample = min(args.n, len(chunks))
    idx = sorted(set(np.linspace(0, len(chunks) - 1, n_sample).round().astype(int)))
    sample = [chunks[i] for i in idx]
    print(f"[{args.device}] {len(chunks)} bins; sampling {len(sample)} for alignment check "
          f"(dur={args.dur:.0f}s, search +-{args.search:.0f}s)\n")
    print(f"{'chunk':<12}{'lag_ms':>9}{'peak':>7}")
    recs = []
    for ch in sample:
        wp = os.path.join(AUD_DIR, f"{ch}.wav"); bp = os.path.join(bin_dir, f"{ch}.bin")
        if not os.path.exists(wp):
            print(f"{ch:<12}  no wav"); continue
        lag, peak, diag = xcorr_lag(aud_env(wp, args.dur), cap_env(bp, args.dur), search_s=args.search)
        recs.append((ch, lag, peak, diag))
        print(f"{ch:<12}{lag:>9.0f}{peak:>7.2f}")
    lags = np.array([r[1] for r in recs]); peaks = np.array([r[2] for r in recs])
    print(f"\nlag  : min {lags.min():.0f}  median {np.median(lags):.0f}  max {lags.max():.0f} ms")
    print(f"peak : min {peaks.min():.2f}  median {np.median(peaks):.2f}  max {peaks.max():.2f}")
    good = np.median(peaks)
    print("\nVERDICT:", "ENVELOPE LEAKS — alignment feasible" if good > 0.4 else
          "WEAK/NO envelope coupling — alignment unreliable, leakage doubtful",
          f"(median peak {good:.2f})")

    if args.viz:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m = min(6, len(recs))
        fig, axes = plt.subplots(m, 1, figsize=(13, 2.2 * m))
        if m == 1:
            axes = [axes]
        for ax, (ch, lag, peak, (a, c, lags_, xc, k)) in zip(axes, recs[:m]):
            fs = FRAME_S; t = np.arange(len(a)) * fs
            shift = int(round(lag / (fs * 1000)))
            cs = np.full_like(c, np.nan)
            if shift > 0:
                cs[:len(cs) - shift] = c[shift:]
            elif shift < 0:
                cs[-shift:] = c[:len(cs) + shift]
            else:
                cs = c.copy()
            ax.plot(t, a, lw=.7, label="audio env")
            ax.plot(t, cs, lw=.7, label=f"{args.device} env +{lag:.0f}ms")
            ax.set_xlim(0, min(40, t[-1])); ax.set_ylim(-2, 5)
            ax.set_title(f"{ch}  lag={lag:.0f}ms  peak={peak:.2f}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{args.device} alignment check (median peak {good:.2f})")
        fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(args.viz, dpi=120)
        print(f"[viz] wrote {args.viz}")

    if args.apply:
        import csv
        print(f"\n[apply] measuring all {len(chunks)} {args.device} chunks…")
        meas = []   # (chunk, lag, peak)
        for i, ch in enumerate(chunks, 1):
            wp = os.path.join(AUD_DIR, f"{ch}.wav"); bp = os.path.join(bin_dir, f"{ch}.bin")
            if not os.path.exists(wp):
                print(f"  {ch}: no wav — skip"); continue
            lag, peak, _ = xcorr_lag(aud_env(wp, args.dur), cap_env(bp, args.dur), search_s=args.search)
            meas.append((ch, lag, peak))
            if i % 25 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)}  {ch} lag={lag:.0f} peak={peak:.2f}")
        # median fallback: low-confidence chunks get the global median lag, not their noisy one
        med = float(np.median([m[1] for m in meas]))
        LOW = 0.40
        rows = [("chunk", "lag_ms_written", "measured_lag_ms", "peak", "fallback")]
        n_fb = 0
        for ch, lag, peak in meas:
            use = lag if peak >= LOW else med
            fb = peak < LOW
            n_fb += fb
            open(os.path.join(bin_dir, f"{ch}.lag"), "w").write(f"{use:.1f}\n")
            rows.append((ch, round(use, 1), round(lag, 1), round(peak, 3), fb))
        with open(os.path.join(bin_dir, "lag_manifest.csv"), "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[apply] wrote {len(meas)} .lag sidecars "
              f"(median {med:.0f} ms; {n_fb} low-peak<{LOW} used median fallback) "
              f"+ lag_manifest.csv")


if __name__ == "__main__":
    main()
