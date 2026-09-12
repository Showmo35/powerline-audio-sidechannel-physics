#!/usr/bin/env python3
"""
estimate_lag.py — Measure the powerline-vs-audio lag for soundbar bin captures
                  from RMS envelopes alone, and write the .lag sidecar.

Physical model
--------------
Each chunk_XXX.bin is a 200 kHz float32 powerline capture that has already been
clipped to the matching reference-audio chunk (chunk_XXX.wav, 16 kHz mono).
The powerline signal is still physically *delayed* behind the audio (DSP /
power-supply latency).  We recover that delay purely from the frame-level RMS
ENVELOPES of the two signals (no spectral content, no ffmpeg):

  1. Frame both signals at FRAME_S (50 ms) and take per-frame RMS.
  2. z-normalise the two envelopes.
  3. Cross-correlate.  np.correlate(a, c) peaks at NEGATIVE numpy-lag when c
     (capture) is delayed behind a (audio).  Physical lag = -numpy_lag.
  4. Search numpy-lag in [-MAX_LAG_S, 0] so the powerline-delayed peak can't
     alias onto the repetition period of the stimulus.
  5. lag_ms = phys_lag_frames * FRAME_S * 1000   (granularity = 50 ms)

This reproduces the align_lag() logic in simple_rx.py without trimming the bin:
the measured value is written to chunk_XXX.lag and analysis scripts apply it.

Usage
-----
  # validate against the 46 known-good lags (no files written):
  python3 estimate_lag.py --validate

  # measure + write .lag for every bin whose sidecar is missing or 0.0:
  python3 estimate_lag.py --apply

  # measure a single chunk (prints, no write unless --apply):
  python3 estimate_lag.py --chunk 47
"""
import argparse, os, sys, wave
import numpy as np
from scipy.signal import correlate as _xcorr

# ── paths ──────────────────────────────────────────────────────────────────
ROOT     = "<REPO_ROOT>/data"
BIN_DIR  = os.path.join(ROOT, "soundbar_bin_captures")
AUD_DIR  = os.path.join(ROOT, "audio_chunks")

# ── constants ──────────────────────────────────────────────────────────────
CAP_SR    = 200_000     # powerline capture sample rate (Hz), float32
FRAME_S   = 0.05        # 50 ms RMS frame — smooth/robust; parabolic refine adds sub-frame res
MAX_LAG_S = 0.700       # search window for the physical delay [0 .. 700] ms
CROP_S    = 60.0        # only the first 60 s of each chunk is enough for lag


def cap_envelope(bin_path, frame_s=FRAME_S, samp_rate=CAP_SR, crop_s=CROP_S):
    """Per-frame RMS of the leading `crop_s` of the 200 kHz float32 capture.

    Only the first crop_s seconds are read (memmap slice), so a 1.44 GB / 30 min
    chunk costs ~48 MB of I/O instead of the whole file.
    """
    frame = int(round(frame_s * samp_rate))           # 10 000 samples / frame
    n_read = int(round(crop_s * samp_rate)) if crop_s else None
    mm = np.memmap(bin_path, dtype=np.float32, mode="r")
    seg = mm[:n_read] if n_read else mm
    n_frames = len(seg) // frame
    x = np.asarray(seg[:n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    env = np.sqrt(np.mean(x * x, axis=1))
    del mm
    return env


def aud_envelope(wav_path, frame_s=FRAME_S, crop_s=CROP_S):
    """Per-frame RMS of the leading `crop_s` of the 16-bit mono reference wav."""
    w = wave.open(wav_path, "rb")
    sr = w.getframerate()
    n  = int(round(crop_s * sr)) if crop_s else w.getnframes()
    n  = min(n, w.getnframes())
    raw = w.readframes(n)
    w.close()
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    frame = int(round(frame_s * sr))
    n_frames = len(x) // frame
    x = x[:n_frames * frame].reshape(n_frames, frame)
    return np.sqrt(np.mean(x * x, axis=1))


def estimate_lag(aud_env, cap_env, frame_s=FRAME_S, max_lag_s=MAX_LAG_S):
    """Return (lag_ms, diagnostics) from the two RMS envelopes.

    The discrete cross-correlation peak gives the lag to within one frame; a
    parabolic fit to the peak and its two neighbours refines it to a continuous
    sub-frame value (≈1 ms), so the result is not snapped to the frame grid.
    """
    n = min(len(aud_env), len(cap_env))
    a = aud_env[:n]
    c = cap_env[:n]
    a = (a - a.mean()) / (a.std() + 1e-12)
    c = (c - c.mean()) / (c.std() + 1e-12)
    xc = _xcorr(a, c, mode="full", method="fft") / n
    lags = np.arange(-(n - 1), n)
    max_lag_frames = int(round(max_lag_s / frame_s))
    valid = (-max_lag_frames <= lags) & (lags <= 0)   # physical delay → neg numpy lag
    idx = np.flatnonzero(valid)
    k = int(idx[np.argmax(xc[idx])])                  # global index of discrete peak

    # ── parabolic sub-frame refinement around the discrete peak ───────────────
    if 0 < k < len(xc) - 1:
        y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    numpy_lag_cont = lags[k] + delta
    lag_ms = max(0.0, -numpy_lag_cont * frame_s * 1000.0)   # physical lag ≥ 0
    peak = float(xc[k])
    return lag_ms, {"n_frames": n, "numpy_lag": int(lags[k]), "delta": delta,
                    "lag_ms": lag_ms, "peak_xcorr": peak, "lags": lags, "xcorr": xc,
                    "aud_env": a, "cap_env": c, "frame_s": frame_s}


def read_sidecar(chunk):
    p = os.path.join(BIN_DIR, f"{chunk}.lag")
    if os.path.exists(p):
        try:
            return float(open(p).read().strip())
        except Exception:
            return None
    return None


def all_chunks():
    bins = sorted(f[:-4] for f in os.listdir(BIN_DIR) if f.endswith(".bin"))
    return bins


def measure_chunk(chunk, crop_s=CROP_S, frame_s=FRAME_S):
    bin_path = os.path.join(BIN_DIR, f"{chunk}.bin")
    wav_path = os.path.join(AUD_DIR, f"{chunk}.wav")
    if not os.path.exists(bin_path) or not os.path.exists(wav_path):
        return None
    ce = cap_envelope(bin_path, frame_s=frame_s, crop_s=crop_s)
    ae = aud_envelope(wav_path, frame_s=frame_s, crop_s=crop_s)
    return estimate_lag(ae, ce, frame_s=frame_s)


# ── modes ──────────────────────────────────────────────────────────────────
def cmd_validate(args):
    """Compare measured lag against the known-good (non-zero) sidecars."""
    known = [(c, read_sidecar(c)) for c in all_chunks()]
    known = [(c, v) for c, v in known if v not in (None, 0.0)]
    print(f"Validating against {len(known)} known-good (non-zero) sidecars\n")
    print(f"{'chunk':<12}{'known':>8}{'measured':>10}{'diff':>8}  {'peak':>6}  ok")
    rows = []
    exact = within1 = 0
    for chunk, kv in known:
        res = measure_chunk(chunk, crop_s=args.dur, frame_s=args.frame)
        if res is None:
            print(f"{chunk:<12}{'--- missing data ---'}")
            continue
        mv, d = res
        diff = mv - kv
        ok = abs(diff) < 1e-6
        near = abs(diff) <= FRAME_S * 1000 + 1e-6     # within one 50 ms frame
        exact += ok
        within1 += near
        rows.append((chunk, kv, mv, diff, d["peak_xcorr"]))
        print(f"{chunk:<12}{kv:>8.0f}{mv:>10.0f}{diff:>8.0f}  {d['peak_xcorr']:>6.2f}  "
              f"{'✓' if ok else ('~' if near else '✗')}")
    nt = len(rows)
    print(f"\nExact match : {exact}/{nt} ({100*exact/nt:.0f}%)")
    print(f"Within ±50ms: {within1}/{nt} ({100*within1/nt:.0f}%)")
    if args.viz:
        _plot_validation(rows, args.viz)
    return rows


def cmd_apply(args):
    targets = []
    for c in all_chunks():
        v = read_sidecar(c)
        if args.force or v in (None, 0.0):
            targets.append(c)
    print(f"Measuring + writing .lag for {len(targets)} chunks "
          f"({'all (--force)' if args.force else 'missing/zero only'})\n")
    manifest = os.path.join(BIN_DIR, "lag_manifest.csv")
    rows = [("chunk", "old_lag_ms", "new_lag_ms", "peak_xcorr", "n_frames")]
    for i, chunk in enumerate(targets, 1):
        old = read_sidecar(chunk)
        res = measure_chunk(chunk, crop_s=args.dur, frame_s=args.frame)
        if res is None:
            print(f"[{i}/{len(targets)}] {chunk}  SKIP (missing bin or wav)")
            continue
        mv, d = res
        lag_path = os.path.join(BIN_DIR, f"{chunk}.lag")
        if not args.dry_run:
            open(lag_path, "w").write(f"{mv:.1f}\n")
        rows.append((chunk, old, round(mv, 1), round(d["peak_xcorr"], 3), d["n_frames"]))
        print(f"[{i}/{len(targets)}] {chunk}  old={str(old):>6}  new={mv:>6.1f} ms  "
              f"peak={d['peak_xcorr']:.2f}  {'(dry-run)' if args.dry_run else '→ written'}")
    if not args.dry_run:
        import csv
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"\n[manifest] old→new values saved to {manifest}")


def cmd_chunk(args):
    chunk = f"chunk_{int(args.chunk):03d}"
    res = measure_chunk(chunk, crop_s=args.dur, frame_s=args.frame)
    if res is None:
        sys.exit(f"{chunk}: missing bin or wav")
    mv, d = res
    kv = read_sidecar(chunk)
    print(f"{chunk}: measured={mv:.0f} ms  sidecar={kv}  peak={d['peak_xcorr']:.2f}")
    if args.apply and not args.dry_run:
        open(os.path.join(BIN_DIR, f"{chunk}.lag"), "w").write(f"{mv:.1f}\n")
        print("  → written")
    if args.viz:
        _plot_single(chunk, d, mv, kv, args.viz)


def cmd_proof(args):
    """Measure every chunk, then visualise the lag distribution and overlay a
    sample of aligned envelopes as visual proof the alignment is correct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = all_chunks()
    print(f"Measuring {len(chunks)} chunks (crop {args.dur:.0f}s)…")
    recs = []   # (chunk, lag_ms, peak, diag)
    for i, ch in enumerate(chunks, 1):
        res = measure_chunk(ch, crop_s=args.dur, frame_s=args.frame)
        if res is None:
            continue
        mv, d = res
        recs.append((ch, mv, d["peak_xcorr"], d))
        if i % 25 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)}")
    lags = np.array([r[1] for r in recs])
    peaks = np.array([r[2] for r in recs])

    # pick 6 sample chunks spanning the lag range (highest peak within each bin)
    order = np.argsort(lags)
    picks = []
    for q in np.linspace(0, len(order) - 1, 6):
        picks.append(recs[order[int(round(q))]])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    # row 0: distribution + quality
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(lags, bins=np.arange(0, lags.max() + 25, 20), color="tab:blue", edgecolor="k")
    ax.set_xlabel("measured lag (ms)"); ax.set_ylabel("# chunks")
    ax.set_title(f"Lag distribution  (n={len(lags)}, median {np.median(lags):.0f} ms)")
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(lags, peaks, s=14, alpha=.6)
    ax.set_xlabel("lag (ms)"); ax.set_ylabel("peak xcorr")
    ax.set_title(f"Alignment quality  (median peak {np.median(peaks):.2f})")
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(np.arange(len(recs)), lags, ".", ms=4)
    ax.set_xlabel("chunk order"); ax.set_ylabel("lag (ms)"); ax.set_title("Lag per chunk")

    # rows 1-3: 6 envelope-overlay proof panels (zoom to first 25 s)
    for n, (ch, mv, pk, d) in enumerate(picks):
        ax = fig.add_subplot(gs[1 + n // 3, n % 3])
        fs = d["frame_s"]; t = np.arange(d["n_frames"]) * fs
        shift = int(round(mv / (fs * 1000)))
        # capture is DELAYED behind audio by `lag`; shift it LEFT (earlier) to align
        cap_sh = np.full_like(d["cap_env"], np.nan)
        if shift > 0:
            cap_sh[:len(cap_sh) - shift] = d["cap_env"][shift:]
        else:
            cap_sh = d["cap_env"].copy()
        ax.plot(t, d["aud_env"], lw=.7, label="audio")
        ax.plot(t, cap_sh, lw=.7, label=f"capture +{mv:.0f}ms")
        ax.set_xlim(0, min(25, t[-1])); ax.set_ylim(-2, 5)
        ax.set_title(f"{ch}  lag={mv:.0f}ms  peak={pk:.2f}", fontsize=10)
        ax.set_xlabel("s"); ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Proper RMS-envelope lag alignment — distribution + aligned overlays",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.viz, dpi=120)
    print(f"\n[viz] wrote {args.viz}")
    print(f"lag  : min {lags.min():.0f}  median {np.median(lags):.0f}  "
          f"max {lags.max():.0f} ms")
    print(f"peak : min {peaks.min():.2f}  median {np.median(peaks):.2f}")


# ── plots ──────────────────────────────────────────────────────────────────
def _plot_validation(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chunks = [r[0] for r in rows]
    known  = np.array([r[1] for r in rows])
    meas   = np.array([r[2] for r in rows])
    peak   = np.array([r[4] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # scatter known vs measured
    jit = (np.random.RandomState(0).rand(len(known)) - .5) * 8
    ax[0].plot([0, 400], [0, 400], "k--", lw=1, alpha=.5)
    sc = ax[0].scatter(known + jit, meas + jit, c=peak, cmap="viridis",
                       s=60, edgecolor="k", linewidth=.4)
    ax[0].set_xlabel("known lag (ms)"); ax[0].set_ylabel("measured lag (ms)")
    ax[0].set_title("Measured vs known (jittered)"); ax[0].set_aspect("equal")
    plt.colorbar(sc, ax=ax[0], label="peak xcorr")

    # per-chunk diff
    diff = meas - known
    idx = np.arange(len(chunks))
    ax[1].bar(idx, diff, color=np.where(diff == 0, "tab:green",
              np.where(np.abs(diff) <= 50, "tab:orange", "tab:red")))
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("chunk index (001..046)"); ax[1].set_ylabel("measured − known (ms)")
    ax[1].set_title("Per-chunk error")

    # histogram of error
    bins = np.arange(diff.min() - 25, diff.max() + 75, 50)
    ax[2].hist(diff, bins=bins, color="tab:blue", edgecolor="k")
    ax[2].set_xlabel("error (ms)"); ax[2].set_ylabel("# chunks")
    exact = int((diff == 0).sum()); near = int((np.abs(diff) <= 50).sum())
    ax[2].set_title(f"Error dist  (exact {exact}/{len(diff)}, ±50ms {near}/{len(diff)})")

    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\n[viz] wrote {out}")


def _plot_single(chunk, d, mv, kv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    fs = d.get("frame_s", FRAME_S)
    t = np.arange(d["n_frames"]) * fs
    shift = int(round(mv / (fs * 1000)))
    # capture is delayed behind audio → shift LEFT (earlier) to align
    cap_sh = np.r_[d["cap_env"][shift:], np.full(shift, np.nan)] if shift > 0 else d["cap_env"]
    ax[0].plot(t, d["aud_env"], label="audio env", lw=.8)
    ax[0].plot(t, d["cap_env"], label="capture env (raw)", lw=.8, alpha=.6)
    ax[0].plot(t, cap_sh, label=f"capture env (shifted +{mv:.0f} ms)", lw=.8)
    ax[0].set_xlim(0, min(120, t[-1])); ax[0].legend(loc="upper right")
    ax[0].set_title(f"{chunk}  measured lag {mv:.0f} ms  (sidecar {kv})  — first 120 s")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("z-RMS")

    lags_ms = d["lags"] * FRAME_S * 1000
    m = (lags_ms >= -MAX_LAG_S * 1000) & (lags_ms <= 50)
    ax[1].plot(-lags_ms[m], d["xcorr"][m])      # x = physical lag (ms)
    ax[1].axvline(mv, color="r", ls="--", label=f"peak {mv:.0f} ms")
    ax[1].set_xlabel("physical lag (ms)"); ax[1].set_ylabel("xcorr")
    ax[1].set_title("RMS-envelope cross-correlation"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="check vs 46 known lags")
    ap.add_argument("--proof", action="store_true",
                    help="measure all chunks + write a distribution/overlay PNG (needs --viz)")
    ap.add_argument("--apply", action="store_true", help="write .lag for missing/zero")
    ap.add_argument("--force", action="store_true", help="with --apply: rewrite all")
    ap.add_argument("--dry-run", action="store_true", help="measure but don't write")
    ap.add_argument("--chunk", type=int, help="measure a single chunk number")
    ap.add_argument("--dur", type=float, default=CROP_S,
                    help=f"seconds to read from the start of each chunk (default {CROP_S:.0f}; 0 = full)")
    ap.add_argument("--frame", type=float, default=FRAME_S,
                    help=f"RMS frame size in seconds (default {FRAME_S}; larger = smoother/robust)")
    ap.add_argument("--viz", type=str, default=None, help="output PNG path for plots")
    args = ap.parse_args()

    if args.chunk is not None:
        cmd_chunk(args)
    elif args.proof:
        if not args.viz:
            ap.error("--proof requires --viz <output.png>")
        cmd_proof(args)
    elif args.validate:
        cmd_validate(args)
    elif args.apply:
        cmd_apply(args)
    else:
        ap.print_help()
