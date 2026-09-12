"""
inference.py
------------
Load a trained PowerlineUNet checkpoint and reconstruct audio from a powerline
IQ-demodulated signal. Spectrograms are in Whisper's exact format (16 kHz,
80 mel bins) so predictions can be fed directly to Whisper for transcription.

Usage:
    python inference.py --mode full     # full chapter inference from .bin
    python inference.py --mode val      # evaluate on val split from train_data.npz
"""

import os, sys, argparse
import numpy as np
import torch
import whisper
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import PowerlineUNet

# ── constants ─────────────────────────────────────────────────────────────────
BASE        = "<REPO_ROOT>"
CKPT_PATH   = os.path.join(BASE, "UNet", "checkpoints", "best_model.pt")
PL_BIN      = os.path.join(BASE, "May29_Alice", "Chap_4_real.bin")
MP3_FILE    = os.path.join(BASE, "Alice_In_Wonderland_mp3", "Alice_In_Wonderland_ch_04.mp3")
OUT_DIR     = os.path.join(BASE, "UNet", "inference_output")
os.makedirs(OUT_DIR, exist_ok=True)

FS          = 200_000
AUDIO_SR    = 16_000       # Whisper's sample rate
FC_CARRIER  = 20_433.35
OFFSET_SEC  = 1.95
N_FFT       = 400          # Whisper STFT params
HOP         = 160
N_MELS      = 80
WIN_SEC     = 1.0
HOP_SEC     = 0.125


# ── signal processing helpers ─────────────────────────────────────────────────

def butter_lowpass(x, fs, cutoff, order=5):
    sos = butter(order, cutoff / (fs / 2), btype='low', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def iq_demodulate(sig, fs, fc, audio_bw=8000):
    t = np.arange(len(sig), dtype=np.float64) / fs
    sig = sig.astype(np.float64)
    I = butter_lowpass(sig * np.cos(2*np.pi*fc*t), fs, audio_bw)
    Q = butter_lowpass(sig * np.sin(2*np.pi*fc*t), fs, audio_bw)
    env = np.abs(I + 1j*Q).astype(np.float32)
    sos_hp = butter(4, 30 / (fs / 2), btype='high', output='sos')
    return sosfiltfilt(sos_hp, env).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out//g, fs_in//g).astype(np.float32)


# ── Whisper-format mel spectrogram ───────────────────────────────────────────

_WHISPER_FB = None

def _get_whisper_filterbank():
    global _WHISPER_FB
    if _WHISPER_FB is None:
        _WHISPER_FB = whisper.audio.mel_filters(
            torch.device('cpu'), N_MELS).numpy()
    return _WHISPER_FB


def log_mel_spec(x, sr=AUDIO_SR):
    """Compute Whisper-format log-mel spectrogram. Returns (80, T)."""
    fb  = _get_whisper_filterbank()
    win = np.hanning(N_FFT)
    nf  = (len(x) - N_FFT) // HOP + 1
    frames = np.stack([x[i*HOP:i*HOP+N_FFT] * win for i in range(nf)])
    mag_sq = np.abs(np.fft.rfft(frames, axis=-1))**2
    mel    = fb @ mag_sq.T

    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def spec_to_audio_griffin_lim(spec, sr=AUDIO_SR, n_iter=60):
    """Approximate inversion of Whisper-format log-mel via Griffin-Lim."""
    # Undo Whisper log scaling: spec = (log10(mel) + 4) / 4
    log_mel = spec * 4.0 - 4.0
    mel_mag = np.power(10.0, log_mel).clip(0)

    # Pseudo-inverse of Whisper mel filterbank -> linear magnitude
    fb = _get_whisper_filterbank()
    fb_pinv = np.linalg.pinv(fb)
    lin_mag = (fb_pinv @ mel_mag).clip(0)
    lin_mag = np.sqrt(lin_mag)  # Whisper uses power spectrum, Griffin-Lim needs magnitude
    lin_mag = np.nan_to_num(lin_mag, nan=0.0, posinf=0.0, neginf=0.0)

    audio = librosa.griffinlim(
        lin_mag, n_iter=n_iter, hop_length=HOP,
        win_length=N_FFT, window='hann', momentum=0.99,
    )
    return audio.astype(np.float32)


# ── full inference ───────────────────────────────────────────────────────────

def run_inference(pl_bin_path, mp3_path, ckpt_path, out_dir, offset_sec=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    print(f"  Best val loss : {ckpt['val_loss']:.5f}  (epoch {ckpt['epoch']})")

    model = PowerlineUNet(base_ch=args.get('base_ch', 64)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  ({model.count_params():,} params)")

    print(f"\nLoading powerline bin: {pl_bin_path}")
    pl = np.fromfile(pl_bin_path, dtype=np.float32).astype(np.float64)
    offset = offset_sec if offset_sec is not None else OFFSET_SEC
    print(f"  Using offset: {offset}s")
    start_samp = int(offset * FS)
    pl = pl[start_samp:]
    print(f"  Trimmed to {len(pl)/FS:.1f}s of chapter content")

    print(f"Loading MP3 ground truth: {mp3_path}")
    mp3, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3 = resample_to(mp3, mp3_sr, AUDIO_SR)

    print("IQ demodulating powerline...")
    pl_demod = iq_demodulate(pl, FS, FC_CARRIER)
    pl_16k   = resample_to(pl_demod, FS, AUDIO_SR)
    L        = min(len(pl_16k), len(mp3))
    pl_16k   = pl_16k[:L];  mp3_al = mp3[:L]

    def rms_norm(x, r=0.05): return x * r / (np.sqrt(np.mean(x**2)) + 1e-9)
    pl_16k = rms_norm(pl_16k); mp3_al = rms_norm(mp3_al)

    win_s = int(WIN_SEC * AUDIO_SR)
    hop_s = int(HOP_SEC * AUDIO_SR)
    starts = list(range(0, L - win_s, hop_s))

    pred_specs  = []
    noisy_specs = []
    print(f"Running model on {len(starts)} windows...")
    with torch.no_grad():
        for s in starts:
            seg  = pl_16k[s:s+win_s]
            spec = log_mel_spec(seg)
            noisy_specs.append(spec)
            inp  = torch.from_numpy(spec[np.newaxis, np.newaxis]).to(device)
            out  = model(inp).squeeze().cpu().numpy()
            pred_specs.append(out)

    # Overlap-add in spectrogram domain
    T_frames = pred_specs[0].shape[1]
    hop_frames = int(HOP_SEC * AUDIO_SR / HOP)
    total_frames = (len(starts) - 1) * hop_frames + T_frames

    def spec_overlap_add(specs):
        acc   = np.zeros((N_MELS, total_frames), dtype=np.float32)
        count = np.zeros(total_frames, dtype=np.float32)
        for i, sp in enumerate(specs):
            t0 = i * hop_frames
            t1 = min(t0 + T_frames, total_frames)
            acc[:, t0:t1]   += sp[:, :t1 - t0]
            count[t0:t1]    += 1
        count = np.maximum(count, 1)
        return acc / count

    print("Building full spectrograms via overlap-add...")
    full_noisy = spec_overlap_add(noisy_specs)
    full_pred  = spec_overlap_add(pred_specs)

    print("Computing ground-truth mel spectrogram...")
    full_clean = log_mel_spec(mp3_al)

    for name, arr in [('full_noisy', full_noisy), ('full_pred', full_pred), ('full_clean', full_clean)]:
        print(f"  {name}: shape={arr.shape}  min={np.nanmin(arr):.3f}  max={np.nanmax(arr):.3f}"
              f"  NaN={np.isnan(arr).sum()}  Inf={np.isinf(arr).sum()}")

    full_noisy = np.nan_to_num(full_noisy, nan=0.0, posinf=0.0, neginf=0.0)
    full_pred  = np.nan_to_num(full_pred,  nan=0.0, posinf=0.0, neginf=0.0)
    full_clean = np.nan_to_num(full_clean, nan=0.0, posinf=0.0, neginf=0.0)

    display_frames = min(full_noisy.shape[1], full_pred.shape[1], full_clean.shape[1])
    max_frames = int(30 * AUDIO_SR / HOP)
    display_frames = min(display_frames, max_frames)

    all_data = np.concatenate([full_noisy[:, :display_frames].ravel(),
                                full_pred[:, :display_frames].ravel(),
                                full_clean[:, :display_frames].ravel()])
    vmin = np.percentile(all_data, 2)
    vmax = np.percentile(all_data, 98)
    print(f"  Color range: vmin={vmin:.3f}  vmax={vmax:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    extent = [0, display_frames * HOP / AUDIO_SR, 0, N_MELS]

    axes[0].imshow(full_noisy[:, :display_frames], aspect='auto', origin='lower',
                   extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Powerline IQ — Whisper Log-Mel Spectrogram')
    axes[0].set_ylabel('Mel bin')

    axes[1].imshow(full_pred[:, :display_frames], aspect='auto', origin='lower',
                   extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title('Model Prediction — Whisper Log-Mel Spectrogram')
    axes[1].set_ylabel('Mel bin')

    axes[2].imshow(full_clean[:, :display_frames], aspect='auto', origin='lower',
                   extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title('MP3 Ground Truth — Whisper Log-Mel Spectrogram')
    axes[2].set_ylabel('Mel bin')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "spectrogram_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved -> {plot_path}")

    n = min(full_pred.shape[1], full_clean.shape[1])
    mse = np.mean((full_pred[:, :n] - full_clean[:, :n])**2)
    mse_noisy = np.mean((full_noisy[:, :n] - full_clean[:, :n])**2)
    print(f"\nSpectrogram MSE  — Noisy vs Clean : {mse_noisy:.4f}")
    print(f"Spectrogram MSE  — Pred  vs Clean : {mse:.4f}")

    # Reconstruct audio for a random 10s segment
    seg_sec   = 10
    seg_frames = int(seg_sec * AUDIO_SR / HOP)
    common_frames = min(full_noisy.shape[1], full_pred.shape[1], full_clean.shape[1])
    max_start = common_frames - seg_frames
    rng = np.random.default_rng(42)
    seg_start = rng.integers(0, max(1, max_start))
    seg_end   = seg_start + seg_frames
    seg_time  = seg_start * HOP / AUDIO_SR
    print(f"\nReconstructing audio for 10s segment starting at {seg_time:.1f}s (Griffin-Lim)...")

    audio_pred  = spec_to_audio_griffin_lim(full_pred[:, seg_start:seg_end],  sr=AUDIO_SR, n_iter=60)
    audio_noisy = spec_to_audio_griffin_lim(full_noisy[:, seg_start:seg_end], sr=AUDIO_SR, n_iter=60)
    audio_clean = spec_to_audio_griffin_lim(full_clean[:, seg_start:seg_end], sr=AUDIO_SR, n_iter=60)

    def peak_norm(x, target=0.9):
        return x * target / (np.max(np.abs(x)) + 1e-9)

    sf.write(os.path.join(out_dir, "reconstructed_audio.wav"),   peak_norm(audio_pred),  AUDIO_SR)
    sf.write(os.path.join(out_dir, "powerline_demodulated.wav"), peak_norm(audio_noisy), AUDIO_SR)
    sf.write(os.path.join(out_dir, "mp3_ground_truth.wav"),      peak_norm(audio_clean), AUDIO_SR)
    print(f"  reconstructed_audio.wav   ({len(audio_pred)/AUDIO_SR:.1f}s)")
    print(f"  powerline_demodulated.wav ({len(audio_noisy)/AUDIO_SR:.1f}s)")
    print(f"  mp3_ground_truth.wav      ({len(audio_clean)/AUDIO_SR:.1f}s)")

    print("Inference complete.")


# ── val inference ────────────────────────────────────────────────────────────

def run_val_inference(data_path, ckpt_path, out_dir, n_samples=10, whisper_model_name='base'):
    """Run model on validation split from train_data.npz (unseen during training)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    print(f"  Best val loss : {ckpt['val_loss']:.5f}  (epoch {ckpt['epoch']})")

    model = PowerlineUNet(base_ch=args.get('base_ch', 64)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  ({model.count_params():,} params)")

    print(f"Loading Whisper model: {whisper_model_name}")
    whisper_model = whisper.load_model(whisper_model_name, device=device)
    print("  Whisper loaded.")

    print(f"\nLoading val split from: {data_path}")
    data = np.load(data_path)
    noisy_val = data['noisy_val']
    clean_val = data['clean_val']
    print(f"  Val samples: {len(noisy_val)}")

    n = min(n_samples, len(noisy_val))
    rng = np.random.default_rng(42)
    start = rng.integers(0, max(1, len(noisy_val) - n))
    idxs = list(range(start, start + n))
    print(f"  Using {n} consecutive val samples: [{idxs[0]}..{idxs[-1]}]")

    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        inp = torch.from_numpy(noisy_val[idxs]).float().to(device)
        out = model(inp).cpu().numpy()
    print(f"  Prediction shape: {out.shape}")

    # Data is already in Whisper's log-mel scale — no de-normalization needed
    noisy_raw = noisy_val[idxs]
    pred_raw  = out
    clean_raw = clean_val[idxs]

    mse_noisy = np.mean((noisy_raw - clean_raw)**2)
    mse_pred  = np.mean((pred_raw  - clean_raw)**2)
    corrs = []
    for i in range(len(idxs)):
        r = np.corrcoef(pred_raw[i].flatten(), clean_raw[i].flatten())[0, 1]
        corrs.append(r)
    print(f"\nVal MSE  — Noisy vs Clean : {mse_noisy:.4f}")
    print(f"Val MSE  — Pred  vs Clean : {mse_pred:.4f}")
    print(f"Val Corr — Pred  vs Clean : {np.mean(corrs):.4f}  (per-sample: {[f'{r:.3f}' for r in corrs]})")

    # Concatenate all samples for plot and audio
    n_cat = len(idxs)
    noisy_cat = np.concatenate([noisy_raw[i, 0] for i in range(n_cat)], axis=1)
    pred_cat  = np.concatenate([pred_raw[i, 0]  for i in range(n_cat)], axis=1)
    clean_cat = np.concatenate([clean_raw[i, 0] for i in range(n_cat)], axis=1)
    total_sec = noisy_cat.shape[1] * HOP / AUDIO_SR
    print(f"\nConcatenated {n_cat} val samples -> {total_sec:.1f}s")

    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    vmin = min(noisy_cat.min(), pred_cat.min(), clean_cat.min())
    vmax = max(noisy_cat.max(), pred_cat.max(), clean_cat.max())
    extent = [0, total_sec, 0, N_MELS]
    kw = dict(aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
    axes[0].imshow(noisy_cat, **kw); axes[0].set_title(f'Noisy ({n_cat} val samples)'); axes[0].set_ylabel('Mel bin')
    axes[1].imshow(pred_cat,  **kw); axes[1].set_title('Model Prediction'); axes[1].set_ylabel('Mel bin')
    axes[2].imshow(clean_cat, **kw); axes[2].set_title('Clean Ground Truth'); axes[2].set_ylabel('Mel bin')
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "val_spectrogram_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved -> {plot_path}")

    print(f"Reconstructing {total_sec:.1f}s audio (Griffin-Lim)...")
    audio_pred  = spec_to_audio_griffin_lim(pred_cat,  sr=AUDIO_SR, n_iter=60)
    audio_noisy = spec_to_audio_griffin_lim(noisy_cat, sr=AUDIO_SR, n_iter=60)
    audio_clean = spec_to_audio_griffin_lim(clean_cat, sr=AUDIO_SR, n_iter=60)

    def peak_norm(x, target=0.9):
        return x * target / (np.max(np.abs(x)) + 1e-9)

    sf.write(os.path.join(out_dir, "val_reconstructed.wav"),  peak_norm(audio_pred),  AUDIO_SR)
    sf.write(os.path.join(out_dir, "val_noisy.wav"),          peak_norm(audio_noisy), AUDIO_SR)
    sf.write(os.path.join(out_dir, "val_clean.wav"),          peak_norm(audio_clean), AUDIO_SR)
    print(f"  val_reconstructed.wav  ({len(audio_pred)/AUDIO_SR:.1f}s)")
    print(f"  val_noisy.wav          ({len(audio_noisy)/AUDIO_SR:.1f}s)")
    print(f"  val_clean.wav          ({len(audio_clean)/AUDIO_SR:.1f}s)")

    # Transcribe all three spectrograms via Whisper
    print("\n" + "=" * 60)
    print("Transcribing via Whisper encoder (direct mel input)")
    print("=" * 60)

    transcripts = {}
    for name, spec in [('noisy', noisy_cat), ('predicted', pred_cat), ('clean', clean_cat)]:
        T = spec.shape[1]
        mel_padded = np.pad(spec, ((0, 0), (0, max(0, 3000 - T)))) if T < 3000 else spec[:, :3000]
        mel_t = torch.from_numpy(mel_padded).float().to(device)
        options = whisper.DecodingOptions(language='en', without_timestamps=True)
        result = whisper.decode(whisper_model, mel_t, options)
        text = result.text.strip()
        transcripts[name] = text
        print(f"  [{name}]: {text[:300]}{'...' if len(text) > 300 else ''}")

    ref = transcripts.get('clean', '')
    print(f"\n{'=' * 60}")
    print("Word Error Rate (vs clean ground truth spectrogram)")
    print(f"{'=' * 60}")
    for name, text in transcripts.items():
        if name == 'clean':
            continue
        wer = _wer(ref, text)
        print(f"  {name:20s}:  WER = {wer:.1%}  ({len(text.split())} words)")
    print(f"  {'clean (ref)':20s}:  {len(ref.split())} words")

    txt_path = os.path.join(out_dir, 'transcriptions.txt')
    with open(txt_path, 'w') as f:
        for name, text in transcripts.items():
            f.write(f"=== {name.upper()} ===\n{text}\n\n")
    print(f"\nTranscripts saved -> {txt_path}")
    print("Val inference complete.")


def run_segment_inference(pl_bin_path, mp3_path, ckpt_path, out_dir,
                          offset_sec=None, seg_start_sec=None, seg_dur_sec=30.0):
    """
    Process a short continuous segment of a chapter, produce overlap-added
    spectrograms, and feed them directly to Whisper for transcription.

    seg_start_sec defaults to the val region start (90% into chapter content),
    ensuring evaluation is always on unseen data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    print(f"  Best val loss : {ckpt['val_loss']:.5f}  (epoch {ckpt['epoch']})")

    model = PowerlineUNet(base_ch=args.get('base_ch', 64)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  ({model.count_params():,} params)")

    # Load Whisper
    print("Loading Whisper model: base")
    whisper_model = whisper.load_model('base', device=device)
    print("  Whisper loaded.")

    # Compute val-region start: 90% into chapter content (same split as training)
    offset = offset_sec if offset_sec is not None else OFFSET_SEC
    print(f"\nLoading powerline bin: {pl_bin_path}")
    pl_all = np.fromfile(pl_bin_path, dtype=np.float32)
    total_dur_sec    = len(pl_all) / FS
    content_dur_sec  = total_dur_sec - offset
    val_start_sec    = offset + 0.9 * content_dur_sec  # first sample not in training set

    if seg_start_sec is None:
        seg_start_sec = val_start_sec
        print(f"  Chapter duration : {total_dur_sec:.1f}s  (content after offset: {content_dur_sec:.1f}s)")
        print(f"  Auto val start   : {seg_start_sec:.1f}s  (90% mark, unseen during training)")
    else:
        if seg_start_sec < val_start_sec:
            print(f"  WARNING: seg_start={seg_start_sec:.1f}s is before val boundary "
                  f"({val_start_sec:.1f}s) — this segment overlaps with training data!")
        else:
            print(f"  seg_start={seg_start_sec:.1f}s  (val boundary at {val_start_sec:.1f}s — OK)")

    margin_sec = 2.0  # extra margin for filter settling
    raw_start = max(0, seg_start_sec - margin_sec)
    raw_dur   = seg_dur_sec + 2 * margin_sec

    s0 = int(raw_start * FS)
    s1 = int((raw_start + raw_dur) * FS)
    s1 = min(s1, len(pl_all))
    pl = pl_all[s0:s1].astype(np.float64)
    print(f"  Loaded {len(pl)/FS:.1f}s segment (with margin)")

    print(f"Loading MP3 ground truth: {mp3_path}")
    mp3, mp3_sr = librosa.load(mp3_path, sr=None, mono=True,
                                offset=seg_start_sec, duration=seg_dur_sec + 2)
    if mp3_sr != AUDIO_SR:
        mp3 = resample_to(mp3, mp3_sr, AUDIO_SR)

    print("IQ demodulating...")
    pl_demod = iq_demodulate(pl, FS, FC_CARRIER)
    pl_16k   = resample_to(pl_demod, FS, AUDIO_SR)

    # Trim margin after demodulation
    margin_samps = int(margin_sec * AUDIO_SR)
    if seg_start_sec >= margin_sec:
        pl_16k = pl_16k[margin_samps:]
    seg_samps = int(seg_dur_sec * AUDIO_SR)
    pl_16k = pl_16k[:seg_samps]
    mp3    = mp3[:seg_samps]
    L = min(len(pl_16k), len(mp3))
    pl_16k = pl_16k[:L]; mp3 = mp3[:L]

    def rms_norm(x, r=0.05): return x * r / (np.sqrt(np.mean(x**2)) + 1e-9)
    pl_16k = rms_norm(pl_16k); mp3 = rms_norm(mp3)

    # Sliding windows + overlap-add
    win_s = int(WIN_SEC * AUDIO_SR)
    hop_s = int(HOP_SEC * AUDIO_SR)
    starts = list(range(0, L - win_s, hop_s))

    pred_specs, noisy_specs = [], []
    print(f"Running model on {len(starts)} windows...")
    with torch.no_grad():
        for s in starts:
            seg = pl_16k[s:s+win_s]
            spec = log_mel_spec(seg)
            noisy_specs.append(spec)
            inp = torch.from_numpy(spec[np.newaxis, np.newaxis]).to(device)
            out = model(inp).squeeze().cpu().numpy()
            pred_specs.append(out)

    T_frames = pred_specs[0].shape[1]
    hop_frames = int(HOP_SEC * AUDIO_SR / HOP)
    total_frames = (len(starts) - 1) * hop_frames + T_frames

    def spec_overlap_add(specs):
        acc   = np.zeros((N_MELS, total_frames), dtype=np.float32)
        count = np.zeros(total_frames, dtype=np.float32)
        for i, sp in enumerate(specs):
            t0 = i * hop_frames
            t1 = min(t0 + T_frames, total_frames)
            acc[:, t0:t1]   += sp[:, :t1 - t0]
            count[t0:t1]    += 1
        count = np.maximum(count, 1)
        return acc / count

    print("Building spectrograms via overlap-add...")
    full_noisy = spec_overlap_add(noisy_specs)
    full_pred  = spec_overlap_add(pred_specs)
    full_clean = log_mel_spec(mp3)

    # Align lengths
    n_frames = min(full_noisy.shape[1], full_pred.shape[1], full_clean.shape[1])
    full_noisy = full_noisy[:, :n_frames]
    full_pred  = full_pred[:, :n_frames]
    full_clean = full_clean[:, :n_frames]

    # Diagnostics
    mse_noisy = np.mean((full_noisy - full_clean)**2)
    mse_pred  = np.mean((full_pred  - full_clean)**2)
    print(f"\nSpectrogram MSE — Noisy vs Clean : {mse_noisy:.4f}")
    print(f"Spectrogram MSE — Pred  vs Clean : {mse_pred:.4f}")

    for name, arr in [('noisy', full_noisy), ('pred', full_pred), ('clean', full_clean)]:
        print(f"  {name}: shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}")

    # Plot
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    extent = [seg_start_sec, seg_start_sec + n_frames * HOP / AUDIO_SR, 0, N_MELS]
    vmin = np.percentile(np.concatenate([full_noisy.ravel(), full_pred.ravel(), full_clean.ravel()]), 2)
    vmax = np.percentile(np.concatenate([full_noisy.ravel(), full_pred.ravel(), full_clean.ravel()]), 98)
    kw = dict(aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
    axes[0].imshow(full_noisy, **kw); axes[0].set_title('Noisy Powerline'); axes[0].set_ylabel('Mel bin')
    axes[1].imshow(full_pred,  **kw); axes[1].set_title('Model Prediction'); axes[1].set_ylabel('Mel bin')
    axes[2].imshow(full_clean, **kw); axes[2].set_title('MP3 Ground Truth'); axes[2].set_ylabel('Mel bin')
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "segment_spectrogram.png")
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"\nPlot saved -> {plot_path}")

    # Transcribe directly via Whisper (spectrograms are already in Whisper format)
    print("\n" + "=" * 60)
    print("Transcribing via Whisper encoder (direct mel input)")
    print("=" * 60)

    transcripts = {}
    for name, spec in [('noisy', full_noisy), ('predicted', full_pred), ('clean', full_clean)]:
        T = spec.shape[1]
        if T < 3000:
            mel_padded = np.pad(spec, ((0, 0), (0, 3000 - T)))
        else:
            mel_padded = spec[:, :3000]
        mel_t = torch.from_numpy(mel_padded).float().to(device)
        options = whisper.DecodingOptions(language='en', without_timestamps=True)
        result = whisper.decode(whisper_model, mel_t, options)
        text = result.text.strip()
        transcripts[name] = text
        print(f"  [{name}]: {text[:300]}{'...' if len(text) > 300 else ''}")

    # Also transcribe original MP3 audio for reference
    print(f"\nTranscribing original MP3 segment ({seg_start_sec:.0f}s-{seg_start_sec+seg_dur_sec:.0f}s)...")
    mp3_for_whisper = mp3[:int(seg_dur_sec * AUDIO_SR)]
    mp3_result = whisper_model.transcribe(
        torch.from_numpy(mp3_for_whisper).float(), language='en')
    mp3_text = mp3_result['text'].strip()
    transcripts['mp3_original'] = mp3_text
    print(f"  [mp3 original]: {mp3_text[:300]}{'...' if len(mp3_text) > 300 else ''}")

    # WER
    ref = transcripts.get('mp3_original', transcripts.get('clean', ''))
    print(f"\n{'=' * 60}")
    print("Word Error Rate (vs original MP3 transcription)")
    print(f"{'=' * 60}")
    for name, text in transcripts.items():
        if name == 'mp3_original':
            continue
        wer = _wer(ref, text)
        print(f"  {name:20s}:  WER = {wer:.1%}  ({len(text.split())} words)")
    print(f"  {'mp3_original (ref)':20s}:  {len(ref.split())} words")

    # Save transcripts
    txt_path = os.path.join(out_dir, 'transcriptions.txt')
    with open(txt_path, 'w') as f:
        for name, text in transcripts.items():
            f.write(f"=== {name.upper()} ===\n{text}\n\n")
    print(f"\nTranscripts saved -> {txt_path}")
    print("Segment inference complete.")


def _wer(ref, hyp):
    """Word Error Rate."""
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    r, h = len(ref_w), len(hyp_w)
    if r == 0: return float(h > 0)
    d = np.zeros((r+1, h+1), dtype=int)
    for i in range(r+1): d[i,0] = i
    for j in range(h+1): d[0,j] = j
    for i in range(1, r+1):
        for j in range(1, h+1):
            s = 0 if ref_w[i-1] == hyp_w[j-1] else 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+s)
    return d[r,h] / r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='segment', choices=['full', 'val', 'segment'])
    parser.add_argument('--pl',   default=PL_BIN)
    parser.add_argument('--mp3',  default=MP3_FILE)
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument('--out',  default=OUT_DIR)
    parser.add_argument('--offset', type=float, default=None)
    parser.add_argument('--seg_start', type=float, default=None,
                        help='Start time in seconds for segment mode (default: auto val boundary at 90%% of chapter)')
    parser.add_argument('--seg_dur', type=float, default=30.0,
                        help='Duration in seconds for segment mode')
    parser.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_data.npz'))
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--whisper_model', default='base', help='Whisper model size for transcription')
    args = parser.parse_args()

    if args.mode == 'val':
        run_val_inference(args.data, args.ckpt, args.out,
                          n_samples=args.n_samples,
                          whisper_model_name=args.whisper_model)
    elif args.mode == 'segment':
        run_segment_inference(args.pl, args.mp3, args.ckpt, args.out,
                              offset_sec=args.offset,
                              seg_start_sec=args.seg_start,
                              seg_dur_sec=args.seg_dur)
    else:
        run_inference(args.pl, args.mp3, args.ckpt, args.out, offset_sec=args.offset)
