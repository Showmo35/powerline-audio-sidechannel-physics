#!/usr/bin/env python3
"""
neural_vocoder.py — Neural vocoder reconstruction using pre-trained WaveRNN.

Pipeline:
  1. Downsample powerline capture (200 kHz) to 22050 Hz
  2. Apply lag correction (powerline lags audio by ~300 ms)
  3. Extract 80-bin mel-spectrogram matching WaveRNN training params
  4. Spectral subtraction: subtract silence-period noise baseline per mel bin
  5. Apply speech/silence mask — clamp silence frames to noise floor
  6. Normalise to match WaveRNN expected input range
  7. Run WaveRNN vocoder → waveform
  8. Save reconstructed WAV and 5-panel comparison figure

Usage:
    python3 neural_vocoder.py
    python3 neural_vocoder.py --no-show
    python3 neural_vocoder.py --lag-ms 300   # override auto-detected lag
"""

import argparse
import glob
import os
import subprocess
import sys
import wave

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as dsp
from scipy.stats import pearsonr
from math import gcd

import torch
import torchaudio
import torchaudio.transforms as T

# ── constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CAPTURE_PATH = os.path.join(SCRIPT_DIR, 'capture.bin')
AUDIO_EXTS   = ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a')
CAP_SR       = 200_000
AUD_SR       = 22_050
FRAME_S      = 0.05          # 50 ms frame for lag/speech detection

# WaveRNN mel params (must match LJSpeech training config)
MEL_N_FFT    = 2048
MEL_WIN      = 1100
MEL_HOP      = 275
MEL_N_MELS   = 80
MEL_FMIN     = 40.0
MEL_FMAX     = 11025.0

DARK  = '#0d0d14'
PANEL = '#1e1e2e'
GRID  = '#313244'
TEXT  = '#cdd6f4'
MUTED = '#a6adc8'
BLUE  = '#89b4fa'
RED   = '#f38ba8'
GRN   = '#a6e3a1'
ORNG  = '#fab387'
YLW   = '#f9e2af'
PURP  = '#cba6f7'

# ── I/O ───────────────────────────────────────────────────────────────────────

def find_audio(directory):
    for pat in AUDIO_EXTS:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def load_capture(path):
    return np.fromfile(path, dtype=np.float32)


def decode_audio(path, rate=AUD_SR):
    cmd = ['ffmpeg', '-v', 'quiet', '-i', path,
           '-f', 'f32le', '-ac', '1', '-ar', str(rate), 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def save_wav(path, data, rate):
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.9
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(rate); wf.writeframes(pcm.tobytes())
    print(f'  saved → {path}  ({os.path.getsize(path)//1024} KB)')

# ── signal processing ─────────────────────────────────────────────────────────

def downsample(x, src_sr, dst_sr):
    g = gcd(int(src_sr), int(dst_sr))
    return dsp.resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def frame_rms(x, sr, frame_s=FRAME_S):
    f = int(frame_s * sr)
    n = len(x) // f
    return np.sqrt(np.mean(x[:n*f].reshape(n, f)**2, axis=1))


def speech_mask(aud, sr, frame_s=FRAME_S, thr_frac=0.05):
    rms = frame_rms(aud, sr, frame_s)
    return rms > thr_frac * rms.max()

# ── mel extraction ────────────────────────────────────────────────────────────

def extract_mel(x_np, sr=AUD_SR):
    """Return log-mel-spectrogram as numpy array [n_mels, T]."""
    x_t = torch.from_numpy(x_np).unsqueeze(0)   # [1, T]
    mel_tf = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=MEL_N_FFT,
        win_length=MEL_WIN,
        hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        power=1.0,          # magnitude (not power) spectrogram
    )
    mel = mel_tf(x_t).squeeze(0).numpy()         # [n_mels, T]
    return mel


def process_mel(mel_raw, speech_m, aud_mel_ref, lag_delay=0, frame_s=FRAME_S):
    """
    Option B: minimal processing — keep the full noisy mel texture.

    Steps:
      1. Lag-shift mel frames to align with audio
      2. Log-scale with a safe floor
      3. Clip to [-11, 2] (Tacotron2/LJSpeech typical range)
      4. Per-bin normalise: shift each mel bin to match the reference audio
         mel's per-bin mean, preserving the powerline's relative variation
      5. Scale the per-frame amplitude by the powerline envelope (normalised
         by the reference audio envelope) so temporal bursts are preserved
    """
    T_mel = mel_raw.shape[1]
    frames_per_mel = frame_s * AUD_SR / MEL_HOP

    # ── 1. lag-shift mel frames
    lag_mel = int(round(lag_delay * frames_per_mel))
    if lag_mel > 0:
        mel = np.roll(mel_raw, -lag_mel, axis=1)
        mel[:, -lag_mel:] = mel_raw[:, -1:]
    elif lag_mel < 0:
        mel = np.roll(mel_raw, -lag_mel, axis=1)
        mel[:, :-lag_mel] = mel_raw[:, :1]
    else:
        mel = mel_raw.copy()

    # ── 2. mel-frame speech mask (for reporting only)
    mel_is_speech = np.zeros(T_mel, dtype=bool)
    for i, is_sp in enumerate(speech_m):
        lo = int(round(i * frames_per_mel))
        hi = int(round((i + 1) * frames_per_mel))
        mel_is_speech[lo:min(hi, T_mel)] = is_sp

    # ── 3. log-scale with a safe floor (no subtraction — keep texture)
    mel_log = np.log(np.maximum(mel, 1e-5))

    # ── 4. clip to Tacotron2 range
    mel_log = np.clip(mel_log, -11.0, 2.0)

    # ── 5. per-bin mean shift: align each mel bin's mean to reference audio
    ref_log     = np.log(np.maximum(aud_mel_ref, 1e-5))
    ref_log     = np.clip(ref_log, -11.0, 2.0)
    ref_bin_mean = ref_log.mean(axis=1, keepdims=True)   # [n_mels, 1]
    src_bin_mean = mel_log.mean(axis=1, keepdims=True)
    mel_log      = mel_log - src_bin_mean + ref_bin_mean

    # ── 6. scale per-frame amplitude by powerline envelope / ref envelope
    # Compute per-frame energy in log-mel space
    cap_env = mel_log.mean(axis=0)                       # [T]
    ref_env = ref_log.mean(axis=0)
    ref_env_interp = np.interp(
        np.linspace(0, 1, T_mel),
        np.linspace(0, 1, len(ref_env)),
        ref_env,
    )
    # Additive gain in log-mel = multiplicative in mel
    gain = ref_env_interp - cap_env                      # match ref energy per frame
    mel_log = mel_log + gain[np.newaxis, :]

    # final clip after gain
    mel_log = np.clip(mel_log, -11.0, 2.0)

    print(f'[mel]   shape={mel_log.shape}  '
          f'range=[{mel_log.min():.2f}, {mel_log.max():.2f}]  '
          f'speech_frames={int(mel_is_speech.sum())}')
    return mel_log.astype(np.float32), mel_is_speech

# ── WaveRNN inference ─────────────────────────────────────────────────────────

def run_wavernn(mel_np, device='cpu'):
    """Run pre-trained WaveRNN on a mel-spectrogram numpy array [n_mels, T].
    Returns waveform as float32 numpy array at AUD_SR (22050 Hz)."""
    print('[WaveRNN] loading pre-trained model (downloads on first run)…')
    bundle  = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    vocoder = bundle.get_vocoder().to(device)
    vocoder.eval()
    print('[WaveRNN] model loaded. Running inference…')

    mel_t = torch.from_numpy(mel_np).unsqueeze(0).to(device)  # [1, n_mels, T]
    lengths = torch.tensor([mel_t.shape[2]], dtype=torch.int64).to(device)

    with torch.no_grad():
        waveform, _ = vocoder(mel_t, lengths)

    wav = waveform.squeeze().cpu().numpy().astype(np.float32)
    print(f'[WaveRNN] output: {len(wav)/AUD_SR:.3f}s  peak={np.abs(wav).max():.4f}')
    return wav

# ── plot helpers ──────────────────────────────────────────────────────────────

def _ax(ax, title, xlabel, ylabel, xlim=None):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, loc='left', pad=4)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if xlim: ax.set_xlim(*xlim)


def plot_results(aud, cap_ds, recon, mel_proc, mel_is_speech,
                 r_env, peak_xcorr, audio_name, out_png):
    duration = len(aud) / AUD_SR
    xlim = (0, duration)
    t_aud   = np.linspace(0, duration, len(aud))
    t_recon = np.linspace(0, duration, len(recon))
    step    = max(1, len(aud) // 60_000)

    # Envelope comparison (50ms RMS)
    fr = int(FRAME_S * AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms  = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
    rec_rms  = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    t_fr     = np.arange(nf) * FRAME_S + FRAME_S / 2

    fig = plt.figure(figsize=(15, 14), facecolor=DARK)
    fig.suptitle(
        f'WaveRNN Neural Vocoder Reconstruction  —  "{audio_name}"\n'
        f'envelope r = {r_env:.3f}  |  '
        f'sample xcorr = {peak_xcorr:.4f}',
        color=TEXT, fontsize=11, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.3,
                           left=0.08, right=0.97, top=0.93, bottom=0.06)

    # ── Panel 1: audio mel-spectrogram ────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    mel_aud = extract_mel(aud)
    mel_aud_db = 20 * np.log10(mel_aud + 1e-5)
    t_mel = np.linspace(0, duration, mel_aud.shape[1])
    ax00.grid(False)
    ax00.pcolormesh(t_mel, np.arange(MEL_N_MELS), mel_aud_db,
                    shading='gouraud', cmap='magma',
                    vmin=np.percentile(mel_aud_db, 5),
                    vmax=np.percentile(mel_aud_db, 99), rasterized=True)
    _ax(ax00, 'Original audio mel-spectrogram (80-bin)', 'Time (s)', 'Mel bin', xlim=xlim)

    # ── Panel 2: processed powerline mel-spectrogram ──────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    t_cap_mel = np.linspace(0, duration, mel_proc.shape[1])
    ax01.grid(False)
    ax01.pcolormesh(t_cap_mel, np.arange(MEL_N_MELS), mel_proc,
                    shading='gouraud', cmap='inferno',
                    vmin=np.percentile(mel_proc, 5),
                    vmax=np.percentile(mel_proc, 99), rasterized=True)
    _ax(ax01, 'Processed powerline mel (normalised)',
        'Time (s)', 'Mel bin', xlim=xlim)

    # ── Panel 3: waveform overlay ─────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, :])
    aud_n   = aud   / (np.abs(aud).max()   + 1e-9)
    recon_n = recon / (np.abs(recon).max() + 1e-9)
    ax10.plot(t_aud[::step],   aud_n[::step],   color=BLUE, lw=0.6, alpha=0.7, label='Original audio')
    ax10.plot(t_recon[::step], recon_n[::step], color=PURP, lw=0.8, alpha=0.85, label='WaveRNN output')
    _ax(ax10, 'Waveform overlay', 'Time (s)', 'Norm. amplitude', xlim=xlim)
    ax10.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 4: envelope comparison ──────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, :])
    aud_rms_n = aud_rms / (aud_rms.max() + 1e-9)
    rec_rms_n = rec_rms / (rec_rms.max() + 1e-9)
    ax20.plot(t_fr, aud_rms_n, color=BLUE, lw=2.0, label='Audio RMS envelope')
    ax20.plot(t_fr, rec_rms_n, color=PURP, lw=2.0, alpha=0.9, label='WaveRNN RMS envelope')
    ax20.text(0.98, 0.92, f'Envelope r = {r_env:.3f}',
              transform=ax20.transAxes, color=YLW, fontsize=10, ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', fc=PANEL, ec=GRID))
    _ax(ax20, 'Envelope comparison (50 ms RMS)', 'Time (s)', 'Norm. RMS', xlim=xlim)
    ax20.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 5: recon mel-spectrogram ────────────────────────────────────────
    ax30 = fig.add_subplot(gs[3, 0])
    mel_rec = extract_mel(recon)
    mel_rec_db = 20 * np.log10(mel_rec + 1e-5)
    t_rec_mel = np.linspace(0, duration, mel_rec.shape[1])
    ax30.grid(False)
    ax30.pcolormesh(t_rec_mel, np.arange(MEL_N_MELS), mel_rec_db,
                    shading='gouraud', cmap='plasma',
                    vmin=np.percentile(mel_rec_db, 5),
                    vmax=np.percentile(mel_rec_db, 99), rasterized=True)
    _ax(ax30, 'WaveRNN output mel-spectrogram', 'Time (s)', 'Mel bin', xlim=xlim)

    # ── Panel 6: cross-correlation ────────────────────────────────────────────
    ax31 = fig.add_subplot(gs[3, 1])
    n_cmp = min(len(aud), len(recon))
    a_n = aud[:n_cmp]   / (np.abs(aud[:n_cmp]).max()   + 1e-9)
    r_n = recon[:n_cmp] / (np.abs(recon[:n_cmp]).max() + 1e-9)
    # FFT method: O(N log N) vs np.correlate's O(N^2) (stalls for long windows).
    xc  = dsp.correlate(a_n, r_n, mode='full', method='fft') / n_cmp
    lag_ms_xc = (np.arange(len(xc)) - n_cmp + 1) / AUD_SR * 1000
    zoom = np.abs(lag_ms_xc) <= 500
    ax31.plot(lag_ms_xc[zoom], xc[zoom], color=PURP, lw=0.8)
    pk = xc[zoom].max()
    pk_lag = lag_ms_xc[zoom][np.argmax(xc[zoom])]
    ax31.axvline(pk_lag, color=YLW, lw=1.5, ls='--', label=f'peak={pk:.4f} @ {pk_lag:.1f} ms')
    ax31.axhline(0, color=GRID, lw=0.6)
    _ax(ax31, 'Cross-correlation: audio vs WaveRNN', 'Lag (ms)', 'Xcorr', xlim=(-500, 500))
    ax31.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.patch.set_facecolor(DARK)
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f'[viz]   saved → {out_png}')

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture',  default=CAPTURE_PATH)
    ap.add_argument('--audio',    default=None)
    ap.add_argument('--no-show',  action='store_true')
    ap.add_argument('--device',   default='cpu', help='torch device (cpu/cuda)')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')

    # ── load ─────────────────────────────────────────────────────────────────
    cap = load_capture(args.capture)
    print(f'[cap]   {len(cap):,} samples @ {CAP_SR} Hz  ({len(cap)/CAP_SR:.3f}s)')

    audio_path = args.audio or find_audio(SCRIPT_DIR)
    if not audio_path:
        sys.exit('[error] no audio file found')
    aud = decode_audio(audio_path, AUD_SR)
    print(f'[audio] {len(aud):,} samples @ {AUD_SR} Hz  ({len(aud)/AUD_SR:.3f}s)')

    duration = min(len(cap) / CAP_SR, len(aud) / AUD_SR)
    cap = cap[:int(duration * CAP_SR)]
    aud = aud[:int(duration * AUD_SR)]

    # ── downsample capture to audio rate ──────────────────────────────────────
    print('[ds]    downsampling capture 200 kHz → 22050 Hz …')
    cap_ds = downsample(cap, CAP_SR, AUD_SR)
    cap_ds = cap_ds[:len(aud)]

    # ── lag: read sidecar or detect via xcorr ────────────────────────────────
    lag_path = os.path.splitext(args.capture)[0] + '.lag'
    if os.path.exists(lag_path):
        lag_ms = float(open(lag_path).read().strip())
        print(f'[lag]   read from sidecar: {lag_ms:+.0f} ms')
    else:
        _aud_rms = frame_rms(aud, AUD_SR)
        _cap_rms = frame_rms(cap_ds, AUD_SR)
        _n = min(len(_aud_rms), len(_cap_rms))
        _a = (_aud_rms[:_n] - _aud_rms[:_n].mean()) / (_aud_rms[:_n].std() + 1e-12)
        _c = (_cap_rms[:_n] - _cap_rms[:_n].mean()) / (_cap_rms[:_n].std() + 1e-12)
        _xc = np.correlate(_a, _c, mode='full') / _n
        _lags = np.arange(-(_n - 1), _n)
        _mf = int(round(0.700 / FRAME_S))
        _valid = (-_mf <= _lags) & (_lags <= 0)   # physical delay → negative numpy lag
        lag_ms = -float(_lags[_valid][np.argmax(_xc[_valid])]) * FRAME_S * 1000
        print(f'[lag]   auto-detected: {lag_ms:+.0f} ms')
    lag_delay = int(round(lag_ms / (FRAME_S * 1000)))

    # ── speech/silence mask (lag-corrected for powerline) ────────────────────
    sp_mask_audio = speech_mask(aud, AUD_SR)
    n_fr = len(sp_mask_audio)
    if lag_delay > 0:
        sp_mask_cap = np.r_[np.zeros(lag_delay, dtype=bool), sp_mask_audio[:-lag_delay]][:n_fr]
    elif lag_delay < 0:
        _d = -lag_delay
        sp_mask_cap = np.r_[sp_mask_audio[_d:], np.zeros(_d, dtype=bool)][:n_fr]
    else:
        sp_mask_cap = sp_mask_audio.copy()
    print(f'[mask]  speech={sp_mask_cap.sum()}  silence={(~sp_mask_cap).sum()} frames (lag-corrected)')

    # ── extract mel from downsampled capture + audio reference ────────────────
    print('[mel]   extracting mel-spectrogram from powerline capture …')
    mel_raw     = extract_mel(cap_ds)
    mel_aud_ref = extract_mel(aud)

    # ── process: normalise to audio mel stats ────────────────────────────────
    mel_proc, mel_is_speech = process_mel(mel_raw, sp_mask_cap, mel_aud_ref, lag_delay)

    # ── WaveRNN inference ─────────────────────────────────────────────────────
    recon = run_wavernn(mel_proc.astype(np.float32), device=args.device)

    # ── metrics ───────────────────────────────────────────────────────────────
    fr = int(FRAME_S * AUD_SR)
    nf = min(len(aud), len(recon)) // fr
    aud_rms_ev = np.array([np.sqrt(np.mean(aud[i*fr:(i+1)*fr]**2))   for i in range(nf)])
    rec_rms_ev = np.array([np.sqrt(np.mean(recon[i*fr:(i+1)*fr]**2)) for i in range(nf)])
    r_env, _ = pearsonr(aud_rms_ev, rec_rms_ev)

    n_cmp = min(len(aud), len(recon))
    a_n = aud[:n_cmp]   / (np.abs(aud[:n_cmp]).max()   + 1e-9)
    r_n = recon[:n_cmp] / (np.abs(recon[:n_cmp]).max() + 1e-9)
    xc  = np.correlate(a_n, r_n, mode='full') / n_cmp
    peak_xcorr = xc.max()

    print(f'[metric] envelope r = {r_env:.4f}')
    print(f'[metric] sample xcorr peak = {peak_xcorr:.4f}')

    # ── save ─────────────────────────────────────────────────────────────────
    print('[wav]   saving …')
    save_wav(os.path.join(SCRIPT_DIR, 'neural_vocoder_out.wav'), recon, AUD_SR)
    save_wav(os.path.join(SCRIPT_DIR, 'original_reference.wav'), aud,   AUD_SR)

    # ── plot ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(SCRIPT_DIR, 'neural_vocoder.png')
    plot_results(aud, cap_ds, recon, mel_proc, mel_is_speech,
                 r_env, peak_xcorr,
                 os.path.basename(audio_path), out_png)

    print()
    print('══ Results ══════════════════════════════════════')
    print(f'  Envelope r       = {r_env:.4f}')
    print(f'  Sample xcorr     = {peak_xcorr:.4f}')
    print()
    print('  Listen:')
    print('    ffplay -nodisp neural_vocoder_out.wav')
    print('    ffplay -nodisp original_reference.wav')
    print('═════════════════════════════════════════════════')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
