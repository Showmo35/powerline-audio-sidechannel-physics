"""
inference.py
------------
Run full-file reconstruction with FullSubNet+-style complex mask model.

Pipeline:
1) Load/align noisy waveform and optional clean reference.
2) Segment into overlapping 4s chunks.
3) Enhance each chunk in complex STFT domain.
4) Overlap-add enhanced STFT chunks.
5) ISTFT to reconstructed waveform.
6) Export waveform + mel spectrogram.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from audio_utils import (
    load_aligned_noisy_clean,
    match_chapter_name,
    mel_spectrogram_from_waveform,
)
from config import AUDIO_SR, DEFAULT_CHECKPOINT_DIR, DEFAULT_INFER_DIR, HOP_LENGTH, N_FFT, WIN_LENGTH, WINDOW_HOP_SEC, WINDOW_SEC
from losses import MultiResolutionSTFTLoss
from metrics import mel_mse_and_corr, reconstruction_metrics
from model import FullSubNetPlusLite
from stft_ops import apply_complex_ratio_mask, build_hann_window, stft_features, stft_to_waveform, waveform_to_stft


def _to_db(mel: np.ndarray):
    return 10.0 * np.log10(np.maximum(mel, 1e-12))


def _save_mel_comparison_plot(
    noisy_wave: np.ndarray,
    enhanced_wave: np.ndarray,
    clean_wave: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    out_png: str,
    plot_sec: float,
):
    n = min(len(noisy_wave), len(enhanced_wave), len(clean_wave))
    if plot_sec is not None and plot_sec > 0:
        n = min(n, int(plot_sec * sr))
    noisy_wave = noisy_wave[:n]
    enhanced_wave = enhanced_wave[:n]
    clean_wave = clean_wave[:n]

    noisy_mel = mel_spectrogram_from_waveform(
        noisy_wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    enhanced_mel = mel_spectrogram_from_waveform(
        enhanced_wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    clean_mel = mel_spectrogram_from_waveform(
        clean_wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    noisy_db = _to_db(noisy_mel)
    enhanced_db = _to_db(enhanced_mel)
    clean_db = _to_db(clean_mel)

    common_t = min(noisy_db.shape[1], enhanced_db.shape[1], clean_db.shape[1])
    noisy_db = noisy_db[:, :common_t]
    enhanced_db = enhanced_db[:, :common_t]
    clean_db = clean_db[:, :common_t]

    stacked = np.concatenate(
        [noisy_db.reshape(-1), enhanced_db.reshape(-1), clean_db.reshape(-1)]
    )
    vmin = float(np.percentile(stacked, 5))
    vmax = float(np.percentile(stacked, 95))

    total_sec = common_t * hop_length / sr
    extent = [0, total_sec, 0, n_mels]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    im0 = axes[0].imshow(
        noisy_db, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=vmin, vmax=vmax
    )
    axes[0].set_title('Noisy Mel Spectrogram')
    axes[0].set_ylabel('Mel Bin')

    axes[1].imshow(
        enhanced_db, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=vmin, vmax=vmax
    )
    axes[1].set_title('Enhanced Mel Spectrogram')
    axes[1].set_ylabel('Mel Bin')

    axes[2].imshow(
        clean_db, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=vmin, vmax=vmax
    )
    axes[2].set_title('Clean Mel Spectrogram')
    axes[2].set_ylabel('Mel Bin')
    axes[2].set_xlabel('Time (s)')

    fig.colorbar(im0, ax=axes, label='Power (dB)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close(fig)


def _load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get('args', {})

    model = FullSubNetPlusLite(
        fb_channels=ckpt_args.get('fb_channels', 48),
        fb_hidden=ckpt_args.get('fb_hidden', 64),
        sb_hidden=ckpt_args.get('sb_hidden', 96),
        subband_size=ckpt_args.get('subband_size', 7),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


def _run_chunked_stft_ola(
    model,
    noisy_wave: np.ndarray,
    device: torch.device,
    sr: int,
    window_sec: float,
    hop_sec: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
):
    original_len = len(noisy_wave)
    if int(hop_sec * sr) % hop_length != 0:
        raise ValueError('chunk hop_sec * sr must be divisible by STFT hop_length for frame alignment.')

    win_samples = int(window_sec * sr)
    chunk_hop_samples = int(hop_sec * sr)

    if len(noisy_wave) < win_samples:
        pad = win_samples - len(noisy_wave)
        noisy_wave = np.pad(noisy_wave, (0, pad))

    starts = list(range(0, max(1, len(noisy_wave) - win_samples + 1), chunk_hop_samples))
    last_start = max(0, len(noisy_wave) - win_samples)
    if starts[-1] != last_start:
        starts.append(last_start)

    window = build_hann_window(win_length, device)

    freq_bins = n_fft // 2 + 1
    chunk_frames = 1 + (win_samples - n_fft) // hop_length

    max_frame_end = 0
    acc = None
    wacc = None

    with torch.no_grad():
        for s in starts:
            seg = noisy_wave[s:s + win_samples]
            seg_t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)

            noisy_spec = waveform_to_stft(seg_t, n_fft, hop_length, win_length, window)
            feat, _, _, _ = stft_features(noisy_spec)
            crm = model(feat)
            enh_spec = apply_complex_ratio_mask(noisy_spec, crm)

            enh_np = enh_spec.squeeze(0).detach().cpu().numpy()  # (F, T)
            if enh_np.shape[1] != chunk_frames:
                chunk_frames = enh_np.shape[1]

            frame0 = s // hop_length
            frame1 = frame0 + enh_np.shape[1]
            max_frame_end = max(max_frame_end, frame1)

            if acc is None:
                acc = np.zeros((freq_bins, frame1), dtype=np.complex64)
                wacc = np.zeros(frame1, dtype=np.float32)
            elif frame1 > acc.shape[1]:
                pad = frame1 - acc.shape[1]
                acc = np.pad(acc, ((0, 0), (0, pad)), mode='constant')
                wacc = np.pad(wacc, (0, pad), mode='constant')

            if enh_np.shape[1] > 2:
                frame_w = np.hanning(enh_np.shape[1]).astype(np.float32)
            else:
                frame_w = np.ones(enh_np.shape[1], dtype=np.float32)

            acc[:, frame0:frame1] += enh_np * frame_w[None, :]
            wacc[frame0:frame1] += frame_w

    wacc = np.maximum(wacc, 1e-6)
    full_spec = acc[:, :max_frame_end] / wacc[:max_frame_end][None, :]

    full_spec_t = torch.from_numpy(full_spec).unsqueeze(0).to(device)
    rec_len = n_fft + hop_length * (full_spec.shape[1] - 1)
    rec_wave = stft_to_waveform(
        full_spec_t,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=rec_len,
    ).squeeze(0).detach().cpu().numpy()

    rec_wave = rec_wave[:original_len]
    return rec_wave.astype(np.float32), full_spec


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, 'best_model.pt')

    model, ckpt = _load_model(ckpt_path, device)
    ckpt_args = ckpt.get('args', {})

    n_fft = args.n_fft if args.n_fft is not None else ckpt_args.get('n_fft', N_FFT)
    hop_length = args.hop_length if args.hop_length is not None else ckpt_args.get('hop_length', HOP_LENGTH)
    win_length = args.win_length if args.win_length is not None else ckpt_args.get('win_length', WIN_LENGTH)

    pl_path = args.pl_bin
    chapter_name = args.chapter_name
    if chapter_name is None:
        chapter_name = match_chapter_name(os.path.basename(pl_path).replace('_real.bin', ''))
        if chapter_name is None:
            raise ValueError('Could not infer chapter_name from file name. Pass --chapter_name explicitly.')

    noisy_wave, clean_wave, _ = load_aligned_noisy_clean(
        pl_path,
        chapter_name,
        preprocess_mode=args.preprocess_mode,
        carrier_hz=args.carrier_hz,
    )

    enhanced_wave, enhanced_spec = _run_chunked_stft_ola(
        model=model,
        noisy_wave=noisy_wave,
        device=device,
        sr=AUDIO_SR,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    out_stem = os.path.splitext(os.path.basename(pl_path))[0]
    wav_out = os.path.join(args.out_dir, f'{out_stem}_enhanced.wav')
    mel_out = os.path.join(args.out_dir, f'{out_stem}_enhanced_mel.npy')
    noisy_mel_out = os.path.join(args.out_dir, f'{out_stem}_noisy_mel.npy')
    clean_mel_out = os.path.join(args.out_dir, f'{out_stem}_clean_mel.npy')
    mel_compare_out = os.path.join(args.out_dir, f'{out_stem}_mel_comparison.png')
    noisy_out = os.path.join(args.out_dir, f'{out_stem}_noisy_aligned.wav')
    clean_out = os.path.join(args.out_dir, f'{out_stem}_clean_aligned.wav')

    sf.write(wav_out, enhanced_wave, AUDIO_SR)
    sf.write(noisy_out, noisy_wave, AUDIO_SR)
    sf.write(clean_out, clean_wave, AUDIO_SR)

    mel = mel_spectrogram_from_waveform(
        enhanced_wave,
        sr=AUDIO_SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=args.n_mels,
    )
    noisy_mel = mel_spectrogram_from_waveform(
        noisy_wave,
        sr=AUDIO_SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=args.n_mels,
    )
    clean_mel = mel_spectrogram_from_waveform(
        clean_wave,
        sr=AUDIO_SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=args.n_mels,
    )
    np.save(mel_out, mel)
    np.save(noisy_mel_out, noisy_mel)
    np.save(clean_mel_out, clean_mel)

    _save_mel_comparison_plot(
        noisy_wave=noisy_wave,
        enhanced_wave=enhanced_wave,
        clean_wave=clean_wave,
        sr=AUDIO_SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=args.n_mels,
        out_png=mel_compare_out,
        plot_sec=args.plot_sec,
    )

    print(f'Device          : {device}')
    print(f'Checkpoint      : {ckpt_path}')
    print(f'Input bin       : {pl_path}')
    print(f'Chapter         : {chapter_name}')
    print(f'Enhanced wav    : {wav_out}')
    print(f'Enhanced mel    : {mel_out} | shape={mel.shape}')
    print(f'Noisy mel       : {noisy_mel_out} | shape={noisy_mel.shape}')
    print(f'Clean mel       : {clean_mel_out} | shape={clean_mel.shape}')
    print(f'Mel comparison  : {mel_compare_out}')

    if args.report_metrics:
        window = build_hann_window(win_length, device)
        ref_len = min(len(enhanced_wave), len(clean_wave))
        pred_t = torch.from_numpy(enhanced_wave[:ref_len]).unsqueeze(0).to(device)
        clean_t = torch.from_numpy(clean_wave[:ref_len]).unsqueeze(0).to(device)

        pred_spec = waveform_to_stft(pred_t, n_fft, hop_length, win_length, window)
        clean_spec = waveform_to_stft(clean_t, n_fft, hop_length, win_length, window)

        mrstft_metric = MultiResolutionSTFTLoss()
        rec_metrics = reconstruction_metrics(
            pred_complex=pred_spec,
            target_complex=clean_spec,
            pred_wave=pred_t,
            target_wave=clean_t,
            mrstft_metric=mrstft_metric,
        )

        mel_mse, mel_corr = mel_mse_and_corr(
            enhanced_wave[:ref_len],
            clean_wave[:ref_len],
            sr=AUDIO_SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=args.n_mels,
        )

        print('\nReconstruction metrics:')
        print(f"  LSD                  : {rec_metrics['lsd']:.6f}")
        print(f"  Spectral convergence : {rec_metrics['spectral_convergence']:.6f}")
        print(f"  Multi-scale STFT     : {rec_metrics['multi_scale_stft']:.6f}")
        print(f'  Mel MSE              : {mel_mse:.6f}')
        print(f'  Mel correlation      : {mel_corr:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FullSubNet+ reconstruction inference on a full powerline file.')

    parser.add_argument('--pl_bin', required=True)
    parser.add_argument('--chapter_name', default=None)

    parser.add_argument('--ckpt', default=os.path.join(DEFAULT_CHECKPOINT_DIR, 'best_model.pt'))
    parser.add_argument('--out_dir', default=DEFAULT_INFER_DIR)

    parser.add_argument('--preprocess_mode', choices=['bandpass', 'iq'], default='bandpass')
    parser.add_argument('--carrier_hz', type=float, default=20433.35)

    parser.add_argument('--window_sec', type=float, default=WINDOW_SEC)
    parser.add_argument('--hop_sec', type=float, default=WINDOW_HOP_SEC)

    parser.add_argument('--n_fft', type=int, default=None)
    parser.add_argument('--hop_length', type=int, default=None)
    parser.add_argument('--win_length', type=int, default=None)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument(
        '--plot_sec', type=float, default=60.0,
        help='Seconds shown in mel comparison PNG (<=0 means full duration).',
    )

    parser.add_argument('--report_metrics', action='store_true')

    main(parser.parse_args())
