"""
prepare_data.py
---------------
Module 6 - HybridCTCAttn data preparation.

CTC data pipeline with bandpass 50-4000 Hz.
Uses UNet predicted mels + text labels from Whisper word timestamps.

Temporal split: 90/10 per chapter, no overlap.

Output .npz:
    predicted_train, noisy_train, clean_train, text_train, text_len_train
    predicted_val, noisy_val, clean_val, text_val, text_len_val
"""

import os, sys, glob, argparse, time, json
import numpy as np
import torch
import whisper
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

# ── paths ──────────────────────────────────────────────────────────────────────
SCRATCH      = '<REPO_ROOT>'
DATA_FOLDERS = {
    'May29_Alice': os.path.join(SCRATCH, 'May29_Alice'),
}
MP3_DIR      = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
_HERE        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT  = os.path.join(_HERE, 'data')

# ── chapter timing ─────────────────────────────────────────────────────────────
CHAPTER_TIMING = {
    'chapter_01': {'start': 1.38, 'mp3_duration': 631.95},
    'chapter_02': {'start': 2.04, 'mp3_duration': 724.14},
    'chapter_04': {'start': 1.95, 'mp3_duration': 1168.38},
    'chapter_05': {'start': 2.17, 'mp3_duration': 794.54},
    'chapter_06': {'start': 2.09, 'mp3_duration': 765.18},
    'chapter_07': {'start': 2.05, 'mp3_duration': 1027.74},
    'chapter_08': {'start': 1.95, 'mp3_duration': 789.39},
    'chapter_10': {'start': 1.63, 'mp3_duration': 1345.72},
    'chapter_11': {'start': 1.61, 'mp3_duration': 601.23},
}

# ── signal params ──────────────────────────────────────────────────────────────
FS       = 200_000
AUDIO_SR = 16_000
BP_LOW   = 50
BP_HIGH  = 4000
N_FFT    = 400
HOP      = 160
N_MELS   = 80

# ── vocabulary ─────────────────────────────────────────────────────────────────
# CTC vocab: 0=blank, 1=space, 2-27=a-z, 28=apostrophe
VOCAB = {' ': 1, "'": 28}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    VOCAB[c] = i + 2
VOCAB_SIZE = 29
BLANK_IDX  = 0

# Attention decoder special tokens
SOS_IDX = 29
EOS_IDX = 30
DEC_VOCAB_SIZE = 31  # 29 + SOS + EOS


def char_to_idx(text):
    """Convert text to list of vocab indices (skip unknown chars)."""
    return [VOCAB[c] for c in text.lower() if c in VOCAB]


# ── signal processing ─────────────────────────────────────────────────────────

def bandpass_filter(x, fs, lo, hi, order=5):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


# ── log-mel spectrogram ───────────────────────────────────────────────────────

def log_mel_spectrogram(x, sr=AUDIO_SR):
    """Compute log-mel spectrogram using librosa mel filterbank."""
    mel_spec = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=0, fmax=sr // 2,
    )
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
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


# ── Whisper transcription with word timestamps ────────────────────────────────

def get_word_timestamps(audio_16k, whisper_model):
    """Use Whisper to get word-level timestamps from a 16 kHz waveform."""
    audio_16k = np.asarray(audio_16k, dtype=np.float32)
    result = whisper_model.transcribe(
        audio_16k, language='en', word_timestamps=True)

    words = []
    for seg in result['segments']:
        if 'words' in seg:
            for w in seg['words']:
                words.append({
                    'word': w['word'].strip().lower(),
                    'start': w['start'],
                    'end': w['end'],
                })
    return words


def text_for_window(words, win_start_sec, win_end_sec):
    """Get text for a time window from word timestamps."""
    text_parts = []
    for w in words:
        if w['end'] > win_start_sec and w['start'] < win_end_sec:
            text_parts.append(w['word'])
    return ' '.join(text_parts).strip()


# ── UNet prediction ──────────────────────────────────────────────────────────

def load_unet(ckpt_path, device):
    """Load the PowerlineUNet from the UNet module."""
    unet_dir = os.path.join(SCRATCH, 'UNet')
    if unet_dir not in sys.path:
        sys.path.insert(0, unet_dir)
    from model_bandpass import PowerlineUNet

    unet = PowerlineUNet(base_ch=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state' in ckpt:
        unet.load_state_dict(ckpt['model_state'])
    elif 'gen_state' in ckpt:
        unet.load_state_dict(ckpt['gen_state'])
    else:
        unet.load_state_dict(ckpt)
    unet.eval()
    print(f"Loaded UNet from {ckpt_path}")
    return unet


@torch.no_grad()
def predict_mels_unet(noisy_mels, unet, device, batch_size=32):
    """Run UNet on all noisy mels to produce predicted mels."""
    predicted = []
    n = len(noisy_mels)
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(noisy_mels[i:i+batch_size]).float().to(device)
        pred = unet(batch)
        predicted.append(pred.cpu().numpy())
    return np.concatenate(predicted, axis=0)


# ── main processing ──────────────────────────────────────────────────────────

def process_chapter(pl_path, chapter_name, win_sec, hop_sec,
                    whisper_model, max_text_len, verbose=True):
    timing = CHAPTER_TIMING[chapter_name]
    offset_sec   = timing['start']
    mp3_duration = timing['mp3_duration']

    if verbose:
        print(f"  Loading {os.path.basename(pl_path)} ...")

    # Load and trim PL signal
    pl_raw = np.fromfile(pl_path, dtype=np.float32).astype(np.float64)
    start_samp = int(offset_sec * FS)
    chap_samps = int(mp3_duration * FS)
    pl_raw = pl_raw[start_samp:start_samp + chap_samps]

    if verbose:
        print(f"    Chapter: {chapter_name} | {len(pl_raw)/FS:.1f}s")

    # Bandpass + downsample
    pl_bp    = bandpass_filter(pl_raw, FS, BP_LOW, BP_HIGH)
    pl_audio = resample_to(pl_bp, FS, AUDIO_SR)

    # Load MP3
    ch_num   = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f"Alice_In_Wonderland_ch_{ch_num}.mp3")
    if not os.path.exists(mp3_path):
        print(f"    WARNING: MP3 not found: {mp3_path}")
        return None, None, None, None

    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

    # Keep a copy for Whisper timestamping
    mp3_for_timestamps = mp3_audio.copy()

    # Align lengths
    L = min(len(pl_audio), len(mp3_audio))
    pl_audio  = pl_audio[:L]
    mp3_audio = mp3_audio[:L]

    # RMS normalize
    pl_audio  = rms_norm(pl_audio)
    mp3_audio = rms_norm(mp3_audio)

    # Get word timestamps from MP3
    if verbose:
        print(f"    Running Whisper for word timestamps ...")
    words = get_word_timestamps(mp3_for_timestamps, whisper_model)
    if verbose:
        print(f"    Got {len(words)} words")

    # Window and compute spectrograms + text labels
    win_samp = int(win_sec * AUDIO_SR)
    hop_samp = int(hop_sec * AUDIO_SR)
    starts   = range(0, L - win_samp, hop_samp)

    noisy_list, clean_list, text_list, text_len_list = [], [], [], []

    for s in starts:
        e = s + win_samp
        win_start_sec = s / AUDIO_SR
        win_end_sec   = e / AUDIO_SR

        # Get text for this window
        text = text_for_window(words, win_start_sec, win_end_sec)
        encoded = char_to_idx(text)

        if len(encoded) == 0:
            continue

        # Pad/truncate text to max_text_len
        text_len = min(len(encoded), max_text_len)
        padded   = encoded[:max_text_len]
        padded  += [BLANK_IDX] * (max_text_len - len(padded))

        # Compute mel spectrograms
        pl_mel  = log_mel_spectrogram(pl_audio[s:e])
        mp3_mel = log_mel_spectrogram(mp3_audio[s:e])

        noisy_list.append(pl_mel[np.newaxis])
        clean_list.append(mp3_mel[np.newaxis])
        text_list.append(np.array(padded, dtype=np.int32))
        text_len_list.append(text_len)

    if verbose:
        print(f"    Windows: {len(noisy_list)} (skipped silent)")
        if noisy_list:
            print(f"    Spec shape: (1, {N_MELS}, {noisy_list[0].shape[-1]})")

    if not noisy_list:
        return None, None, None, None

    return (np.stack(noisy_list).astype(np.float32),
            np.stack(clean_list).astype(np.float32),
            np.stack(text_list),
            np.array(text_len_list, dtype=np.int32))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nModule 6 - HybridCTCAttn Data Preparation")
    print(f"Output: {args.out_dir}")
    print(f"Window: {args.win_sec}s  Hop: {args.hop_sec}s")
    print(f"Max text len: {args.max_text_len}")
    print(f"Bandpass: {BP_LOW}-{BP_HIGH} Hz")
    print(f"Split: temporal (90/10 per chapter)\n")

    # Device for UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Whisper for word timestamps
    print("Loading Whisper model for timestamps ...")
    whisper_model = whisper.load_model(args.whisper_model)
    print(f"Whisper model: {args.whisper_model}\n")

    # Collect per-chapter data with temporal split
    all_train_noisy, all_train_clean = [], []
    all_train_text,  all_train_tlen  = [], []
    all_val_noisy,   all_val_clean   = [], []
    all_val_text,    all_val_tlen    = [], []

    for folder_name in args.folders:
        if folder_name not in DATA_FOLDERS:
            print(f"WARNING: unknown folder '{folder_name}'")
            continue

        data_dir = DATA_FOLDERS[folder_name]
        pl_files = sorted(glob.glob(os.path.join(data_dir, '*_real.bin')))
        print(f"Folder: {folder_name} ({len(pl_files)} files)")

        for pl_path in pl_files:
            file_id = os.path.basename(pl_path).replace('_real.bin', '')
            chapter_name = match_chapter_name(file_id)
            if chapter_name is None:
                print(f"  SKIP {file_id}: no chapter timing")
                continue

            t0 = time.time()
            noisy, clean, text, tlen = process_chapter(
                pl_path, chapter_name, args.win_sec, args.hop_sec,
                whisper_model, args.max_text_len)

            if noisy is None:
                continue

            # Temporal split: first 90% train, last 10% val
            n_win = len(noisy)
            split_idx = int(0.9 * n_win)
            all_train_noisy.append(noisy[:split_idx])
            all_train_clean.append(clean[:split_idx])
            all_train_text.append(text[:split_idx])
            all_train_tlen.append(tlen[:split_idx])
            all_val_noisy.append(noisy[split_idx:])
            all_val_clean.append(clean[split_idx:])
            all_val_text.append(text[split_idx:])
            all_val_tlen.append(tlen[split_idx:])
            print(f"    Done in {time.time()-t0:.1f}s "
                  f"(train: {split_idx}, val: {n_win - split_idx})")

    if not all_train_noisy:
        print("ERROR: no data collected.")
        sys.exit(1)

    noisy_train = np.concatenate(all_train_noisy, axis=0)
    clean_train = np.concatenate(all_train_clean, axis=0)
    text_train  = np.concatenate(all_train_text,  axis=0)
    tlen_train  = np.concatenate(all_train_tlen,  axis=0)
    noisy_val   = np.concatenate(all_val_noisy,   axis=0)
    clean_val   = np.concatenate(all_val_clean,   axis=0)
    text_val    = np.concatenate(all_val_text,    axis=0)
    tlen_val    = np.concatenate(all_val_tlen,    axis=0)

    # Shuffle train set (val stays in order for temporal consistency)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(noisy_train))
    noisy_train = noisy_train[perm]
    clean_train = clean_train[perm]
    text_train  = text_train[perm]
    tlen_train  = tlen_train[perm]

    print(f"\nTotal: train={len(noisy_train)}, val={len(noisy_val)}")
    print(f"Split: temporal (first 90% of each chapter = train, last 10% = val)")
    print(f"Spec shape: {noisy_train.shape[1:]}")
    print(f"Text shape: {text_train.shape}")

    # --- UNet prediction ---
    print(f"\nLoading UNet for mel prediction ...")
    unet = load_unet(args.unet_ckpt, device)

    print(f"Running UNet on train set ({len(noisy_train)} samples) ...")
    predicted_train = predict_mels_unet(noisy_train, unet, device,
                                         batch_size=args.unet_batch)
    print(f"Running UNet on val set ({len(noisy_val)} samples) ...")
    predicted_val = predict_mels_unet(noisy_val, unet, device,
                                       batch_size=args.unet_batch)

    print(f"Predicted mel shapes: train={predicted_train.shape}, val={predicted_val.shape}")

    # --- Save ---
    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        predicted_train=predicted_train,
        noisy_train=noisy_train,
        clean_train=clean_train,
        text_train=text_train,
        text_len_train=tlen_train,
        predicted_val=predicted_val,
        noisy_val=noisy_val,
        clean_val=clean_val,
        text_val=text_val,
        text_len_val=tlen_val,
    )
    print(f"\nSaved -> {out_path}")

    # Save vocab for reference
    vocab_path = os.path.join(args.out_dir, 'vocab.json')
    idx_to_char = {0: '<blank>'}
    for c, i in VOCAB.items():
        idx_to_char[i] = c
    vocab_info = {
        'char_to_idx': VOCAB,
        'idx_to_char': idx_to_char,
        'vocab_size': VOCAB_SIZE,
        'sos_idx': SOS_IDX,
        'eos_idx': EOS_IDX,
        'dec_vocab_size': DEC_VOCAB_SIZE,
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_info, f, indent=2)
    print(f"Vocab -> {vocab_path}")
    print("Done.")


if __name__ == '__main__':
    default_unet = os.path.join(SCRATCH, 'UNet', 'checkpoints_bp', 'best_model.pt')
    parser = argparse.ArgumentParser(
        description='Module 6 HybridCTCAttn data preparation')
    parser.add_argument('--folders', nargs='+', default=['May29_Alice'])
    parser.add_argument('--win_sec', type=float, default=5.0)
    parser.add_argument('--hop_sec', type=float, default=1.0)
    parser.add_argument('--max_text_len', type=int, default=200)
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--out_name', default='hybrid_data.npz')
    parser.add_argument('--whisper_model', default='base.en')
    parser.add_argument('--unet_ckpt', default=default_unet)
    parser.add_argument('--unet_batch', type=int, default=32)
    args = parser.parse_args()
    main(args)
