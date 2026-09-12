"""
prepare_data.py
---------------
Prepare training data for CTC transcription with Conformer encoder.
Module 5 - Conformer: identical pipeline to Module 1/2/3.

Pipeline per chapter:
  1. Bandpass filter raw powerline signal (50-4000 Hz)
  2. Downsample to 16 kHz
  3. Window at 5-sec / 1-sec hop
  4. Compute noisy log-mel spectrograms
  5. Run pretrained UNet on each noisy mel -> predicted mel
  6. Get word-level timestamps from MP3 via Whisper -> text labels
  7. Temporal split: first 90% of each chapter = train, last 10% = val

Output .npz:
    predicted_train, predicted_val  -- UNet output mels
    noisy_train, noisy_val          -- raw bandpass mels (for reference)
    clean_train, clean_val          -- MP3 ground truth mels
    text_train, text_val            -- encoded text labels
    text_len_train, text_len_val    -- true text lengths
"""

import os, sys, glob, argparse, time
import numpy as np
import torch
import whisper
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

# Add parent UNet dir so we can import model_bandpass
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'UNet'))
from model_bandpass import PowerlineUNet

# ── Paths ────────────────────────────────────────────────────────────────────
SCRATCH = '<REPO_ROOT>'
DATA_FOLDERS = {
    'May29_Alice': os.path.join(SCRATCH, 'May29_Alice'),
}
MP3_DIR = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
_HERE   = os.path.dirname(os.path.abspath(__file__))

# ── Chapter timing ───────────────────────────────────────────────────────────
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

# ── Signal params ────────────────────────────────────────────────────────────
FS       = 200_000
AUDIO_SR = 16_000
BP_LOW   = 50
BP_HIGH  = 4000
N_FFT    = 400
HOP      = 160
N_MELS   = 80

# ── Vocabulary ───────────────────────────────────────────────────────────────
VOCAB = {' ': 1, "'": 28}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    VOCAB[c] = i + 2
VOCAB_SIZE = 29
BLANK_IDX  = 0


def char_to_idx(text):
    return [VOCAB[c] for c in text.lower() if c in VOCAB]


# ── Signal processing ────────────────────────────────────────────────────────

def bandpass_filter(x, fs, lo, hi, order=5):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x, target=0.05):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return (x * target / rms).astype(np.float32)


# ── Whisper mel ──────────────────────────────────────────────────────────────

_WHISPER_FB = None

def _get_whisper_filterbank():
    global _WHISPER_FB
    if _WHISPER_FB is None:
        _WHISPER_FB = whisper.audio.mel_filters(
            torch.device('cpu'), N_MELS).numpy()
    return _WHISPER_FB


def log_mel_spectrogram(x, sr=AUDIO_SR):
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


# ── Whisper word timestamps ─────────────────────────────────────────────────

def get_word_timestamps(audio_16k, whisper_model):
    audio_16k = np.asarray(audio_16k, dtype=np.float32)
    result = whisper_model.transcribe(audio_16k, language='en',
                                      word_timestamps=True)
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
    text_parts = []
    for w in words:
        if w['end'] > win_start_sec and w['start'] < win_end_sec:
            text_parts.append(w['word'])
    return ' '.join(text_parts).strip()


# ── UNet prediction ─────────────────────────────────────────────────────────

def run_unet_batch(model, noisy_mels, device, batch_size=32):
    """Run pretrained UNet on a batch of noisy mels. Returns predicted mels."""
    predicted = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(noisy_mels), batch_size):
            batch = torch.from_numpy(noisy_mels[i:i+batch_size]).float().to(device)
            pred = model(batch).cpu().numpy()
            predicted.append(pred)
    return np.concatenate(predicted, axis=0)


# ── Process one chapter ─────────────────────────────────────────────────────

def process_chapter(pl_path, chapter_name, win_sec, hop_sec,
                    whisper_model, max_text_len, verbose=True):
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
        print(f"    Chapter: {chapter_name} | {len(pl_raw)/FS:.1f}s")

    # Bandpass + downsample
    pl_bp    = bandpass_filter(pl_raw, FS, BP_LOW, BP_HIGH)
    pl_audio = resample_to(pl_bp, FS, AUDIO_SR)

    # Load MP3
    ch_num   = chapter_name.split('_')[1]
    mp3_path = os.path.join(MP3_DIR, f"Alice_In_Wonderland_ch_{ch_num}.mp3")
    if not os.path.exists(mp3_path):
        print(f"    WARNING: MP3 not found: {mp3_path}")
        return None

    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

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

    # Window and compute spectrograms + text
    win_samp = int(win_sec * AUDIO_SR)
    hop_samp = int(hop_sec * AUDIO_SR)
    starts   = range(0, L - win_samp, hop_samp)

    noisy_list, clean_list, text_list, text_len_list = [], [], [], []

    for s in starts:
        e = s + win_samp
        win_start_sec = s / AUDIO_SR
        win_end_sec   = e / AUDIO_SR

        text = text_for_window(words, win_start_sec, win_end_sec)
        encoded = char_to_idx(text)

        if len(encoded) == 0:
            continue

        text_len = min(len(encoded), max_text_len)
        padded   = encoded[:max_text_len]
        padded  += [BLANK_IDX] * (max_text_len - len(padded))

        pl_mel  = log_mel_spectrogram(pl_audio[s:e])
        mp3_mel = log_mel_spectrogram(mp3_audio[s:e])

        noisy_list.append(pl_mel[np.newaxis])
        clean_list.append(mp3_mel[np.newaxis])
        text_list.append(np.array(padded, dtype=np.int32))
        text_len_list.append(text_len)

    if verbose:
        print(f"    Windows: {len(noisy_list)} (skipped silent)")

    if not noisy_list:
        return None

    return {
        'noisy': np.stack(noisy_list).astype(np.float32),
        'clean': np.stack(clean_list).astype(np.float32),
        'text':  np.stack(text_list),
        'tlen':  np.array(text_len_list, dtype=np.int32),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nModule 5 - Conformer: Data Preparation")
    print(f"Output: {args.out_dir}")
    print(f"Window: {args.win_sec}s  Hop: {args.hop_sec}s")
    print(f"Max text len: {args.max_text_len}")
    print(f"UNet checkpoint: {args.unet_ckpt}")
    print(f"Split: temporal (90/10)\n")

    # Load UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    ckpt = torch.load(args.unet_ckpt, map_location=device, weights_only=False)
    unet = PowerlineUNet(base_ch=ckpt.get('args', {}).get('base_ch', 64)).to(device)
    unet.load_state_dict(ckpt['model_state'])
    unet.eval()
    print(f"UNet loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', 0):.6f})")

    # Load Whisper
    print("Loading Whisper model for timestamps ...")
    whisper_model = whisper.load_model(args.whisper_model)
    print(f"Whisper model: {args.whisper_model}\n")

    all_train, all_val = [], []

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
            result = process_chapter(
                pl_path, chapter_name, args.win_sec, args.hop_sec,
                whisper_model, args.max_text_len)

            if result is None:
                continue

            # Run UNet on all noisy windows
            print(f"    Running UNet on {len(result['noisy'])} windows ...")
            predicted = run_unet_batch(unet, result['noisy'], device,
                                       batch_size=args.unet_batch)
            result['predicted'] = predicted

            # Temporal split: first 90% = train, last 10% = val
            n_win = len(result['noisy'])
            split_idx = int(0.9 * n_win)

            train_chunk = {k: v[:split_idx] for k, v in result.items()}
            val_chunk   = {k: v[split_idx:] for k, v in result.items()}
            all_train.append(train_chunk)
            all_val.append(val_chunk)

            print(f"    Done in {time.time()-t0:.1f}s "
                  f"(train: {split_idx}, val: {n_win - split_idx})")

    if not all_train:
        print("ERROR: no data collected.")
        sys.exit(1)

    # Concatenate all chapters
    def concat_chunks(chunks):
        keys = chunks[0].keys()
        return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}

    train = concat_chunks(all_train)
    val   = concat_chunks(all_val)

    # Shuffle train set
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(train['noisy']))
    train = {k: v[perm] for k, v in train.items()}

    print(f"\nTotal: train={len(train['noisy'])}, val={len(val['noisy'])}")
    print(f"Spec shape: {train['noisy'].shape[1:]}")
    print(f"Split: temporal (first 90% of each chapter = train, last 10% = val)")

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        predicted_train=train['predicted'],
        predicted_val=val['predicted'],
        noisy_train=train['noisy'],
        noisy_val=val['noisy'],
        clean_train=train['clean'],
        clean_val=val['clean'],
        text_train=train['text'],
        text_val=val['text'],
        text_len_train=train['tlen'],
        text_len_val=val['tlen'],
    )
    print(f"\nSaved -> {out_path}")
    print(f"  Train: {len(train['noisy']):,} windows")
    print(f"  Val:   {len(val['noisy']):,} windows")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', nargs='+', default=['May29_Alice'])
    parser.add_argument('--win_sec', type=float, default=5.0)
    parser.add_argument('--hop_sec', type=float, default=1.0)
    parser.add_argument('--max_text_len', type=int, default=200)
    parser.add_argument('--out_dir', default=os.path.join(_HERE, 'data'))
    parser.add_argument('--out_name', default='transcribe_data.npz')
    parser.add_argument('--unet_ckpt',
        default=os.path.join(SCRATCH, 'UNet', 'checkpoints_bp', 'best_model.pt'))
    parser.add_argument('--unet_batch', type=int, default=32)
    parser.add_argument('--whisper_model', default='base.en')
    args = parser.parse_args()
    main(args)
