"""
prepare_data.py
---------------
Module 8: Prepare powerline audio data for Whisper fine-tuning.

Pipeline per chapter:
  1. Read *_real.bin (real IQ component, sampled at 200 kHz)
  2. Bandpass filter 50-4000 Hz
  3. Downsample to 16 kHz (Whisper's expected sample rate)
  4. Window into 10-sec clips with 2-sec hop
  5. Get word-level timestamps from MP3 via Whisper teacher model
  6. Temporal split: first 90% = train, last 10% = val

Output: data/powerline_whisper_data.npz
    audio_train  (N, 160000) float32  -- 10s @ 16kHz
    audio_val    (M, 160000) float32
    text_train   (N,) object          -- reference transcript strings
    text_val     (M,) object
    sr           scalar               -- always 16000

This mirrors the prepare_data.py in Module3 but outputs raw waveforms
instead of mel spectrograms, so Whisper can process them directly.
"""

import os, sys, glob, argparse, time
from math import gcd

import numpy as np
import whisper
import librosa
from scipy.signal import butter, sosfiltfilt, resample_poly

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH    = '<REPO_ROOT>'
MP3_DIR    = os.path.join(SCRATCH, 'Alice_In_Wonderland_mp3')
DATA_ROOTS = {'May29_Alice': os.path.join(SCRATCH, 'May29_Alice')}

# ---------------------------------------------------------------------------
# Chapter timing (start offset into the capture, mp3 duration)
# ---------------------------------------------------------------------------
CHAPTER_TIMING = {
    'chapter_01': {'start': 1.38,  'mp3_duration': 631.95},
    'chapter_02': {'start': 2.04,  'mp3_duration': 724.14},
    'chapter_04': {'start': 1.95,  'mp3_duration': 1168.38},
    'chapter_05': {'start': 2.17,  'mp3_duration': 794.54},
    'chapter_06': {'start': 2.09,  'mp3_duration': 765.18},
    'chapter_07': {'start': 2.05,  'mp3_duration': 1027.74},
    'chapter_08': {'start': 1.95,  'mp3_duration': 789.39},
    'chapter_10': {'start': 1.63,  'mp3_duration': 1345.72},
    'chapter_11': {'start': 1.61,  'mp3_duration': 601.23},
}

# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------
FS       = 200_000   # capture sample rate
AUDIO_SR = 16_000    # Whisper sample rate
BP_LOW   = 50
BP_HIGH  = 4_000
TRAIN_RATIO = 0.9


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def bandpass_filter(x, fs, lo, hi, order=5):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def resample_to(x, fs_in, fs_out):
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def rms_norm(x, target=0.1):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x * target / rms).astype(np.float32)


# ---------------------------------------------------------------------------
# Chapter / file helpers
# ---------------------------------------------------------------------------
def match_chapter_name(file_id):
    fid = file_id.lower()
    for ch in CHAPTER_TIMING:
        num = ch.split('_')[1]
        if (f'chapter_{num}' in fid or f'chap_{num}' in fid
                or f'chap_{int(num)}' in fid or f'chap{int(num)}' in fid):
            return ch
    return None


def get_mp3_path(chapter_name):
    ch_num = chapter_name.split('_')[1]
    return os.path.join(MP3_DIR, f'Alice_In_Wonderland_ch_{ch_num}.mp3')


# ---------------------------------------------------------------------------
# Whisper-based word-timestamp alignment
# ---------------------------------------------------------------------------
def get_word_timestamps(audio_16k, teacher):
    result = teacher.transcribe(
        np.asarray(audio_16k, dtype=np.float32),
        language='en', word_timestamps=True,
    )
    words = []
    for seg in result['segments']:
        for w in seg.get('words', []):
            words.append({
                'word':  w['word'].strip().lower(),
                'start': float(w['start']),
                'end':   float(w['end']),
            })
    return words


def text_for_window(words, win_start, win_end):
    parts = [w['word'] for w in words
             if w['end'] > win_start and w['start'] < win_end]
    return ' '.join(parts).strip()


# ---------------------------------------------------------------------------
# Process one chapter
# ---------------------------------------------------------------------------
def process_chapter(pl_path, chapter_name, win_sec, hop_sec,
                    teacher, max_text_len, verbose=True):
    timing      = CHAPTER_TIMING[chapter_name]
    offset_sec  = timing['start']
    mp3_dur     = timing['mp3_duration']

    if verbose:
        print(f'  Loading {os.path.basename(pl_path)} ...')

    # Load and crop to chapter length
    pl_raw   = np.fromfile(pl_path, dtype=np.float32).astype(np.float64)
    s0       = int(offset_sec * FS)
    n_chap   = int(mp3_dur * FS)
    pl_raw   = pl_raw[s0: s0 + n_chap]

    if verbose:
        print(f'    {chapter_name} | {len(pl_raw)/FS:.1f}s at {FS}Hz')

    # Bandpass + downsample
    pl_bp    = bandpass_filter(pl_raw, FS, BP_LOW, BP_HIGH)
    pl_audio = resample_to(pl_bp, FS, AUDIO_SR)       # (N,) at 16kHz

    # Load MP3 (clean reference)
    mp3_path = get_mp3_path(chapter_name)
    if not os.path.exists(mp3_path):
        print(f'    WARNING: MP3 not found: {mp3_path}')
        return None
    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    if mp3_sr != AUDIO_SR:
        mp3_audio = resample_to(mp3_audio, mp3_sr, AUDIO_SR)

    # Get word timestamps from the clean MP3
    if verbose:
        print('    Running Whisper teacher on MP3 ...')
    words = get_word_timestamps(mp3_audio, teacher)
    if verbose:
        print(f'    Got {len(words)} word timestamps')

    # Build train/val split point
    duration_sec  = len(pl_audio) / AUDIO_SR
    split_sec     = duration_sec * TRAIN_RATIO
    win_samples   = int(win_sec * AUDIO_SR)
    hop_samples   = int(hop_sec * AUDIO_SR)

    train_clips, train_texts = [], []
    val_clips,   val_texts   = [], []

    n_windows = (len(pl_audio) - win_samples) // hop_samples + 1
    for i in range(n_windows):
        start_samp = i * hop_samples
        end_samp   = start_samp + win_samples
        if end_samp > len(pl_audio):
            break

        win_start_sec = start_samp / AUDIO_SR
        win_end_sec   = end_samp   / AUDIO_SR

        text = text_for_window(words, win_start_sec, win_end_sec)
        if not text:
            continue

        # Truncate text to max_text_len characters
        if len(text) > max_text_len:
            text = text[:max_text_len]

        clip = pl_audio[start_samp:end_samp].copy()
        clip = rms_norm(clip)   # normalise amplitude

        if win_start_sec < split_sec:
            train_clips.append(clip)
            train_texts.append(text)
        else:
            val_clips.append(clip)
            val_texts.append(text)

    if verbose:
        print(f'    train={len(train_clips)}  val={len(val_clips)}')

    return {
        'train_clips': train_clips,
        'train_texts': train_texts,
        'val_clips':   val_clips,
        'val_texts':   val_texts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Whisper teacher model
    print(f'Loading Whisper teacher ({args.teacher_model}) ...')
    teacher = whisper.load_model(args.teacher_model)
    print('Teacher loaded.')

    all_train_clips, all_train_texts = [], []
    all_val_clips,   all_val_texts   = [], []

    for folder_name in args.folders:
        data_dir = os.path.join(SCRATCH, folder_name)
        if not os.path.isdir(data_dir):
            print(f'WARNING: folder not found: {data_dir}')
            continue

        pl_files = sorted(glob.glob(os.path.join(data_dir, '*_real.bin')))
        print(f'\nFolder {folder_name}: {len(pl_files)} real.bin files')

        for pl_path in pl_files:
            file_id      = os.path.basename(pl_path).replace('_real.bin', '')
            chapter_name = match_chapter_name(file_id)
            if chapter_name is None:
                print(f'  SKIP {file_id}: cannot match chapter')
                continue
            print(f'\n  -> {file_id}  ({chapter_name})')
            t0 = time.time()
            result = process_chapter(
                pl_path, chapter_name,
                win_sec=args.win_sec, hop_sec=args.hop_sec,
                teacher=teacher, max_text_len=args.max_text_len,
            )
            if result is None:
                continue
            all_train_clips.extend(result['train_clips'])
            all_train_texts.extend(result['train_texts'])
            all_val_clips.extend(result['val_clips'])
            all_val_texts.extend(result['val_texts'])
            print(f'    Processed in {time.time()-t0:.1f}s')

    print(f'\nTotal: train={len(all_train_clips)}  val={len(all_val_clips)}')
    if not all_train_clips:
        raise RuntimeError('No training clips generated.')

    # Stack into arrays
    audio_train = np.stack(all_train_clips).astype(np.float32)
    audio_val   = np.stack(all_val_clips).astype(np.float32)

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        audio_train = audio_train,
        audio_val   = audio_val,
        text_train  = np.array(all_train_texts, dtype=object),
        text_val    = np.array(all_val_texts,   dtype=object),
        sr          = np.array(AUDIO_SR),
    )
    print(f'\nSaved: {out_path}')
    print(f'  audio_train: {audio_train.shape}')
    print(f'  audio_val  : {audio_val.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders',       nargs='+', default=['May29_Alice'])
    parser.add_argument('--win_sec',       type=float, default=10.0,
                        help='Window length in seconds (max 30 for Whisper)')
    parser.add_argument('--hop_sec',       type=float, default=2.0)
    parser.add_argument('--out_dir',       default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'data'))
    parser.add_argument('--out_name',      default='powerline_whisper_data.npz')
    parser.add_argument('--max_text_len',  type=int, default=448)
    parser.add_argument('--teacher_model', default='base',
                        help='openai-whisper model for MP3 alignment')
    args = parser.parse_args()
    main(args)
