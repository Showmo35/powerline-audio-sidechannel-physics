"""
prepare_data.py
---------------
Build paired noisy/clean waveform windows for FullSubNet+-style complex STFT
training with a strict temporal train/val split.

Key rule:
- Validation is temporally held out for each recording.
- A guard region around the split prevents train/val leakage with overlap.

Outputs .npz with:
- noisy_train: (N_train, win_samples)
- clean_train: (N_train, win_samples)
- noisy_val  : (N_val, win_samples)
- clean_val  : (N_val, win_samples)
"""

import argparse
import glob
import json
import os

import numpy as np

from audio_utils import (
    load_aligned_noisy_clean,
    make_window_starts,
    match_chapter_name,
    temporal_partition_indices,
)
from config import (
    AUDIO_SR,
    DATA_FOLDERS,
    DEFAULT_DATA_DIR,
    DEFAULT_DATA_NPZ,
    SPLIT_GUARD_SEC,
    TRAIN_RATIO,
    WINDOW_HOP_SEC,
    WINDOW_SEC,
)


def _slice_windows(noisy: np.ndarray, clean: np.ndarray, starts: np.ndarray, win: int):
    if len(starts) == 0:
        return None, None
    noisy_list = [noisy[s:s + win] for s in starts]
    clean_list = [clean[s:s + win] for s in starts]
    return np.stack(noisy_list).astype(np.float32), np.stack(clean_list).astype(np.float32)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    win_samples = int(args.window_sec * AUDIO_SR)
    hop_samples = int(args.hop_sec * AUDIO_SR)

    all_train_noisy = []
    all_train_clean = []
    all_val_noisy = []
    all_val_clean = []
    records = []

    print(f'Output directory : {args.out_dir}')
    print(f'Output file      : {args.out_name}')
    print(f'Preprocess mode  : {args.preprocess_mode}')
    print(f'Window/Hop       : {args.window_sec}s / {args.hop_sec}s')
    print(f'Temporal split   : train_ratio={args.train_ratio:.2f}, guard={args.guard_sec:.2f}s')
    print(f'Folders          : {args.folders}\n')

    for folder_name in args.folders:
        if folder_name not in DATA_FOLDERS:
            print(f'SKIP unknown folder: {folder_name}')
            continue

        data_dir = DATA_FOLDERS[folder_name]
        pl_files = sorted(glob.glob(os.path.join(data_dir, '*_real.bin')))
        print(f'Folder {folder_name}: {len(pl_files)} files')

        for pl_path in pl_files:
            file_id = os.path.basename(pl_path).replace('_real.bin', '')
            chapter_name = match_chapter_name(file_id)

            if chapter_name is None:
                print(f'  SKIP {file_id}: no chapter timing match')
                continue

            try:
                noisy, clean, mp3_path = load_aligned_noisy_clean(
                    pl_path,
                    chapter_name,
                    preprocess_mode=args.preprocess_mode,
                    carrier_hz=args.carrier_hz,
                )
            except FileNotFoundError as exc:
                print(f'  SKIP {file_id}: {exc}')
                continue

            common = min(len(noisy), len(clean))
            noisy = noisy[:common]
            clean = clean[:common]

            train_end, val_start = temporal_partition_indices(
                n_samples=common,
                sr=AUDIO_SR,
                train_ratio=args.train_ratio,
                guard_sec=args.guard_sec,
                window_sec=args.window_sec,
            )
            if val_start < train_end:
                print(f'  SKIP {file_id}: split overlap after guard (train_end={train_end}, val_start={val_start})')
                continue

            train_starts = make_window_starts(0, train_end, win_samples, hop_samples)
            val_starts = make_window_starts(val_start, common, win_samples, hop_samples)

            train_noisy, train_clean = _slice_windows(noisy, clean, train_starts, win_samples)
            val_noisy, val_clean = _slice_windows(noisy, clean, val_starts, win_samples)

            if train_noisy is None or val_noisy is None:
                print(f'  SKIP {file_id}: insufficient data after split/guard')
                continue

            all_train_noisy.append(train_noisy)
            all_train_clean.append(train_clean)
            all_val_noisy.append(val_noisy)
            all_val_clean.append(val_clean)

            record = {
                'file_id': file_id,
                'chapter_name': chapter_name,
                'mp3_path': mp3_path,
                'num_samples': int(common),
                'duration_sec': round(common / AUDIO_SR, 3),
                'train_end_sample': int(train_end),
                'val_start_sample': int(val_start),
                'train_windows': int(len(train_starts)),
                'val_windows': int(len(val_starts)),
            }
            records.append(record)
            print(
                f"  {file_id}: dur={record['duration_sec']}s | "
                f"train={record['train_windows']} win | val={record['val_windows']} win"
            )

        print()

    if not all_train_noisy or not all_val_noisy:
        raise RuntimeError('No train/val windows collected. Check paths and chapter mappings.')

    noisy_train = np.concatenate(all_train_noisy, axis=0)
    clean_train = np.concatenate(all_train_clean, axis=0)
    noisy_val = np.concatenate(all_val_noisy, axis=0)
    clean_val = np.concatenate(all_val_clean, axis=0)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(noisy_train))
    noisy_train = noisy_train[perm]
    clean_train = clean_train[perm]

    out_path = os.path.join(args.out_dir, args.out_name)
    np.savez_compressed(
        out_path,
        noisy_train=noisy_train,
        clean_train=clean_train,
        noisy_val=noisy_val,
        clean_val=clean_val,
        sample_rate=np.array([AUDIO_SR], dtype=np.int32),
        window_sec=np.array([args.window_sec], dtype=np.float32),
        hop_sec=np.array([args.hop_sec], dtype=np.float32),
        train_ratio=np.array([args.train_ratio], dtype=np.float32),
        guard_sec=np.array([args.guard_sec], dtype=np.float32),
        preprocess_mode=np.array([args.preprocess_mode]),
    )

    meta_path = out_path.replace('.npz', '_records.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    print('Saved dataset:')
    print(f'  {out_path}')
    print(f'  {meta_path}')
    print(f'Train windows: {len(noisy_train):,}')
    print(f'Val windows  : {len(noisy_val):,}')
    print(f'Window shape : ({win_samples},)')
    print('Temporal split is strict: validation windows are from held-out later time regions only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare FullSubNet+ waveform windows with strict temporal split.')
    parser.add_argument('--folders', nargs='+', default=['May29_Alice'])
    parser.add_argument('--preprocess_mode', choices=['bandpass', 'iq'], default='bandpass')
    parser.add_argument('--carrier_hz', type=float, default=20433.35)

    parser.add_argument('--window_sec', type=float, default=WINDOW_SEC)
    parser.add_argument('--hop_sec', type=float, default=WINDOW_HOP_SEC)

    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO)
    parser.add_argument('--guard_sec', type=float, default=SPLIT_GUARD_SEC)

    parser.add_argument('--out_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--out_name', default=os.path.basename(DEFAULT_DATA_NPZ))
    parser.add_argument('--seed', type=int, default=42)

    main(parser.parse_args())
