# FullSubNetPlus Reconstruction Pipeline

This folder contains a reconstruction-only complex-STFT pipeline with a FullSubNet+-style model.

## What it implements

1. Paired noisy/clean waveform windows (`4s`, overlapping).
2. Complex STFT features (`real`, `imag`, `mag/log-mag`).
3. FullSubNet+-style CRM estimator (fullband context + subband recurrent masking).
4. Composite loss:
   - complex MSE (`real/imag`)
   - log-magnitude L1
   - phase-consistency term
   - multi-resolution STFT loss
5. Inference with overlap-add in STFT chunk domain, then ISTFT to waveform.
6. Strict reconstruction evaluation metrics:
   - LSD
   - spectral convergence
   - multi-scale STFT loss
   - mel MSE / mel correlation

## Strict temporal split (no val leakage)

`prepare_data.py` performs split on each recording timeline and applies a guard region around the split before windowing.
Train windows come from earlier time; validation windows come from later held-out time only.

## Quick start

```bash
cd '<REPO_ROOT>/baselines/FullSubNetPlus'

python prepare_data.py \
  --folders May29_Alice \
  --preprocess_mode bandpass \
  --window_sec 4.0 \
  --hop_sec 1.0 \
  --train_ratio 0.9 \
  --guard_sec 4.0

python train.py \
  --data data/train_data_fullsubnetplus.npz \
  --out_dir checkpoints \
  --epochs 200 \
  --batch_size 12

python evaluate.py \
  --data data/train_data_fullsubnetplus.npz \
  --ckpt checkpoints/best_model.pt \
  --out_json checkpoints/val_metrics.json

python inference.py \
  --pl_bin '<REPO_ROOT>/May29_Alice/Chap_4_real.bin' \
  --chapter_name chapter_04 \
  --ckpt checkpoints/best_model.pt \
  --out_dir inference_output \
  --report_metrics
```

## Files

- `prepare_data.py`: builds strict temporal train/val waveform windows.
- `dataset.py`: dataloaders for waveform pairs.
- `model.py`: FullSubNet+-style CRM estimator.
- `losses.py`: composite reconstruction loss.
- `stft_ops.py`: STFT feature/mask helpers.
- `train.py`: model training.
- `inference.py`: full-file STFT overlap-add reconstruction.
- `metrics.py`: required reconstruction metrics.
- `evaluate.py`: validation evaluation script.
- `run_fsp_prepare.sh`: HPC Slurm submit script for data prep.
- `run_fsp_train.sh`: HPC Slurm submit script for training.
- `run_fsp_eval.sh`: HPC Slurm submit script for validation metrics.
- `run_fsp_inference.sh`: HPC Slurm submit script for full-file inference.
- `run_fsp_full.sh`: HPC Slurm submit script for prep -> train -> eval -> inference.
