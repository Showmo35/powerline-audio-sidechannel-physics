# Cascade: Pretrained UNet -> CTC Encoder

Combines the pretrained bandpass U-Net (spectrogram denoiser) with a CTC
encoder (text decoder) in a two-phase training pipeline.

**Hypothesis**: The UNet's spectrogram denoising (71% MSE reduction, 0.79
correlation) should give the CTC encoder cleaner input than raw bandpass
spectrograms, improving CER over the E2E baseline.

## Architecture

```
Noisy Bandpass Mel (B, 1, 80, T)
        |
  [Pretrained BandpassUNet]  -- from UNet/checkpoints_bp/best_model.pt
        |
  Denoised Mel (B, 1, 80, T)
        |
  [CTCEncoder: Conv2d x2 (stride 2) + BiGRU]
        |
  CTC Logits (T/4, B, 29)
        |
  [Greedy Decode]
        |
  Predicted Text
```

## Two-Phase Training

- **Phase 1** (100 epochs): UNet frozen, CTC encoder trains alone on
  denoised spectrograms. Loss = CTC only.
- **Phase 2** (60 epochs): UNet unfrozen with 10x lower LR, both train
  end-to-end. Loss = CTC + 0.5 * (L1 + MSTFT) reconstruction.

## Data Split

Both UNet and E2E data use **temporal splits**: first 90% of each chapter
is training, last 10% is validation. This prevents data leakage between
the separately-trained UNet and the CTC encoder.

- UNet data: `UNet/data/train_data_bp.npz` (1-sec windows, no text)
- E2E data: `E2E/data/e2e_data.npz` (5-sec windows with text labels)

## Run Full Pipeline (recommended)

This re-prepares both datasets with consistent temporal splits, retrains
the UNet, retrains the E2E CTC baseline, trains the cascade, and runs
inference on both for comparison.

```bash
cd "<REPO_ROOT>/Cascade"
./run_full_pipeline.sh
```

**Estimated time**: ~21 hours compute (critical path), plus SLURM queue wait.

Job dependency graph:

```
prep_unet (30m) ──→ train_unet (10h) ──┐
                                        ├──→ train_cascade (10h) ──→ inference (30m)
prep_e2e  (3h)  ──→ train_e2e  (10h) ──┘
```

Monitor with: `squeue -u $USER`

## Scripts

| Script | Purpose |
|--------|---------|
| `run_full_pipeline.sh` | Master launcher, submits all jobs with dependencies |
| `run_prep_unet.sh` | UNet data prep (temporal split) |
| `run_prep_e2e.sh` | E2E data prep (temporal split) |
| `run_train_unet.sh` | Retrain UNet |
| `run_train_e2e_baseline.sh` | Retrain E2E CTC baseline |
| `run_cascade_train.sh` | Train cascade model |
| `run_inference_all.sh` | Run both models, print comparison |

## Local Run (manual)

```bash
python train_cascade.py \
  --data ../E2E/data/e2e_data.npz \
  --unet_ckpt ../UNet/checkpoints_bp/best_model.pt \
  --out checkpoints \
  --phase1_epochs 100 \
  --phase2_epochs 60

python inference_cascade.py \
  --data ../E2E/data/e2e_data.npz \
  --ckpt checkpoints/best_model.pt \
  --out inference_output \
  --n_samples 5
```

## Comparison Baseline

All results below will use the same temporal split for fair comparison.

| Model | Avg CER | Notes |
|-------|---------|-------|
| E2E CTC-only | TBD | UNet from scratch + CTC (retrained) |
| **Cascade** | **TBD** | Pretrained UNet + CTC (this folder) |
