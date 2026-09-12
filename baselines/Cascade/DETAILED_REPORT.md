# Cascade Project Detailed Report

Generated: 2026-03-31 (America/New_York)  
Project path: `<REPO_ROOT>/baselines/Cascade`

## 1) Executive Summary

This project implements a two-stage speech transcription pipeline:
1. A pretrained bandpass U-Net denoises mel spectrograms.
2. A CTC encoder transcribes denoised spectrograms to text.

Current status based on available artifacts/logs:
- Pipeline execution completed end-to-end on **March 30, 2026**.
- Core artifacts exist: trained checkpoints, training curve image, inference reports, sample mel plots, and audio reconstructions.
- Best cascade checkpoint selected by validation loss came from **Phase 1 (UNet frozen)**, not Phase 2 fine-tuning.
- In the sampled comparison (5 validation examples), cascade underperformed the E2E CTC baseline on CER:
  - E2E baseline avg CER: **0.6567**
  - Cascade avg CER: **0.6935**
  - Delta: **+0.0368 absolute** (**+5.60% relative**, higher is worse)

## 2) Project Snapshot

- File count in `Cascade`: **51** files
- Major directories:
  - `checkpoints/`
  - `inference_output/`
  - `__pycache__/`
- Key source files:
  - `model.py`
  - `train_cascade.py`
  - `inference_cascade.py`
  - `dataset.py`
- Orchestration scripts:
  - `run_full_pipeline.sh`
  - `run_prep_unet.sh`
  - `run_prep_e2e.sh`
  - `run_train_unet.sh`
  - `run_train_e2e_baseline.sh`
  - `run_cascade_train.sh`
  - `run_inference_all.sh`

## 3) Architecture and Training Design

### 3.1 Model Composition

From `model.py`:
- **UNet denoiser** (`BandpassUNet`)
  - 4 downsampling stages (time-axis pooling), residual bottleneck, 4 upsampling stages.
  - Global residual output: `out_conv(d1) + x`.
- **CTC encoder** (`CTCEncoder`)
  - 2 Conv2d downsampling layers (stride 2x2), followed by BiGRU and linear projection.
  - Produces CTC logits `(T', B, V)` where `V=29`.
- Vocabulary:
  - blank index 0, characters include space, apostrophe, and a-z.

### 3.2 Loss and Optimization Strategy

From `train_cascade.py` and `model.py`:
- Total loss in cascade:
  - `total = beta * CTC + alpha * (L1 + MSTFT)`
- Two phases:
  - **Phase 1**: UNet frozen, `alpha=0` (CTC-only).
  - **Phase 2**: UNet unfrozen, reconstruction enabled (`alpha=0.5`) with differential LR.
- Optimizer/schedule:
  - AdamW + CosineAnnealingLR.
  - Gradient clipping at 1.0.
  - Mixed precision enabled on CUDA.
  - Early stopping patience: 25.

## 4) Data Pipeline and Split Integrity

### 4.1 Intended Split Policy

`README.md` and scripts indicate temporal split policy:
- First 90% of each chapter -> train
- Last 10% of each chapter -> validation

This is applied consistently to avoid leakage across UNet and E2E/Cascade training.

### 4.2 Data Preparation Outcomes (March 30, 2026)

From `Prep_UNet_46188283.out`:
- UNet data (`train_data_bp.npz`):
  - Train: **21,219** windows
  - Val: **2,361** windows
  - Spec shape: `(1, 80, 98)`

From `Prep_E2E_46188284.out`:
- E2E data (`e2e_data.npz` used by Cascade):
  - Train: **5,274** windows
  - Val: **590** windows
  - Spec shape: `(1, 80, 498)`
  - Text shape: `(5274, 200)`

## 5) Training and Runtime Results

### 5.1 SLURM Job Timeline

From `.out` logs:
- `Prep_UNet_46188283.out`: 00:03:34
- `Prep_E2E_46188284.out`: 00:09:34
- `Train_UNet_46188285.out`: 03:36:57
- `Train_E2E_CTC_46188286.out`: 00:36:53
- `Cascade_Train_46188287.out`: 01:02:21
- `Infer_All_46188288.out`: 00:00:34

### 5.2 UNet Retraining (Upstream Dependency)

From `Train_UNet_46188285.out`:
- Epochs run: **204** (early stopped)
- Best val loss: **1.15704** at epoch 174
- First epoch val loss: 1.39468
- Improvement first -> best: **17.04%**

### 5.3 E2E CTC Baseline Retraining

From `Train_E2E_CTC_46188286.out`:
- Epochs run: **72** (early stopped)
- Best val loss: **3.96284** (epoch 42)
- Best CTC-CER observed in training log: **0.7066** (epoch 57)

### 5.4 Cascade Training

From `Cascade_Train_46188287.out` and checkpoint metadata:
- Parameter counts:
  - UNet: 18,110,849
  - CTC encoder: 2,586,621
  - Total: 20,697,470
- Phase 1 (frozen UNet):
  - Ran 57 epochs (early stop)
  - Best val loss in phase: **2.78405** (phase-local epoch 32)
  - Best CER in phase: **0.7154** (phase-local epoch 56)
- Phase 2 (fine-tune both):
  - Ran 25 epochs (early stop)
  - Best val loss in phase: **3.38904** (phase-local epoch 9)
  - Best CER in phase: **0.7106** (phase-local epoch 25)

Important checkpoint fact:
- `checkpoints/best_model.pt` metadata:
  - `phase`: **Phase 1 (UNet frozen)**
  - `epoch`: **31** (0-based global epoch index)
  - `val_loss`: **2.784054214889939**
  - `val_cer`: **0.7518455497757485**

Interpretation:
- Phase 2 improved CER relative to best-loss checkpoint but worsened total val loss (which includes reconstruction terms), so model selection by total val loss retained a Phase 1 checkpoint.

## 6) Inference Outputs and Quality

### 6.1 Cascade Inference Summary

From `inference_output/summary.json`:
- Validation set size: 590
- Evaluated samples: 5 (`indices: [0,1,2,3,4]`)
- Avg CER: **0.6934738756**
- Avg WER: **0.9003947368**
- CER std: 0.00753
- WER std: 0.02042

### 6.2 Sample-Level Behavior

From `inference_output/predictions.tsv` and `transcription_compare.txt`:
- Predictions are heavily compressed/degenerate (frequent short tokens like `o`, `to`, `e`).
- Sample CER range is narrow (~0.683 to ~0.701), indicating consistently poor character-level transcription for these first 5 validation examples.

### 6.3 Baseline vs Cascade (Same Inference Job)

From `Infer_All_46188288.out`:
- E2E CTC baseline avg CER: **0.6567042668**
- Cascade avg CER: **0.6934738756**
- Absolute difference: **+0.0367696088** (cascade worse)
- Relative difference: **+5.60%** CER (cascade worse)

## 7) Artifact Inventory

### 7.1 Checkpoints (`checkpoints/`)

- `best_model.pt` (80 MB) - selected by minimum val loss (Phase 1)
- `last_model.pt` (80 MB) - final state after all phases
- `ckpt_epoch_0019.pt`, `ckpt_epoch_0039.pt`, `ckpt_epoch_0076.pt`
- `training_curves.png` (99 KB)

### 7.2 Inference Artifacts (`inference_output/`)

- `predictions.tsv`
- `transcription_compare.txt`
- `summary.json`
- 5 sample triplets each:
  - `sample_XXXX_mel.png`
  - `sample_XXXX_actual.wav`
  - `sample_XXXX_pred.wav`

All expected inference artifacts were present with timestamps around **2026-03-30 18:54 EDT**.

## 8) Warnings and Operational Notes

From `.err` logs:
- `Prep_UNet_46188283.err` / `Prep_E2E_46188284.err` contained repeated mpg123 ID3 comment parsing warnings:
  - `[src/libmpg123/id3.c:process_comment():587] error: No comment text / valid description?`
- `Prep_E2E_46188284.err` also showed Whisper warning on CPU fp16 fallback:
  - `FP16 is not supported on CPU; using FP32 instead`

Impact assessment:
- These did not stop job completion and output generation.
- The Whisper warning suggests part of preprocessing likely ran without GPU acceleration.

## 9) Risks / Gaps Identified

1. **Selection metric mismatch risk**
   - Best checkpoint is selected by total val loss, not CER.
   - In Phase 2, CER improves while total loss worsens, causing potentially suboptimal ASR model selection.

2. **Evaluation sample size risk**
   - Current comparison uses only 5 validation samples; this is insufficient for robust model ranking.

3. **Inference quality concern**
   - Predicted text patterns suggest CTC decoding collapse/limited language fidelity on sampled outputs.

4. **Cross-objective tension**
   - Combining reconstruction and transcription objectives may improve denoising while hurting final token discrimination under current weighting (`alpha=0.5`).

## 10) Recommended Next Actions

1. Re-run inference on a much larger subset (or full 590 validation samples) for stable CER/WER estimates.
2. Save and compare **best-CER checkpoint** in addition to best-loss checkpoint.
3. Evaluate Phase 2 checkpoints directly for ASR metrics even when total loss is higher.
4. Sweep `alpha` (e.g., 0.0, 0.1, 0.25, 0.5) and `unet_lr_factor` to rebalance ASR vs reconstruction.
5. Add beam search or LM-assisted decoding to test whether collapse is partly decoding-limited.

## 11) Key Paths for Quick Access

- Project root: `<REPO_ROOT>/baselines/Cascade`
- Report: `<REPO_ROOT>/baselines/Cascade/DETAILED_REPORT.md`
- Main training log: `<REPO_ROOT>/baselines/Cascade/Cascade_Train_46188287.out`
- Inference summary: `<REPO_ROOT>/baselines/Cascade/inference_output/summary.json`
- Comparison log: `<REPO_ROOT>/baselines/Cascade/Infer_All_46188288.out`
