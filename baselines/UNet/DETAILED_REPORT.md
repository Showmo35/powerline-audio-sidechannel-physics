# UNet Project Detailed Report

Generated: 2026-03-31 (America/New_York)  
Project path: `<REPO_ROOT>/baselines/UNet`

## 1) Executive Summary

`UNet` is the spectrogram reconstruction stack used upstream by Cascade and related transcription experiments.

The most complete recent root run is `BP_46141511` (March 29, 2026):
- bandpass data prep (random split) -> UNet training -> validation inference
- total runtime: **05:12:35**
- best checkpoint in that run: `checkpoints_bp/best_model.pt` with val loss **1.10689** at epoch 295
- val reconstruction metrics (10-sample eval):
  - Noisy vs clean MSE: **0.2635**
  - Pred vs clean MSE: **0.0754**
  - Pred-clean correlation: **0.7869**

The folder also contains an active subproject `UNet-CTC` (March 31, 2026) that trains a CTC recognizer on UNet-predicted mels:
- inference CER: **0.7260**
- inference WER: **1.0452**

## 2) Scope and Inventory

- File count: **219**
- Primary code:
  - `prepare_data.py`, `prepare_data_bandpass.py`, `prepare_data_hpc.py`
  - `model.py`, `model_bandpass.py`
  - `train.py`, `train_bandpass.py`
  - `inference.py`
- Main output families:
  - `checkpoints_bp/`, `inference_output_bp/`
  - `checkpoints_default/`, `inference_output_default/`
  - `checkpoints_recon/`
  - `checkpoints/` (full pipeline with perceptual loss)
- Nested subproject:
  - `UNet-CTC/` (prep/train/infer for transcription over UNet outputs)

## 3) Pipeline Variants in Root UNet

From `README.md` and checkpoint metadata:

1. **Default** (random split, no perceptual loss)
- best val loss: **1.1773633393** (`checkpoints_default/best_model.pt`, epoch 299)

2. **Recon** (temporal split, no perceptual loss)
- best val loss: **1.3030050365** (`checkpoints_recon/best_model.pt`, epoch 58)

3. **Full** (temporal split, with Whisper perceptual loss)
- best val loss: **1.4546802012** (`checkpoints/best_model.pt`, epoch 23)

4. **Bandpass pipeline** (`run_bandpass.sh`)
- best val loss: **1.1570416995** (`checkpoints_bp/best_model.pt`, epoch 174 metadata)
- in the dedicated `BP_46141511` run, training reached 300 epochs and best in-log val loss was **1.10689** at epoch 295.

## 4) Bandpass End-to-End Run (BP_46141511)

### 4.1 Data preparation (random split)

From `BP_46141511.out`:
- source folder: `May29_Alice`
- windows: 1.0s, hop: 0.25s
- bandpass: 50–4000 Hz
- resulting dataset:
  - train: **21,222**
  - val: **2,358**
  - shape: `(1, 80, 98)`
- output: `data/train_data_bp.npz`

Current `data/train_data_bp.npz` now has temporal-split counts from later prep:
- train: 21,219
- val: 2,361
- shape unchanged `(1, 80, 98)`

### 4.2 Training

From `BP_46141511.out`:
- model params: **18,110,849**
- objective: `L1 + MSTFT`
- epochs executed: 300
- best val loss in run: **1.10689**
- checkpoint path: `checkpoints_bp/`

### 4.3 Inference

From `BP_46141511.out` and `inference_output_bp/transcriptions.txt`:
- mode: validation, 10 consecutive val samples
- metrics:
  - Noisy-clean MSE 0.2635 -> Pred-clean MSE 0.0754
  - Pred-clean correlation 0.7869
- Whisper transcript quality in this sample remained poor:
  - predicted WER vs clean transcript reported as **100%**

## 5) UNet-CTC Subproject (UNet-CTC)

### 5.1 Run timeline (March 31, 2026)

- `UNet_Prep_46225506.out`: 00:08:03
- `UNet_CTC_46225507.out`: 00:04:33
- `UNet_Infer_46225508.out`: 00:00:12

### 5.2 Data and training

From `UNet_Prep_46225506.out` and NPZ inspection:
- temporal split dataset built with UNet-predicted mels + text labels
- train: 5,274 / val: 590
- `predicted_train`: `(5274, 1, 80, 498)`
- `predicted_val`: `(590, 1, 80, 498)`

From `UNet_CTC_46225507.out`:
- CTC params: 2,586,621
- best val_loss: **2.7563** (epoch 9)
- early stopping at epoch 39
- best checkpoint: `UNet-CTC/checkpoints/best_model.pt`

### 5.3 Inference

From `UNet_Infer_46225508.out` and `UNet-CTC/inference_output/results.json`:
- CER: **0.7260356260**
- WER: **1.0451688009**
- validation set size: 590

## 6) Warnings and Operational Notes

### 6.1 Root BP warnings

`BP_46141511.err`:
- repeated mpg123 ID3 comment warnings.

### 6.2 UNet-CTC warnings

`UNet-CTC/UNet_CTC_46225507.err`:
- torch warning about constructing tensors from tensors (`torch.tensor(text_lens, ...)`).

`UNet-CTC/UNet_Prep_46225506.err`:
- repeated mpg123 warnings
- ICC deprecation / unsupported option warnings

### 6.3 Path consistency risk

Scripts in `UNet-CTC` currently `cd` into `/UNet/transcribe`, but that path does not exist in the current tree (`UNet-CTC` is the actual directory).
This is a reproducibility risk if jobs are relaunched without path updates.

## 7) Artifacts and Sizes

- `checkpoints_bp/best_model.pt`: 72,527,307 bytes (~70 MB)
- `data/train_data_bp.npz`: 1,135,125,134 bytes
- `UNet-CTC/checkpoints/best_model.pt`: 31,070,581 bytes (~30 MB)
- `UNet-CTC/data/transcribe_data.npz`: 2,321,383,875 bytes

## 8) Risks / Gaps

1. Reconstruction quality improved strongly, but downstream transcription quality is still weak (high WER).
2. Multiple pipeline families coexist with different splits/losses; easy to mix incompatible artifacts.
3. `UNet-CTC` scripts reference stale path (`/UNet/transcribe`) relative to current folder layout.

## 9) Recommended Next Actions

1. Fix `UNet-CTC` launcher paths to `UNet-CTC` and re-run a clean prep/train/infer cycle.
2. Add explicit split and provenance tags in checkpoint/output names.
3. Evaluate transcription metrics on larger sample sets for each UNet variant, not only single illustrative outputs.
4. If ASR is priority, tune reconstruction objective toward recognizer-friendly features (or joint optimization with ASR head).

## 10) Key Paths

- Report: `<REPO_ROOT>/baselines/UNet/DETAILED_REPORT.md`
- Bandpass run log: `<REPO_ROOT>/baselines/UNet/BP_46141511.out`
- Bandpass best ckpt: `<REPO_ROOT>/baselines/UNet/checkpoints_bp/best_model.pt`
- Bandpass inference text: `<REPO_ROOT>/baselines/UNet/inference_output_bp/transcriptions.txt`
- UNet-CTC train log: `<REPO_ROOT>/baselines/UNet/UNet-CTC/UNet_CTC_46225507.out`
- UNet-CTC inference summary: `<REPO_ROOT>/baselines/UNet/UNet-CTC/inference_output/results.json`
