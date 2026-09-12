# Experiments Project Detailed Report

Generated: 2026-03-31 (America/New_York)  
Project path: `<REPO_ROOT>/baselines/Experiments`

## 1) Executive Summary

This folder contains three controlled ASR experiments on the same E2E dataset (train=5274, val=590):
- `exp1_pure_ctc`: CTC encoder directly on noisy mel (no UNet).
- `exp2_ctc_weighted`: E2E UNet+CTC with higher CTC weight (`alpha=0.5`, `beta=2.0`).
- `exp3_larger_ctc`: Cascade with pretrained UNet and larger CTC head (GRU 512, 3 layers).

Latest comparison run (`Exp_Infer_46228965.out`, Mar 31, 2026) reports:
- **Best CER**: `exp1_pure_ctc` (0.6865)
- Best WER: `exp1_pure_ctc` (0.9403)
- `exp2_ctc_weighted` was close in CER (0.6913) but worse in WER (1.0978)
- `exp3_larger_ctc` underperformed both (CER 0.7206)

## 2) Scope and Inventory

- File count: **42**
- Core code files:
  - `model.py`
  - `train_pure_ctc.py`
  - `inference_all.py`
  - `dataset.py`
- SLURM launchers:
  - `run_exp1.sh`, `run_exp2.sh`, `run_exp3.sh`, `run_inference.sh`, `run_all.sh`
- Outputs:
  - `exp1_pure_ctc/`, `exp2_ctc_weighted/`, `exp3_larger_ctc/`
  - `results/` (full comparison)
  - `results_smoke/` (single-sample smoke comparison)

## 3) Experiment Definitions

### 3.1 Exp1: Pure CTC (no UNet)

From `model.py` and `train_pure_ctc.py`:
- Model: `PureCTCEncoder` only (conv downsampling + BiGRU + linear).
- Params: **2,586,621**.
- Objective: CTC only.

### 3.2 Exp2: CTC-weighted E2E

From `Exp2_46225283.out`:
- Model: full E2E model (`24,461,563` params).
- Mode: `ctc`.
- Weights: recon=0.5, ctc=2.0, seq=0.0.

### 3.3 Exp3: Larger CTC Cascade

From `Exp3_46225284.out`:
- Pretrained UNet loaded from `UNet/checkpoints_bp/best_model.pt`.
- CTC encoder enlarged to hidden=512, layers=3.
- Params:
  - UNet: 18,110,849
  - CTC: 13,033,981
  - Total: **31,144,830**
- Two-phase training (freeze UNet, then fine-tune both).

## 4) Training Runs and Outcomes (March 31, 2026)

### 4.1 Runtime

- `Exp1_46225282.out`: 00:12:54
- `Exp2_46225283.out`: 00:41:11
- `Exp3_46225284.out`: 01:24:36

### 4.2 Exp1 Results

From `Exp1_46225282.out`:
- Epochs run: 80 (early stopping)
- Best val loss: **2.866561** (epoch 50)
- Best log CER: **0.7187** (epoch 64)
- Best checkpoint metadata:
  - epoch 50
  - val_loss 2.8665611099552466
  - val_cer 0.7519206609112332

### 4.3 Exp2 Results

From `Exp2_46225283.out`:
- Epochs run: 80 (early stopping)
- Best val loss: **6.246653** (epoch 50)
- Best log CTC-CER: **0.7042** (epoch 57)
- Best checkpoint metadata:
  - epoch 50
  - val_loss 6.246653234636462

### 4.4 Exp3 Results

From `Exp3_46225284.out`:
- Phase 1 (frozen UNet): 71 epochs (early stop)
  - best val loss: **2.766781** (phase epoch 41)
  - best CER in phase: **0.7179** (phase epoch 69)
- Phase 2 (fine-tune both): 30 epochs (early stop)
  - best val loss: **3.48787**
  - best CER in phase: **0.7096**
- Best checkpoint metadata:
  - global epoch 40
  - phase: Phase 1 (UNet frozen)
  - val_loss 2.7667810208088643
  - val_cer 0.7564034419214025

Observation:
- As in Cascade, checkpoint selection is loss-driven; lower CER in later epochs/phases does not necessarily become `best_model.pt`.

## 5) Inference Comparison

### 5.1 Failed first inference job

`Exp_Infer_46225285.err`:
- `ImportError: cannot import name 'PureCTCEncoder' from 'model'` because Python resolved `Cascade/model.py` instead of local `Experiments/model.py`.

### 5.2 Fixed inference job

`Exp_Infer_46228965.out` (00:00:11):
- Import issue fixed via explicit dynamic module loading in `inference_all.py`.
- Results saved to `results/summary.json` and `results/comparison.txt`.

Final 10-sample summary (`results/summary.json`):
- `exp1_pure_ctc`: CER **0.6864688443**, WER **0.9403362573**
- `exp2_ctc_weighted`: CER **0.6913307432**, WER **1.0977567079**
- `exp3_larger_ctc`: CER **0.7206194635**, WER **1.0110410217**

## 6) Artifacts

- `exp1_pure_ctc/best_model.pt`: 10 MB
- `exp2_ctc_weighted/best_model.pt`: 94 MB
- `exp3_larger_ctc/best_model.pt`: 119 MB
- `results/comparison.txt`: full per-sample reference/pred comparison (10 samples)
- `results_smoke/`: early smoke-test comparison

## 7) Warnings / Errors

- Training stderr files (`Exp1`, `Exp2`, `Exp3`) are empty.
- One inference run failed (`Exp_Infer_46225285`), then a later run succeeded (`Exp_Infer_46228965`).

## 8) Risks / Gaps

1. Model selection metric (val loss) and reporting metric (CER/WER) can diverge.
2. Inference currently uses only first 10 val samples for comparison; not full-val ranking.
3. Predictions in all three experiments are still linguistically degraded (token repetition/compression).

## 9) Recommended Next Actions

1. Add full-validation comparison mode in `inference_all.py` (`n_samples = all val`).
2. Save both best-loss and best-CER checkpoints for each experiment.
3. Compare decoding variants (beam search / LM-assisted) before architecture changes.
4. If Exp2 is retained, tune `alpha:beta` since WER regressed despite similar CER.

## 10) Key Paths

- Report: `<REPO_ROOT>/baselines/Experiments/DETAILED_REPORT.md`
- Main comparison: `<REPO_ROOT>/baselines/Experiments/results/comparison.txt`
- Summary JSON: `<REPO_ROOT>/baselines/Experiments/results/summary.json`
- Inference fix log: `<REPO_ROOT>/baselines/Experiments/Exp_Infer_46228965.out`
