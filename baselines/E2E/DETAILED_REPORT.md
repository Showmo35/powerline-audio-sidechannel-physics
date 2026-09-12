# E2E Project Detailed Report

Generated: 2026-03-31 (America/New_York)  
Project path: `<REPO_ROOT>/baselines/E2E`

## 1) Executive Summary

`E2E` implements end-to-end powerline-to-text modeling with three training modes:
- `ctc`: direct CTC decoding from reconstructed mel features.
- `conditioned`: envelope-conditioned autoregressive decoder.
- `joint`: both heads trained together.

Current state has mixed artifact provenance:
- `checkpoints_conditioned` and `checkpoints_joint` are from **March 29, 2026** (random split run).
- `checkpoints_ctc` was overwritten on **March 30, 2026** by a retrain launched from the Cascade pipeline (temporal split).
- Current inference outputs are not all from the same data split/run date.

This means cross-mode comparison in the folder is currently not fully apples-to-apples.

## 2) Scope and Inventory

- File count: **128**
- Main source files: `model.py`, `train_e2e.py`, `inference_e2e.py`, `prepare_data.py`, `dataset.py`
- Run scripts: `run_e2e_prepare.sh`, `run_e2e_ctc.sh`, `run_e2e_conditioned.sh`, `run_e2e_joint.sh`, `run_e2e_inference.sh`, `run_all_e2e.sh`
- Checkpoint dirs:
  - `checkpoints_ctc/`
  - `checkpoints_conditioned/`
  - `checkpoints_joint/`
- Inference dirs:
  - `inference_output_ctc/`
  - `inference_output_conditioned/`
  - `inference_output_joint/`

## 3) Model and Objective Design

From `model.py` and `train_e2e.py`:
- Shared backbone: U-Net (`PowerlineUNet`) predicts clean mel from noisy mel.
- Text heads:
  - `CTCEncoder` (conv downsampling + BiGRU + linear to vocab=29).
  - `EnvelopeConditionedDecoder` (Transformer decoder conditioned on acoustic memory).
- Composite loss in `E2ELoss`: reconstruction + CTC + seq, weighted by (`alpha`, `beta`, `gamma`).
- Mode mapping:
  - `ctc`: `alpha=1`, `beta=1`, `gamma=0`
  - `conditioned`: `alpha=1`, `beta=0`, `gamma=1`
  - `joint`: `alpha=1`, `beta=1`, `gamma=0.5`

## 4) Data Snapshot

Current `data/e2e_data.npz` (mtime: 2026-03-30 14:20 EDT):
- `noisy_train`: `(5274, 1, 80, 498)`
- `clean_train`: `(5274, 1, 80, 498)`
- `text_train`: `(5274, 200)`
- `text_len_train`: `(5274,)`
- `noisy_val`: `(590, 1, 80, 498)`
- `clean_val`: `(590, 1, 80, 498)`
- `text_val`: `(590, 200)`
- `text_len_val`: `(590,)`

Note: the original E2E prep job (`E2E_Prep_46142657.out`, Mar 29) used **random split** and produced train=5277 / val=587. The current dataset is later temporal split output.

## 5) Training Runs and Outcomes

### 5.1 Original E2E runs (March 29, 2026)

From folder logs:
- `E2E_Prep_46142657.out`: 00:09:15
- `E2E_CTC_46142658.out`: 01:37:36
- `E2E_CondLM_46142659.out`: 02:41:26
- `E2E_Joint_46142660.out`: 02:47:12

Log-level results:
- CTC run (`E2E_CTC_46142658.out`):
  - 180 epochs completed
  - best val loss in log: **3.54617**
  - best CTC-CER in log: **0.6673**
- Conditioned run (`E2E_CondLM_46142659.out`):
  - 220 epochs completed
  - best val loss in log: **1.03144**
  - seq CER unstable in log (low early value, then much higher later)
- Joint run (`E2E_Joint_46142660.out`):
  - 220 epochs completed
  - best val loss in log: **3.37843**
  - best CTC-CER in log: **0.5891**

### 5.2 Current checkpoint metadata (as stored now)

- `checkpoints_ctc/best_model.pt` (mtime Mar 30):
  - epoch: 42
  - val_loss: **3.9628427351**
  - mode args: `ctc`
  - This corresponds to the later temporal-split retrain run (`/Cascade/Train_E2E_CTC_46188286.out`).
- `checkpoints_conditioned/best_model.pt` (mtime Mar 29):
  - epoch: 209
  - val_loss: **1.0314402884**
- `checkpoints_joint/best_model.pt` (mtime Mar 29):
  - epoch: 200
  - val_loss: **3.3784335973**

## 6) Inference Outputs

Two inference logs exist in this folder:
- `E2E_Infer_46185884.out` (00:00:44)
- `E2E_Infer_46186675.out` (00:01:06)

Current summary files on disk:
- `inference_output_ctc/summary.json` (mtime Mar 30 18:54):
  - `n_val_total`: 590
  - `avg_cer_ctc`: **0.6567042668**
- `inference_output_joint/summary.json` (mtime Mar 30 13:21):
  - `n_val_total`: 587
  - `avg_cer_ctc`: **0.5987549637**
  - `avg_cer_seq`: **2.1565629987**
- `inference_output_conditioned/summary.json` (mtime Mar 30 15:27):
  - `n_val_total`: 587
  - `avg_cer_seq`: **2.1922971637**
  - device recorded as `cpu`

Interpretation:
- Joint CTC head appears best among available 5-sample summaries.
- Conditioned seq predictions remain poor by CER.
- Mixed `n_val_total` values (587 vs 590) confirm inconsistent evaluation provenance.

## 7) Warnings and Errors

`.err` files are mostly clean except prep warnings:
- repeated mpg123 ID3 comment warnings.
- Whisper warning: `FP16 is not supported on CPU; using FP32 instead`.

No training crash logs were found in E2E run stderr files.

## 8) Risks / Gaps

1. Mixed-provenance checkpoints and inference outputs reduce result comparability.
2. Only 5-sample inference summaries are too small for reliable model ranking.
3. Conditioned/joint seq decoding quality is low in current outputs (long repetitive text).
4. Folder-local training logs do not represent the newest CTC checkpoint lineage unless Cascade logs are consulted.

## 9) Recommended Next Actions

1. Re-run all three modes from one consistent split (prefer temporal) and regenerate all three inference outputs together.
2. Evaluate on larger validation subsets (or full val) for stable CER/WER.
3. Track checkpoint provenance explicitly in filenames or metadata (`split`, `date`, `jobid`).
4. Keep a single comparison table built from same-date artifacts only.

## 10) Key Paths

- Report: `<REPO_ROOT>/baselines/E2E/DETAILED_REPORT.md`
- Data: `<REPO_ROOT>/baselines/E2E/data/e2e_data.npz`
- CTC ckpt: `<REPO_ROOT>/baselines/E2E/checkpoints_ctc/best_model.pt`
- Conditioned ckpt: `<REPO_ROOT>/baselines/E2E/checkpoints_conditioned/best_model.pt`
- Joint ckpt: `<REPO_ROOT>/baselines/E2E/checkpoints_joint/best_model.pt`
- Latest CTC retrain log (external): `<REPO_ROOT>/baselines/Cascade/Train_E2E_CTC_46188286.out`
