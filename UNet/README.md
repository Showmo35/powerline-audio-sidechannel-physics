# Powerline-to-Audio Reconstruction

Recovers speech from powerline electromagnetic measurements using a U-Net that reconstructs clean mel spectrograms from IQ-demodulated powerline signals.

## Pipeline

```
Raw .bin (200kHz) -> IQ demod -> 16kHz -> mel spectrogram -> UNet -> clean spectrogram -> Whisper -> text
```

## Setup

an HPC cluster with conda env `tf_gpu`. All scripts use SLURM.

## Three Pipelines

### 1. Default (random split, no Whisper)

Simple UNet reconstruction with random 90:10 train/val split.

```bash
sbatch run_default_prepare.sh   # prepare data with random split (once)
sbatch run_default_train.sh     # train: L1 + MSTFT
sbatch run_default_inference.sh # inference: val-only spectrogram + Griffin-Lim audio
```

| | |
|---|---|
| Data | `data/train_data_random.npz` |
| Checkpoints | `checkpoints_default/` |
| Output | `inference_output_default/` |
| Split | Random shuffle 90:10 |
| Loss | L1 + Multi-Scale STFT |

### 2. Recon (temporal split, no Whisper)

Same UNet reconstruction but with temporal split per chapter.

```bash
# Uses existing train_data.npz (temporal split)
sbatch run_recon_train.sh
sbatch run_recon_inference.sh
```

| | |
|---|---|
| Data | `data/train_data.npz` |
| Checkpoints | `checkpoints_recon/` |
| Output | `inference_output_recon/` |
| Split | Temporal: first 90% of each chapter = train, last 10% = val |
| Loss | L1 + Multi-Scale STFT |

### 3. Full (temporal split, Whisper perceptual loss)

Adds Whisper perceptual loss for speech-aware optimization + Whisper transcription at inference.

```bash
# Uses existing train_data.npz (temporal split)
sbatch run_training.sh
sbatch run_inference.sh
```

| | |
|---|---|
| Data | `data/train_data.npz` |
| Checkpoints | `checkpoints/` |
| Output | `inference_output/` |
| Split | Temporal: first 90% of each chapter = train, last 10% = val |
| Loss | L1 + Multi-Scale STFT + Whisper perceptual |

## Data Preparation

```bash
# Temporal split (used by Recon and Full pipelines)
python prepare_data_hpc.py --split temporal

# Random split (used by Default pipeline)
python prepare_data_hpc.py --split random --out_name train_data_random.npz
```

## Files

| File | Purpose |
|------|---------|
| `prepare_data_hpc.py` | Raw .bin + .mp3 -> paired mel spectrograms (`--split random\|temporal`) |
| `model.py` | U-Net architecture, MSTFT loss, Whisper perceptual loss |
| `dataset.py` | PyTorch dataset with augmentation (gain, noise, masking) |
| `train.py` | Training loop with early stopping (`--percep_w 0` disables Whisper) |
| `inference.py` | 3 modes: `full`, `val`, `segment` (overlap-add + Whisper transcription + WER) |

## Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--base_ch` | 64 | U-Net width (18M params) |
| `--l1_w` | 1.0 | L1 reconstruction weight |
| `--mstft_w` | 1.0 | Multi-scale STFT weight |
| `--percep_w` | 1.0 | Whisper perceptual weight (0 = recon-only) |
| `--patience` | 30 | Early stopping epochs |
| `--batch` | 32 | Batch size |
| `--split` | temporal | Data split: `temporal` or `random` |
