# E2E: Powerline Envelope -> Text

This folder now supports both requested end-to-end objectives:

1. **Direct envelope -> text model** (`--mode ctc`)
2. **Envelope-conditioned language model** (`--mode conditioned`)

You can also train both jointly (`--mode joint`).

## HPC Job Scripts

This folder includes `sbatch` scripts mirroring `UNet` style:

- `run_e2e_prepare.sh`
- `run_e2e_ctc.sh`
- `run_e2e_conditioned.sh`
- `run_e2e_joint.sh`
- `run_e2e_inference.sh`
- `run_all_e2e.sh` (submits all with dependencies)

Submit everything (recommended):

```bash
cd "<REPO_ROOT>/baselines/E2E"
./run_all_e2e.sh
```

## Data (manual local run)

Prepare windows from powerline `.bin` and MP3 alignment:

```bash
python prepare_data.py --out_dir data --out_name e2e_data.npz
```

Output file:
- `data/e2e_data.npz` with noisy/clean mel + text labels.

## Train Objective 1: Direct Envelope -> Text (CTC, local run)

```bash
python train_e2e.py \
  --data data/e2e_data.npz \
  --mode ctc \
  --out checkpoints_ctc
```

## Train Objective 2: Envelope-Conditioned Decoder (local run)

```bash
python train_e2e.py \
  --data data/e2e_data.npz \
  --mode conditioned \
  --out checkpoints_conditioned
```

## Train Both Jointly End-to-End (local run)

```bash
python train_e2e.py \
  --data data/e2e_data.npz \
  --mode joint \
  --out checkpoints_joint
```

## One-Shot Script (HPC submit launcher)

`run_all_e2e.sh` now submits HPC jobs and prints job IDs.

## Validation Inference

Run standalone val-set inference (saves text predictions + CER + mel plots):

```bash
python inference_e2e.py \
  --data data/e2e_data.npz \
  --ckpt checkpoints_joint/best_model.pt \
  --out inference_output_joint \
  --mode joint \
  --n_samples 5
```

HPC batch version (runs CTC + conditioned + joint, each on 5 val samples):

```bash
sbatch run_e2e_inference.sh
```

## Optional: text-only char LM pretraining

```bash
python train_lm.py --out_path lm.pt
```

This LM is not required for the two end-to-end objectives, but can still be
used later for shallow fusion decoding experiments.
