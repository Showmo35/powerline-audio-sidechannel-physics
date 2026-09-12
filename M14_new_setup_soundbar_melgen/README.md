# M14_new_setup_soundbar_melgen — PowerLine-Flow (PLF)

**Conditional rectified-flow DiT that regenerates the audio mel-spectrogram from
the powerline capture.**  A fundamentally different goal from M10–M13 (which all
tried *word-level ASR* and pinned at ~100 % WER): instead of recovering text, M14
**generates** the audio log-mel conditioned on the powerline signal.

## Why generative (not regression)

The lag-aligned 200 kSps capture matches the audio **energy envelope** almost
perfectly — median frame-RMS cross-correlation ≈ **0.88** after the lag fix
(`../Capture_Analysis/estimate_lag.py`).  But fine phonetic detail is weak
(M13's verdict). So powerline→mel is **one-to-many**: many plausible mels share
one envelope.  An L1/regression model averages them into a gray blur. A
**conditional generative** model instead samples one sharp, realistic mel that is
consistent with the envelope — the right tool for "perfectly regenerate".

Paradigm: **rectified flow matching** (Voicebox / Stable-Audio family) with a
**DiT** backbone — current SOTA for conditional audio/mel generation, chosen over
classic diffusion (more steps) and GANs (unstable, mode-collapse).

## Pipeline

```
powerline .bin (200 kSps)
   │  + per-chunk .lag  (RMS-envelope lag, measured in Capture_Analysis)
   │  decimate 200k→32k  ⇒ raw 0-16 kHz waveform   ← model input
   ▼
PowerlineEncoder  (strided-conv stem + RoPE transformer)  ─►  condition c[T,d]
   ▼
DiT flow-transformer   x_t (noisy mel) ⊕ c , adaLN-zero(flow-time t), RoPE
   ▼   trained by  ‖v_θ(x_t,t,c) − (x1−x0)‖²
generated log-mel  ◄─ few-step Euler ODE (noise→mel), classifier-free guided
```

- **All 202 chunks** are used; windows (4 s) are read on the fly, lag-aligned.
- Input is the **raw 0-16 kHz** waveform at 32 kHz (decimation = the band-limit).
- Target is the 16 kHz reference log-mel (80 bins, 100 fps, 400 frames/window).
- Split is **by chunk** (every 12th held out, 16 test chunks).

## Files

| file | role |
|------|------|
| `config.py` | `PLFConfig`: rates, windowing, mel target, encoder strides, DiT dims, flow params |
| `data_io.py` | on-the-fly `.bin`→32 kHz / `.wav` / `.lag` window readers (no ffmpeg) |
| `dataset.py` | `PLFWindows`: (raw 0-16 kHz, log-mel) pairs over all 202 chunks + mel stats |
| `models.py` | `PLF` — condition encoder + DiT (adaLN-zero, RoPE) + flow-matching loss + ODE sampler |
| `train.py` | flow-matching trainer; EMA; eval = mel L1 / mel r / envelope r on held-out chunks |
| `viz.py` | random held-out windows: input STFT · target mel · generated mel · envelope overlay |
| `run_train.slurm` | one GPU job (train then render `outputs/plf_samples.png`) |

## Run

```bash
cd "<REPO_ROOT>/M14_new_setup_soundbar_melgen"
sbatch run_train.slurm --epochs 60 --batch 32 --lr 3e-4
# samples:
python viz.py --ckpt outputs/best.pt --n 6 --out outputs/plf_samples.png
```

## Metrics

- **mel L1** (log-mel) — lower better; checkpoint selection.
- **mel r** — Pearson correlation of the full generated vs target mel.
- **env r** — Pearson of the per-frame energy envelope (the quantity the capture
  is known to preserve; expected to be high).
