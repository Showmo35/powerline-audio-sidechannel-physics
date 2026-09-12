# setup_soundbar_learned — learned front-end (setup variant)

Sibling variant of [`../setup_soundbar`](../setup_soundbar). **Same physical setup**
(LibriSpeech through a soundbar, USRP @ 200 kSps on the AC cord); **different
method.** Instead of the fixed 8-harmonic AM-sideband *sum* — which Step 4 of the
base module proved erases phonetic content — a **U-Net learns powerline → clean
mel enhancement** from the paired data.

This module is self-contained. It consumes only the base module's utterance
**manifest** (a data artifact: `utt_id, chunk, text, timing`) and owns its own
features, model, and outputs.

## Why this exists

Step 4 (base module) diagnostic: the bottleneck is the front-end, not the ASR
harness. The fixed sum's match to clean speech, measured here:

```
fixed 8-harmonic sum vs clean log-mel:  r_all 0.34 · r_speech 0.33 · r_formant 0.0
```

i.e. it tracks the *envelope* but has **no formant structure**. The learned model
is given the un-collapsed signal — each of the 16 sidebands (8 harmonics ×
upper/lower) demodulated to baseband as its own channel — and learns the optimal
combination.

## Pipeline

```
powerline .bin → per-harmonic mel STACK [16,80,T] ─┐
                                                    ├─ U-Net → predicted clean mel [80,T]
clean .wav     → clean log-mel TARGET    [80,T]  ───┘            ↑ trained, L1 loss
```

| file | role |
|------|------|
| `config.py`        | variant config (`SetupConfig`, paths, `SHARED_MANIFEST`) |
| `data_io.py`       | windowed `.bin`/`.wav`/`.lag` readers |
| `frontend_stack.py`| `am_sideband_stack` (16-ch input), `clean_logmel` (target), mains |
| `dataset_enh.py`   | per-chunk builder → `outputs/dataset_enh/chunk_XXX.{x,y}.npz` |
| `model_unet.py`    | `MelUNet` (~2.1M params, 16→80 mel image-to-image) |
| `train_enh.py`     | train + **probe**: learned vs fixed mel-match to clean |
| `build_stack.slurm`| CPU array 1-46 (feature build) |
| `run_train_enh.slurm` | gpu-exp (training) |

## Run

```bash
sbatch build_stack.slurm                 # ~20 GB float16 stacks + targets
sbatch run_train_enh.slurm --epochs 40 --batch 32 --lr 3e-4
```

## The decision metric

`train_enh.py` reports, on held-out chunks 41-46, the correlation of the
**learned** mel and the **fixed** mel against the clean reference — overall, on
speech frames, and **in the formant band on speech frames** (the key number).

- learned ≫ fixed in formant band ⇒ phonetic info IS recoverable → retry ASR on
  the enhanced mel.
- learned ≈ fixed ≈ 0 ⇒ the information is not in the capture → fix the capture
  (SNR / bandwidth), not the algorithm.

## Roadmap / results

- [x] Step 5a — build stacks (`build_stack.slurm`): 46 chunks, 14 GB.
- [x] Step 5b — train U-Net (`train_enh.py`). **Features improved 4×**: formant-band
      r (speech) **0.10 (fixed) → 0.44 (learned)** on held-out chunks 41-46.
- [x] Step 5c — ASR test (`asr_eval.py`). **WER did NOT move**: learned mel = **100%**
      = fixed; clean-mel-through-same-adapter ceiling 76.8%; honest learned→GL→native
      = 100% (worse than base 94.6%, which had reference leakage).
- **Verdict: the bottleneck is the CAPTURE, not the model.** 4× better features →
      zero WER gain ⇒ word-level phonetic detail is not in the 200 kSps soundbar
      capture in recoverable form. A generative vocoder would only hallucinate
      plausible-but-wrong words (the `speech_presence.py` illusion).
- [ ] **Step 6 — capture-side audit (not ML):** sideband SNR vs noise floor per
      harmonic; check whether speech rides a Class-D switching carrier
      (250 kHz–1 MHz, above the 200 kSps 0–100 kHz window) using the wideband
      captures; decide on recapture at higher SNR / bandwidth.
