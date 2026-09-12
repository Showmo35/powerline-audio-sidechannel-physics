# M17_full200k_specgen — PLF-W on a lossless full-200 kHz input

Recreates **M15** (flow-matching DiT + CTC word-loss, utterance-aligned, same
chunk split) with the last model-side loophole closed. Every earlier attempt fed
the model a *reduced* view of the capture:

| module | what its input path discarded |
|---|---|
| M14 | everything above 16 kHz (decimation) |
| M15 stream A | conv stem funneled 200 k samples/s → 38.4 k dims/s |
| M15 stream B | wideband **mel** (lossy filterbank) |
| **M17** | **nothing** — zero-loss reshape; all compression is learned |

## The two changes

1. **Lossless patchify input.** The raw 200 kHz window is unfolded into 10 ms
   patches of 2000 samples with 50 % overlap — a pure reshape; every sample
   enters the model. A learned `Linear 2000→1024→d` is the only compression,
   trained end-to-end and *punished by the CTC word-loss* if it discards
   phonetics. No decimation, no STFT, no fixed filterbank.
2. **High-res target.** 513-bin linear-frequency log-STFT of the 16 kHz
   reference (n_fft 1024, hop 160, 100 fps) instead of the 80-bin mel — 6.4× the
   frequency resolution, and Griffin-Lim inverts it directly (better audio/WER
   ceiling than mel-GL).

Scaled up vs M15: d_model 512, 6 encoder + 10 DiT layers (~60 M params).

## Train/test separation (enforced)

Same chunk split as M14/M15/M16 (every 12th chunk held out), plus utterance-level
disjointness: the manifest contains duplicate utt_ids, so `dataset.split_rows`
**drops from train** any row whose utt_id or normalized text occurs in a test
chunk (test side untouched). `check_split.py` audits this and `train.py` asserts
it at startup — training refuses to run if any overlap remains.

## Pipeline

```
raw 200 kHz (utterance window, lag-aligned, RMS-normed)
  unfold(2000, hop 1000)          ← zero-loss reshape (2·T tokens)
  LayerNorm → Linear 2000→1024 → GELU → Linear →d     (learned compression)
  conv-pool ×2 → 100 fps → RoPE transformer ─► c[T,d]
     c ─► DiT flow-transformer ─► generated 513-bin log-STFT  (flow loss, masked)
     c ─► CTC head ─► char logits ─► WER                      (CTC word loss)
```

## Run

```bash
cd "<REPO_ROOT>/capacity_bound/M17_full200k_specgen"
python check_split.py                       # audit (also runs inside the job)
sbatch run_train.slurm --epochs 30 --batch 6 --out-dir outputs/run1
```

## Reading the result

- **WER clearly < 100 %** → the words were in the high-rate detail earlier
  preprocessing discarded — pursue immediately.
- **WER stays ~100 %** (CTC stuck ≈ 2.4, char-prior regime) → with a provably
  information-complete input, preprocessing is exonerated: the capture-wall
  verdict (M13–M16) is final and the next lever is hardware.
