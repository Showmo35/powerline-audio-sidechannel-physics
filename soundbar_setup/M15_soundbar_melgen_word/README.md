# M15_soundbar_melgen_word — PowerLine-Flow-Word (PLF-W)

Extends **M14** (PowerLine-Flow) to chase **words**, not just the envelope. M14
showed the 0–16 kHz capture reconstructs the mel *envelope* well (env_r ≈ 0.85)
but carries ~0 lexical content (generated audio → **~100 % WER** via Whisper),
because the map is one-to-many. M15 attacks that on two fronts:

1. **Multi-stream input from the FULL 200 kHz capture** (M14 only saw 0–16 kHz):
   - **stream A** — raw 200 kHz waveform → learned strided-conv front-end. This
     alone contains every mains-harmonic **AM sideband** `n·60 ± f` (the >16 kHz
     detail M14 discarded).
   - **stream B** — wideband log-mel of the same 200 kHz window (0–100 kHz),
     exposing the harmonic/sideband structure explicitly.
2. **CTC word-loss** on the shared encoder, supervised by the **true LibriSpeech
   text** (exact utterance timing, reconstructed for all 202 chunks in
   `full_manifest.json` — validated against the M10 manifest to 0.3 ms). Total
   `loss = flow_matching + lambda_ctc · CTC`. Sweeping `lambda_ctc` trades mel_r
   for WER — the tradeoff the capture allows.

Windows are **utterance-aligned** (exact text per item); audio cropped/padded to
`max_dur_s` (16 s). Split by chunk (every 12th held out).

## Pipeline

```
raw 200 kHz (utterance window, lag-aligned)
  ├─ streamA: strided conv (÷2000) ──┐
  └─ streamB: wideband log-mel(0-100k)┴─ fuse ─ transformer ─► c[T,d]
       c ─► DiT flow-transformer ─► generated mel      (flow loss, masked)
       c ─► CTC head ─► char logits ─► WER             (CTC word loss)
```

## Files
`config.py` `data_io.py` `dataset.py` (PLFWUtterances, from manifest) `models.py`
(PLFW: MultiStreamEncoder + DiT + CTC head) `train.py` (joint trainer, mel_r/env_r
+ WER, resume) `text.py` (char vocab + CTC + WER) `full_manifest.json`.

## Run
```bash
cd "<REPO_ROOT>/soundbar_setup/M15_soundbar_melgen_word"
sbatch -M <cluster_name> run_train.slurm --epochs 30 --batch 8 --lambda-ctc 1.0 --out-dir outputs/run1
```

## The question
Does raw 200 kHz + AM sidebands + a punishing word-loss move WER off ~100 %?
- **WER drops** → the phonetic detail *is* in the high-freq sidebands; pursue.
- **WER stays ~100 %** → confirms M13/M14: the words aren't recoverable from this
  capture, and the wall is capture-side (SNR/bandwidth), not the model.
