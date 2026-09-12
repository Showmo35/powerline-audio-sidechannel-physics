# M16_asr_on_plf_mels — from-scratch ASR on PowerLine-Flow generated mels

M14's generated mels look close to the real ones (mel_r ≈ 0.73, env_r ≈ 0.87)
but Whisper reads **~100 % WER** off the vocoded audio. Two live explanations:

- **(a) content absent** — the capture never carried the phonetics
  (M13/M14/M15 verdict), or
- **(b) content present but distorted** — a systematic "PLF accent" that
  zero-shot Whisper cannot read, but a model *trained on it* could.

M16 separates them: train a **char-CTC ASR from scratch (not Whisper)** on
(PLF-generated mel → true LibriSpeech text) pairs. It will learn whatever
consistent phonetic cues exist in the generated mels, however distorted.

**Control:** a twin model — identical architecture, hyperparameters, manifest,
and split — trained on the **real** reference mels. Its WER is the pipeline
upper bound; the twin gap is exactly the lexical content the PLF mels lack.

**Split:** strictly the same as M14/M15 — every 12th chunk held out. The flow
model may memorise its training chunks, so only test-chunk WER counts.

## Pipeline

```
M14 last.pt (EMA, frozen)                     full_manifest.json (M15, true text)
   │  gen_mels.py: tile each utterance into 4 s windows,
   │  sample mel per tile (32-step CFG ODE), concat, crop
   ▼
gen_mels/{utt_id}.npy  [80, T] log-mel ──┐
                                          ├─ train_asr.py --input gen   (experiment)
real reference mel (on the fly) ─────────┴─ train_asr.py --input real  (control)
                    both → conv ÷4 + transformer (8×256) + char-CTC (~8 M params)
```

## Files
`config.py` `gen_mels.py` (M14 sampler over all 202 chunks, resumable)
`dataset.py` (MelText: gen .npy / real on-the-fly, shared split) `models.py`
(ASRCTC) `train_asr.py` (CTC trainer, WER/CER eval per epoch, resume)
`text.py` (char vocab + WER, copied from M15) `run_gen.slurm` `run_asr.slurm`.

## Run
```bash
cd "<REPO_ROOT>/M16_asr_on_plf_mels"
GEN=$(sbatch --parsable --dependency=afterany:<M14_JOB> run_gen.slurm)
sbatch run_asr.slurm --input real --resume            # control, independent
sbatch --dependency=afterok:$GEN run_asr.slurm --input gen --resume
```

## Reading the result (test-chunk WER)
| asr_real | asr_gen | conclusion |
|---|---|---|
| low | ~100 % | content absent → **capture-side wall confirmed**, hardware next |
| low | clearly < 100 % | content present but distorted → pursue model-side (b) |
| high | — | pipeline/label problem; fix before concluding anything |
