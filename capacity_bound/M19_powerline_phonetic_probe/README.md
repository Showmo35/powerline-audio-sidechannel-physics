# M19_powerline_phonetic_probe — is phonetic info physically in the powerline?

Not another end-to-end ASR. A direct **regression probe** that measures how much
of the audio's **spectral shape** (the phonetic part) is recoverable from the
powerline — cleanly separated from loudness.

## The one number that was never measured

Every prior result folds the envelope in: mel_r 0.73, env_r 0.88. The phonetic
question is about the **envelope-removed** spectrum. So we predict the true audio
log-mel from powerline features and report, on held-out chunks:

- `env_r` — loudness only (corr of per-frame mean energy)
- **`PHON_r`** — **envelope-removed**: subtract each frame's mean, then correlate →
  the fraction of audio spectral *shape* (phonemes) recovered ← **the answer**
- `phon_r_chance` — shuffled-pairing floor for PHON_r

## Feature families

- **env** — per-frame RMS/loudness of the 200 kHz signal. Baseline; PHON_r ≈ 0 by
  construction (a scalar per frame has no spectral shape).
- **wide** — wideband log-STFT of the FULL 200 kHz (n_fft 4096 → 48.8 Hz bins,
  0–100 kHz). Contains every mains harmonic *n·60* and its AM sidebands *n·60 ± f*
  — the only place the audio spectrum could physically hide. If phonetic info is
  anywhere in the powerline, a probe on `wide` can reach it.

## Reading the result

| PHON_r (wide) | meaning |
|---|---|
| ≈ 0 (≈ env, ≈ chance) | no phonetic info in the powerline — capture wall is **physical**; next step is hardware |
| ≫ env, ≫ chance | audio spectrum **is** imprinted (likely in the sidebands); M15 just didn't extract it → pursue this representation |

## Files
`config.py` `data_io.py` (full 200 kHz reader) `dataset.py` (frame-aligned
wide-STFT / env / true-mel windows, chunk split) `probe.py` (conv regression +
envelope-removed metrics) `run_probe.slurm`.

## Run
```bash
cd "<REPO_ROOT>/capacity_bound/M19_powerline_phonetic_probe"
sbatch run_probe.slurm            # trains env then wide, prints PHON_r each epoch
```
