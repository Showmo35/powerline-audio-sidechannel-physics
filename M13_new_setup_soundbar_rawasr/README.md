# M13_new_setup_soundbar_rawasr — raw-signal ASR sweep (Modules 3-7)

Sibling variant of [`../M10_new_setup_soundbar`](../M10_new_setup_soundbar).
**Same soundbar data** (LibriSpeech through a soundbar, USRP @ 200 kSps on the AC
cord). **Different method:** instead of M10's hand-designed AM-sideband mel or M12's
fixed harmonic stack, every model ingests the **raw 200 kSps capture** and a
**shared learned front-end** demodulates/downsamples it end-to-end. Five recognizer
/ enhancer architectures ported from **Modules 3-7** then sit on top.

> Why "raw"? You cannot feed 1-3 M raw samples straight into a CTC/Conformer
> (a Conformer's attention is O(N²)). The principled realization is a strided
> 1-D conv front-end (SincNet/wav2vec-style) that *learns* the demodulation — so
> nothing about the 60 Hz harmonics/sidebands is hard-coded. This is the strongest
> test of the project's core question: with a fully learned front-end on the raw
> capture, does word-level WER finally move off ~100%?

Modules 8 (Whisper-FT) and 9 (U-Net+Whisper) are **omitted on purpose** — M10 and
M12 already cover those, both pinned at ~100-103%.

## Pipeline

```
powerline .bin (200 kSps, raw)  ─►  RawFrontEnd  ─►  feature image [B,1,80,T]  ─►  arch  ─►  CTC-greedy word WER
   window per utterance via             (strided 1-D                                   (chunks 41-46 held out)
   manifest timing + .lag               conv stack,
   (read on the fly)                    learned)
```

| arch | source | model | training loss |
|------|--------|-------|---------------|
| `m3` | Module3 BeamSearchLM | CTC Conv+BiGRU | CTC |
| `m4` | Module4 PerceptualLoss | U-Net enhance → CTC | CTC + MSTFT(enh, clean mel) |
| `m5` | Module5 Conformer | Conformer CTC | CTC |
| `m6` | Module6 HybridCTCAttn | Conformer + CTC + attn decoder | 0.7·CTC + 0.3·attn CE |
| `m7` | Module7 FullSubNet | FullSubNet enhance → CTC | CTC + MSTFT(enh, clean mel) |

All archs are scored by **CTC-greedy word WER** for an apples-to-apples comparison.
Enhancement archs (m4, m7) additionally predict a clean log-mel (target from the
16 kHz reference wav) — the "on raw data" analogue of their original enhance→ASR
chain. The original Module-4 *perceptual* term used a frozen pretrained CTC encoder;
there is no such checkpoint for this data, so m4 uses L1+multi-scale-STFT consistency
to the clean mel instead (noted as a faithful-as-possible deviation).

## Files

| file | role |
|------|------|
| `config.py` | `RawConfig`: paths, `SHARED_MANIFEST`, front-end strides/kernels, target mel |
| `data_io.py` | windowed `.bin`/`.wav`/`.lag` readers (no ffmpeg) |
| `text.py` | char vocab (shared with Modules 3-7) + word-level WER |
| `frontend_raw.py` | `RawFrontEnd` — shared learned strided-conv front-end |
| `models.py` | the 5 archs (ported) + unified `RawASR` wrapper |
| `dataset.py` | `RawUttDataset` — raw windows read on the fly + clean-mel targets |
| `train.py` | unified trainer (`--arch m3..m7`), CTC/attn/enh loss, WER eval |
| `run_train.slurm` | one GPU job for one arch (`sbatch run_train.slurm m5 ...`) |
| `run_all.sh` | submit all five |
| `collect_results.py` | gather per-arch logs → one WER table |

No feature pre-build step: raw windows are read straight from the `.bin` captures
using the manifest's per-utterance timing + per-chunk `.lag` (powerline lags audio).

## Run

```bash
cd "<REPO_ROOT>/M13_new_setup_soundbar_rawasr"
bash run_all.sh --epochs 30 --batch 8 --lr 3e-4      # 5 GPU jobs
# or one at a time:
sbatch run_train.slurm m5 --epochs 30 --batch 8
python collect_results.py                            # WER table when done
```

Each job writes `outputs/<arch>/best.pt` and `outputs/<arch>/train_log.json`.

## Results (2026-06-29)

20 epochs each, train chunks 1-40 / test 41-46 (830 utts, 28,304 words), CTC-greedy WER.

| arch | model | params | best WER | final WER |
|------|-------|--------|----------|-----------|
| m3 | CTC BiGRU | 2.75M | 98.64% | ~113% |
| m4 | U-Net enh+CTC | 20.9M | 98.18% | ~102% |
| m5 | Conformer CTC | 9.48M | **96.56%** | ~100% |
| m6 | Hybrid CTC/Attn | 12.8M | 96.89% | ~101% |
| m7 | FullSubNet+CTC | 2.96M | 98.47% | ~115% |

**All five pin at ~97-99%.** The slightly-sub-100 "best" is NOT word recovery: the
hypotheses are degenerate (`a e e e e e e e e …` — the model collapses to spamming the
most frequent characters), and "best" is just the noisiest eval point where that spam
happens to align with a few high-frequency words; by the final epoch WER climbs back to
100-115% from insertions. This is the same failure mode as M10/M11/M12.

**Verdict — complete across front-ends now.** Fixed AM-sideband mel (M10 ~103%),
learned U-Net front-end on the harmonic stack (M12 100%), generative GAN (M11 99.8%),
and now a **fully learned front-end on the raw 200 kSps signal** with five different
recognizers (M13 ~97-99%) all fail identically. A learned front-end with full freedom
over the raw capture recovers no more than the hand-designed one. Word-level phonetic
information is not present in the 200 kSps soundbar AC-cord capture in a recoverable
form. The remaining lever is the **capture** (SNR / bandwidth / a higher-freq switching
carrier), not the model — see the base module's Step-6 capture-audit plan.

## Hypothesis / what to expect

The base module (Step 4) and M11/M12 already showed the bottleneck is the **capture**,
not the model: a fixed front-end (M10, ~103%), a learned U-Net front-end (M12, 100%),
and a generative GAN (M11, 99.8%) all pin near 100%. M13 closes the last gap — a
**fully learned front-end on the raw signal** with five different recognizers. If
these also pin at ~100%, the evidence is complete: word-level phonetic information is
not present in the 200 kSps soundbar AC-cord capture in a recoverable form, and the
next lever is capture-side (SNR / bandwidth / a higher-freq switching carrier), not
ML. A surprise drop in WER for any arch would be the first counter-evidence and worth
chasing.
