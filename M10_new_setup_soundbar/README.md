# setup_soundbar — powerline-current → speech ASR (ground-up)

One module folder per physical capture **setup**. This module is the **soundbar**
setup: LibriSpeech `train-clean-100` played through a soundbar while a USRP samples
the AC-cord current at 200 kSps (real). Future setups (different appliance, probe,
or sample rate) get their own sibling folder (`setup_<name>/`) with their own
`config.py`; nothing is imported across setups.

## The idea

The soundbar's audio amplitude-modulates the mains harmonics, so the speech
spectrum lives in the **sidebands** around 60/120/180… Hz:

```
audio(f)  →  powerline energy at  n·f_mains ± f
```

The pipeline folds those sidebands back to baseband, mels them, and treats the
result as a speech spectrogram for reconstruction / ASR. **No neural vocoder** is
used (WaveRNN is ~9× slower than real-time → ~180 GPU-h for the full 20 h corpus);
Griffin-Lim reconstruction is ~real-time and good enough to feed Whisper, so the
exact same path scales to the whole dataset.

## Modules

| file | role |
|------|------|
| `config.py`      | all setup-specific constants (`SetupConfig`, data paths) |
| `data_io.py`     | windowed readers for `.bin` / `.wav` / `.lag` (no ffmpeg) |
| `frontend.py`    | `detect_mains`, **`am_sideband_mel`** — the blind front-end |
| `reconstruct.py` | `mel_to_audio` (Griffin-Lim), `save_wav` |
| `asr.py`         | Whisper wrapper + dependency-free `wer()` |
| `run_eval.py`    | **STEP 1** entrypoint: window → mel → audio → Whisper → WER |
| `labels.py`      | exact LibriSpeech utterance → (chunk, time) alignment + transcripts |
| `dataset.py`     | **STEP 2** per-chunk feature builder (utterance mels + transcripts) |
| `merge_manifest.py` | combine per-chunk manifests → `manifest.jsonl` + stats |
| `run.slurm` / `build_dataset.slurm` | SLURM submit (eval: gpu-exp; dataset: cpu array 1-46) |
| `outputs/`       | WER JSON, wavs, slurm logs, `dataset/` |

## Dataset (`outputs/dataset/`)

Built by `sbatch build_dataset.slurm` (46-task CPU array, ~30 s/chunk) then
`python merge_manifest.py`:

- `flac_index.json` — cached utterance→global-sample map (28,539 utts, 100.6 h)
- `chunk_XXX.npz` — float16 AM-sideband mels keyed by `utt_id` (757 MB total)
- `manifest.jsonl` — **6,610 utterances / 22.82 h / 223,891 words** across all 46
  chunks; each row: `utt_id, chunk, text, start_s, end_s, dur_s, lag_ms,
  f_mains_hz, n_mels, n_frames, shard`

Alignment is exact (verified: chunk_002's utterances match the Step-1 reference
transcript word-for-word). Powerline windows are lag-corrected per `.lag`.

## Run

```bash
cd "<REPO_ROOT>/setup_soundbar"
sbatch run.slurm                                   # chunk_002, 0–60 s, Whisper small
sbatch run.slurm --chunk chunk_005 --start 120 --dur 60 --model medium.en
```

Each run writes `outputs/eval_<chunk>_s<start>_d<dur>.json` with both transcripts
and the WER breakdown (S/D/I).

## Pipeline roadmap

- [x] **Step 1 — intelligibility gate**: WER of Griffin-Lim reconstruction vs
      clean-reference pseudo-truth (`run_eval.py`).
- [x] **Step 2 — dataset build**: 46 chunks → AM-sideband mels segmented to
      LibriSpeech utterances, paired with exact `.trans.txt` transcripts.
      → 6,610 utterances / 22.82 h in `outputs/dataset/manifest.jsonl`.
- [x] **Step 3 — fine-tune Whisper** on powerline mels (`train_whisper.py`,
      `whisper_features.py`, `data.py`). whisper-small, train chunks 1-40, test
      41-46. **Result: did NOT beat baseline** — best test WER 103.6% (final
      106.7%) vs 94.6%. Train loss fell (4.95→2.88) but the decoder collapsed to a
      generic LM prior, ~ignoring the encoder. Lesson: mel r 0.89 measured
      *envelope*, not phonetic content. Suspects: (a) feature-convention mismatch
      (htk@22k fed to a slaney@16k encoder); (b) word-level info may not survive
      the AM front-end.
- [x] **Step 4 — diagnostic matrix** (`diagnose.py`, eval-only). WER:
      A clean-native **4.0%** · B clean→our-adapter **56.3%** · C powerline-OTS
      **100%** · D powerline-FT test **102.8%** · E powerline-FT **train 105.5%**.
      **Verdict: the bottleneck is the front-end/signal, not the harness or
      convention.** Harness is fine (A=4%); adapter is lossy-but-not-fatal
      (B=56%); the cliff is B→C — swapping clean audio for the powerline mel
      through the *same* adapter erases all word content, and the model can't even
      fit training data (E≈100%). Phonetic info is not present in a learnable form.
- [ ] **Step 5 — front-end / signal-presence probe** (NOT more ASR training):
      does word-level phonetic info survive the channel at all? Frame-level phone
      discriminability test on powerline mels; try alternative demodulation
      (single best harmonic, coherent demod, per-harmonic channels) vs the
      8-harmonic magnitude sum; assess capture SNR. If phones are below chance,
      revisit capture (higher SNR / bandwidth) before any more ASR.

Status of the setup: 46 powerline `.bin` chunks (~23 h) paired with 16 kHz
reference `.wav` chunks; mains 60 Hz; powerline lags audio (see each `.lag`).
