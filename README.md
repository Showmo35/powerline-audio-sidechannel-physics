# Powerline Acoustic Side-Channel — Physics & Capacity Bound

Channel-physics characterization and the open-vocabulary capacity bound for
a power-line conducted-audio side channel: a device drawing power from an
AC outlet leaks the audio it plays onto the electrical line it's plugged
into. Target venues: IEEE S&P, USENIX Security, ACM CCS, NDSS.

Companion repo: `powerline-audio-sidechannel-attack` — the positive-result
half (constrained-vocabulary digit/command recovery) that this negative
result and physical explanation motivate. See `PAPER_DIRECTION.md` and
`FULL_PAPER_DIRECTIONS.md` for the full paper thesis and execution plan.

This repository is released anonymously for double-blind review; author
and institutional identity have been withheld from all files.

## Headline result

Open-vocabulary speech content is **physically absent** from the conducted
power line — confirmed by ~10 independent methods (end-to-end ASR, a
phonetic probe, retrieval, a context/language-model decoder, word-pair
physics, dual-band analysis) and by controlled chirp/tone experiments on
multiple devices. Only the loudness/prosody envelope survives, in the
0–16 kHz mains-sideband band; fine phonetics are averaged out by the
device's power supply.

## Layout (roughly chronological)

- **Channel physics**: `Capture_Analysis/` (chirp/tone reconstruction,
  cross-device alignment, mains-sideband analysis), plus root-level
  `generate_tone_sweep.py`, `phonetic_probe.py`, `powerline_bandpass_spectrogram.py`,
  `simple_rx.grc`, `GNUradio_codes/`.
- **Early ASR-pipeline baselines** (established the initial "content is
  missing" observation before it was rigorously confirmed):
  `Module3_BeamSearchLM/` … `Module9_UNetWhisper/`, `UNet/`, `UNet_Old/`,
  `FullSubNetPlus/`, `Cascade/`, `E2E/`, `Experiments/`, `Training_Scripts/`.
- **Soundbar setup iterations**: `M10_new_setup_soundbar/` through
  `M15_soundbar_melgen_word/` (raw ASR, GAN/learned front-ends, mel
  generation).
- **The capacity-bound investigation (M16–M27)**: ASR-on-generated-mels,
  full-resolution spectrogram generation, a ViT+CTC+LM pipeline, a direct
  phonetic-content probe, a closed-set word classifier, open-vocabulary
  cross-modal retrieval, a context/language-model rescue attempt (tested to
  its honest conclusion — it fails), word-pair physics, and dual-band
  analysis. Each is an independent line of evidence for the same
  conclusion.
- **Mains-harmonic-comb hypothesis (also negative)**: `M28_mains_comb/`,
  `M31_comb_retrieval_generator/` — tests and rejects the idea that
  phonetic content rides on individual mains harmonics.
- **Multi-device validation**: monitor and TV captures
  (`Capture_Analysis/check_monitor_align.py`,
  `M22_retrieval_generator/outputs_monitor/`) confirming the channel (and
  the open-vocab wall) generalizes beyond the original soundbar.
- Raw capture data: `Powerline_Data_Captures/` (not fully mirrored here —
  see its own `readme.md`).

## Note on scope

`M28_constrained_vocab/`, `M29_constrained_supervised/`, and
`M30_retrieval_constrained/` — the constrained-vocabulary attack — have
been split out into the companion `powerline-audio-sidechannel-attack`
repo, since they answer a different question (can we recover a *specific
kind* of secret) than this repo (what does the channel physically carry,
and what's the honest ceiling on open speech).

## Setup

Most modules are standalone: a `config.py`, a training/eval script, and a
SLURM (`*.slurm`/`*.sh`) submit script. Typical dependencies across the
project: `torch`, `numpy`, `scipy`, `librosa`, `matplotlib`, `transformers`
(for the Whisper-finetune modules). Raw capture audio (`.flac`/`.wav`/
`.bin`) is not included due to size; scripts expect a data directory laid
out as described in each module's own `README.md` / `config.py`. SLURM
scripts use placeholders (`<your_allocation>`, `<cluster_name>`,
`<REPO_ROOT>`) — fill in your own cluster's values before submitting.

## License

No license has been specified yet. All rights reserved pending manuscript
decision; contact the authors (post-review) for reuse terms.
