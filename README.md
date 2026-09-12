# Powerline Acoustic Side-Channel — Physics & Capacity Bound

Channel-physics characterization and the open-vocabulary capacity bound for
a power-line conducted-audio side channel: a device drawing power from an
AC outlet leaks the audio it plays onto the electrical line it's plugged
into. Target venues: IEEE S&P, USENIX Security, ACM CCS, NDSS.

Companion repo: `powerline-audio-sidechannel-attack` — the positive-result
half (constrained-vocabulary digit/command recovery) that this negative
result and physical explanation motivate. See `docs/PAPER_DIRECTION.md` and
`docs/FULL_PAPER_DIRECTIONS.md` for the full paper thesis and execution plan.

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

## Repository layout

```
physics/           Channel-physics tooling: Capture_Analysis/ (chirp/tone
                    reconstruction, cross-device alignment, mains-sideband
                    analysis) plus the root-level tone/spectrogram scripts.

baselines/          Early ASR-pipeline iterations that established the
                    initial "content is missing" observation before it was
                    rigorously confirmed: Module3-Module9, UNet, UNet_Old,
                    FullSubNetPlus, Cascade, E2E, Experiments,
                    Training_Scripts, and their run outputs
                    (model_runs/, output/, test_new_setup/).

soundbar_setup/     M10-M15: soundbar capture setup iterations (raw ASR,
                    GAN/learned front-ends, mel generation).

capacity_bound/     M16-M27, M28_mains_comb, M31: the capacity-bound
                    investigation. ASR-on-generated-mels, full-resolution
                    spectrogram generation, a ViT+CTC+LM pipeline, a direct
                    phonetic-content probe, a closed-set word classifier,
                    open-vocabulary cross-modal retrieval (including
                    multi-device monitor/TV validation), a
                    context/language-model rescue attempt (tested to its
                    honest conclusion — it fails), word-pair physics,
                    dual-band analysis, and the mains-harmonic-comb
                    hypothesis (also rejected). Each module is an
                    independent line of evidence for the same conclusion.

data/               Capture pipeline + dataset manifests (Powerline_Data_
                    Captures contents). Raw audio is not included; see
                    data/readme.md.

notebooks/          Exploratory and evaluation notebooks.

docs/               PAPER_DIRECTION.md, FULL_PAPER_DIRECTIONS.md — the
                    paper thesis and execution plan.
```

## Note on scope

`M28_constrained_vocab/`, `M29_constrained_supervised/`, and
`M30_retrieval_constrained/` — the constrained-vocabulary attack — live in
the companion `powerline-audio-sidechannel-attack` repo, since they answer
a different question (can we recover a *specific kind* of secret) than
this repo (what does the channel physically carry, and what's the honest
ceiling on open speech).

## Setup

Most modules under `soundbar_setup/` and `capacity_bound/` are standalone:
a `config.py`, a training/eval script, and a SLURM (`*.slurm`/`*.sh`)
submit script. Typical dependencies across the project: `torch`, `numpy`,
`scipy`, `librosa`, `matplotlib`, `transformers` (for the Whisper-finetune
modules in `baselines/`). Raw capture audio (`.flac`/`.wav`/`.bin`) is not
included due to size; scripts expect a data directory laid out as
described in each module's own `README.md` / `config.py`. SLURM scripts
use placeholders (`<your_allocation>`, `<cluster_name>`, `<REPO_ROOT>`) —
fill in your own cluster's values before submitting.

## License

No license has been specified yet. All rights reserved pending manuscript
decision; contact the authors (post-review) for reuse terms.
