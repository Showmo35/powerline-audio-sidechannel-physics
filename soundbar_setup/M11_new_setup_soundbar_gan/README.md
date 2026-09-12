# setup_soundbar_gan — conditional generative front-end (setup variant)

Sibling variant of `../setup_soundbar`. **Same soundbar data; generative method.**
A HiFi-GAN-style generator maps the powerline per-harmonic mel stack → clean
16 kHz **waveform**, trained with adversarial + feature-matching + mel + content
losses. Built to test whether a *generative* objective with a *faithfulness*
(content) loss recovers words where L1 regression (the U-Net variant) blurred.

## Pipeline

```
powerline .bin → 16 kHz per-harmonic mel STACK [16,80,T] ─► HiFi-GAN G ─► 16 kHz wave
                                                                 ▲  L_adv (MPD+MSD)
clean .wav (16 kHz) ─────────────────────────────────────────────┤  + L_fm
                                                                  ├  + L_mel ×45
                          frozen wav2vec2 (ASR) feature match ────┘  + L_content ×15
eval: Whisper WER on generated audio (held-out chunks 41-46)
```

| file | role |
|------|------|
| `config.py` / `data_io.py` | config + windowed readers (16 kHz) |
| `frontend_stack16.py` | powerline → 16 kHz per-harmonic mel stack |
| `dataset_gan.py` / `build_stack.slurm` | per-chunk stacks (13 GB), CPU array |
| `models.py` | HiFi-GAN Generator (5.8M) + MPD + MSD |
| `train_gan.py` / `run_train_gan.slurm` | GAN training + Whisper-WER eval |

## Result (judged by WER, as intended)

**Best WER 99.8% · final 100.1%** (30 epochs, 10.8k steps, 3 h, V100).

WER trajectory tells the story: steps 800–6400 → 100% with **empty** transcripts
(Whisper hears non-speech); from step ~7200 the content+adversarial losses push
the output to speech-like audio Whisper *attempts* — but the words are **wrong**,
and at step 8800 WER hit 102.9% from **hallucinated insertions**. It became more
*plausible*, never more *correct*.

## Verdict (now confirmed across 3 architectures)

| approach | best WER |
|---|---|
| fine-tuned Whisper (fixed front-end) | ~103% |
| learned U-Net enhancement (4× better features) | 100% |
| **conditional HiFi-GAN + content loss** | **99.8%** |
| clean-audio ceiling | ~4% |

A regression model, a generative model *with* an ASR-faithfulness loss, and a
fine-tuned ASR all pin at ~100%. The word-level phonetic information is **not in
the 200 kSps soundbar AC-cord capture**. The remaining lever is the **capture**
(SNR / bandwidth / carrier band — e.g. a Class-D switching carrier above the
100 kHz window), not the model. See the base module's Step-6 capture-audit plan.
