---
name: project-phonetic-bandwidth-2khz
description: "AC-cord channel has ~2 kHz phonetic bandwidth — coarse sub-2kHz formant/voicing detail IS recoverable, but fricatives (>2kHz) and all switching carriers are not"
metadata: 
  node_type: memory
  type: project
  originSessionId: 775c1bce-06a0-4213-8128-23a31d769278
---

Direct word-detection experiment (2026-07-17) quantifying HOW MUCH phonetic detail the AC-cord channel carries. Built `make_word_stimulus.py` (formant-synth "which" /w ɪ tʃ/ vs "this" /ð ɪ s/, 30 each, RMS-matched so loudness carries no class info; discriminating cue planted SPECTRALLY: which→1.5-4kHz affricate, this→4-9kHz sibilant). Captured live on the USRP @192.168.10.4 (4 MSps, 56 s, gain 1.0) while playing through the HDMI/soundbar sink → `word_capture.bin` (896 MB). Analysis: `phonetic_probe.py` + fricative-band ratio test. Capture validated: broadband power envelope tracks stimulus r=0.38 @ lag 0.194 s.

**Result — phonetic bandwidth ≈ 2 kHz:**
- **Switching carriers** (1460/880/309/1230 kHz): NO phonetic detail. 0 Hz tracked bandwidth; spectral-shape which/this classification at chance (p>0.2); fricative-band ratio AUC≈0.48 (chance). Consistent with [[project-wideband-carrier-negative]] — carriers are dead for audio, period.
- **Direct baseband (<2 kHz)**: phonetic detail PRESENT and genuinely spectral (not envelope). Spectral-shape-only which/this classification = **0.97 balanced acc, p=0.002** (LDA+5-fold CV, 400-perm null), while loudness-envelope-timing-only sits at chance (0.38). Per-band corr(demod, true audio) across 60 tokens stays above the shuffle null up to ~2.1 kHz.
- **Direct baseband (>2 kHz, the fricatives)**: NOT recovered. /tʃ/(1.5-4k) vs /s/(4-9k) band-ratio AUC=0.25 (n.s., p=0.12); both word classes look identical in the fricative bands (meanD≈-3.57 each). Oracle on the clean stimulus = AUC 1.000, so the cue is real — the channel just drops it.

**Interpretation:** the 97% spectral classification rides ENTIRELY on the sub-2kHz region (the /w/-vs-/ð/ onset + F2 motion), NOT the fricatives. So the channel preserves coarse low-formant/voicing/onset structure — enough to separate two acoustically-dissimilar words by spectrum — but loses high-formant and all fricative/sibilant detail (place-of-articulation cues). This refines, not contradicts, [[project-powerline-no-formants]]: fine formant reconstruction for intelligibility is still out; a thin ~2kHz slice of coarse phonetic detail is in, good only for limited-vocabulary spotting of dissimilar words.

**Method lesson:** "can we classify the two words" is the WRONG question — envelope/timing alone can win it. To test phonetics, split features into SPECTRAL (per-frame-energy-normalized mel, loudness removed) vs TEMPORAL (1-D loudness shape) and require the spectral split to beat a permutation null; and test the planted band-specific cue directly with a band-energy-ratio discriminant vs an oracle on the clean stimulus.
