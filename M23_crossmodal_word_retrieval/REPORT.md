# M23 — Cross-Modal Query-by-Example Word Spotting from Powerline Emanations

**Author:** Anonymous (double-blind review)
**Date:** 2026-07-06
**Status:** Design + analysis (pre-experiment). Decides whether the "generate-and-retrieve" idea overturns the capture wall or characterizes the envelope channel.

---

## 0. TL;DR

Every attempt so far has been **generate-and-recognize**: powerline → mel → decode as if it were clean speech. That imposes a punishing bar — the reconstruction must be a faithful, *decodable* speech signal — and it fails at ≈100% WER because the phonetic content is physically absent (see M16, M19, M20, CHIRP verdict).

The new idea is **generate-and-retrieve**: enroll a labeled gallery of *real* word-mels ("all the before-mels"), and at test time return the nearest neighbor of the powerline-generated mel. We never ask the mel to be intelligible — only that a generated `before` land closer to real `before`s than to anything else. The observed correlations support this in principle:

- gen-`before` ↔ real-`before` ≈ **0.54**
- gen-`before` ↔ any-other-word ≈ **0.09** (≈ chance)

A 0.54-vs-0.09 gap is exactly what a nearest-neighbor classifier feeds on. **Retrieval tolerates a degraded reconstruction that ASR cannot.** This is correct in principle.

**But** — and this is the whole report — given M14/M20/CHIRP, that 0.54 is almost certainly the **energy envelope / prosody channel re-badged**, not recovered phonetics. Retrieval doesn't punch through the wall; it lowers the bar to *exactly the height of the channel already known to be open*. That defines both why it will work and where it stops.

The scientific value of this module is therefore **not the hit rate** — it's the **control**. The single experiment that matters is envelope-null retrieval (Condition B below). Run it first.

---

## 1. Why the insight is correct

Two tasks load on different information:

| Task | What it requires | Where that information lives |
|---|---|---|
| **Recognition (ASR)** | reconstruction invertible to text — every phone survives | fine spectral structure, formants, 300–3400 Hz |
| **Retrieval (kNN)** | reconstruction merely *discriminative* — stable per-word signature | *any* consistent class-separating feature |

ASR needs phonetics. Nearest-neighbor needs only a stable signature — and duration, syllable rhythm, energy contour, and onset shape are a perfectly good signature for a handful of words. So a 0.54-vs-0.09 gap absolutely feeds a kNN classifier. The framing is sound: **retrieval < recognition as an information bar.**

---

## 2. What the 0.54 is actually made of

This is where the project's own accumulated evidence must be applied honestly rather than overridden by an exciting number.

Every prior module converged on one fact — **the powerline channel carries the envelope/prosody, not the phonetics:**

- **M14 (PowerLine-Flow mel-gen):** env_r 0.87 but generated-audio WER ≈ 100% (Griffin-Lim ceiling 1.4%). The envelope reconstructs; the content does not.
- **M16 (ASR on PLF mels):** real-mel control 44.6% WER vs PLF-mel twin ≈ 100% WER. Content **absent**, not merely degraded.
- **M19 (phonetic probe):** direct regression of envelope-removed mel from wide-STFT powerline → no phonetic signal.
- **M20 (closed-set word classifier):** 200 kHz macro-F1 **31%** *and* envelope-only **36%** (chance 3.3%). Harmonics added **nothing** over the envelope. The separability is **prosody/duration**, and the real-envelope channel is genuinely positive.
- **CHIRP verdict (definitive):** controlled chirp/tone on 3 devices → coherence **0.004**, no HF carrier, mains-comb only. Content **physically absent**; the PSU averages it out.

Conclusion: two real `before` mels correlate at 0.54 because they share duration, rhythm, energy contour, and onset — and the generator reproduces exactly those. **The 0.54 is the M14 envelope (r≈0.87) propagated into correlation space, not returned phonetics.** Retrieval works precisely because it only needs the channel that was already open.

> Retrieval didn't lower the bar past the wall. It lowered the bar to the exact height of the wall's one opening — the envelope.

That is not a criticism; it is the reason it will work. But it is also the ceiling.

---

## 3. The experiment that decides everything

Run retrieval **three times, identically**, changing only the feature. The entire scientific content lives in two deltas.

| Cond. | Feature fed to kNN | Isolates |
|---|---|---|
| **A. Full-mel** | generated mel, as-is | prosody **+** any phonetics |
| **B. Envelope-null** | mel with per-frame energy divided out (each frame L2- or energy-normalized), keeping only *spectral shape* | phonetics **alone** |
| **C. Envelope-only** | 1-D energy contour + duration only, no spectrum | prosody **alone** |

Decision rules:

- **A − C > 0** → the spectrum contributed something beyond prosody.
- **B ≫ chance** → that something is **phonetic**. *This would overturn the CHIRP verdict.*
- **A ≈ C and B ≈ chance** → "cross-modal word spotting" is real but is **prosodic word spotting**; the mel/vocoder stack is decoration around a 1-D envelope. (This is what M20 predicts.)

**B is the headline test. Run it first.** It is the only outcome that could change the project's verdict; everything else quantifies a channel already known to exist.

**Prior (honest):** given M14/M20/CHIRP, expect **A ≈ C ≫ B ≈ chance**. In that case M23 becomes the *definitive characterization* of the envelope channel, not a refutation of the wall — a stronger, cleaner result than either "we recovered speech" (unsupportable) or a bare negative.

---

## 4. The scaling problem — the 0.09 is a mean, and the mean lies

kNN accuracy is **not** governed by the average between-class similarity (0.09). It is governed by the **tail** — the single nearest wrong-class gallery item — and that tail is set by *prosodic neighbors*:

- `before` vs `again` vs `under` — 2 syllables, similar stress, similar duration — correlate not at 0.09 but at ~0.3–0.4.
- With a vocabulary of N words, retrieval takes a **max over N−1 confusers**. The expected maximum of the between-class similarity distribution rises with N and, at vocabulary scale, crosses 0.54.

Concrete prediction: accuracy is high on a small, prosodically-spread gallery (the current 0.54-vs-0.09 regime) and **degrades toward the collision point as prosodic-neighborhood density grows** — not gracefully, but toward chance. This is the classic query-by-example scaling failure, and it maps one-to-one onto "prosody separates a few words; phonetics would be needed to separate many."

**Reporting rule:** never report a single accuracy number. Report:
1. accuracy vs gallery size N (learning/collision curve),
2. accuracy vs a prosodic-diversity control (words matched vs mismatched on syllable count + duration),
3. the confusion matrix — collisions should cluster by prosody, not by phoneme overlap. That clustering *is* the evidence.

---

## 5. Framing for publication

Do **not** frame this as "we recovered speech via retrieval." Frame it as the capstone that *quantifies* the envelope channel and closes the paper:

> **Cross-modal query-by-example word spotting from powerline emanations.**
> Powerline emanations do not carry decodable speech (CHIRP: coherence 0.004; ASR WER ≈ 100%), but they carry a stable prosodic signature sufficient for nearest-neighbor word retrieval against a real-mel gallery. We show retrieval accuracy is (a) driven by the energy envelope — envelope-only and full-mel retrieval are statistically indistinguishable — and (b) collapses toward chance as vocabulary and prosodic-neighborhood density grow. This yields an information-theoretic characterization of the side channel: **prosody-limited, phonetics-absent, vocabulary-bounded.**

This is a stronger paper than either extreme. In the emanations/side-channel security literature, a *bounded* side channel with a measured capability envelope and a measured failure edge is exactly the valued contribution.

---

## 6. Concrete protocol

**Data.** Reuse the M14/M20 aligned corpus: powerline-generated mels (test) + real word-mels (gallery). Enforce the **M17 split fix** — no speaker/utterance leakage between gallery and query; drop duplicates.

**Gallery.** For each class, enroll K real-mel exemplars (sweep K ∈ {1, 3, 5, all}).

**Query.** Each powerline-generated mel → feature transform (A/B/C) → similarity to every gallery item.

**Similarity.** Cosine on flattened, time-aligned/length-normalized mels; also try DTW-cosine to remove the duration cue in an ablation — if accuracy drops sharply under DTW, that is direct proof the cue is duration.

**Classifier.** k-NN (k ∈ {1,3,5}), majority/soft vote. Report top-1 and top-5.

**Controls (mandatory).**
- B (envelope-null) and C (envelope-only) as in §3.
- DTW ablation (removes duration).
- Prosody-matched vs prosody-mismatched gallery subsets (§4).
- Chance line = 1/N and a permutation test on labels.

**Metrics.** top-1 / top-5 accuracy, macro-F1, mAP; all as curves vs N and vs prosodic diversity; confusion matrix with prosodic vs phonetic collision annotation.

**Compute.** `tf_gpu` env; heavy runs via SLURM GPU (nextgen A100). No ffmpeg in-pipeline.

---

## 7. Go / no-go

| Outcome | Interpretation | Action |
|---|---|---|
| **B ≫ chance** | phonetics present after all | **major** — re-open recovery; revisit CHIRP |
| **A > C, B ≈ chance** | spectral *shape* helps but non-phonetically (timbre/formant-envelope, still prosodic) | interesting nuance; report the A−C margin |
| **A ≈ C ≫ chance** (expected) | pure prosodic retrieval | headline the channel characterization + scaling collapse |
| **A ≈ C ≈ chance** | even prosody doesn't survive generation | retrieval adds nothing over M20; stop |

**Bottom line:** the idea is correct and worth running, but its value is the control, not the hit. Run B first — it is the only result that can change the verdict. Then report A/N and C as the scaling curve and baseline. My strong prior is A ≈ C ≫ B ≈ chance, which turns M23 into the definitive characterization of the envelope channel: **prosody-limited, phonetics-absent, vocabulary-bounded.**

---

## Appendix — related modules
M14 (mel-gen, env_r 0.87) · M16 (ASR-on-PLF-mels, content absent) · M19 (phonetic probe) · M20 (word classifier, prosody-driven) · M17 (split-leakage fix) · CHIRP verdict (coherence 0.004, content physically absent).
