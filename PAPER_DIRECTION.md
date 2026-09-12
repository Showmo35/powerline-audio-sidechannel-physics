# Powerline Acoustic Side-Channel — Top-Tier Paper Direction

*Target venues: IEEE S&P (Oakland), USENIX Security, ACM CCS, NDSS.*
*(Not the ML conferences — this is a novel side-channel attack.)*

---

## Thesis (the one sentence that carries the paper)

> A device drawing power from an AC outlet leaks the audio it plays onto the
> **conducted power line**. An attacker on the same electrical circuit — via a
> malicious charger, a smart plug, or a tap one room / apartment away — **cannot
> transcribe open speech**, but **can reliably recover the short, constrained
> utterances that matter for security**: spoken PINs / OTPs, phone-banking digits,
> and voice-assistant commands.

Three things top reviewers reward, all present: a **new channel**, a **real
secret**, and an **honest capacity bound**.

---

## Why the constrained-vocabulary limit is a FEATURE, not a weakness

The "prosody-only / no open-vocab ASR" result is scientifically valuable and does
**not** reduce the threat, because the secrets worth stealing are already short,
closed-vocabulary utterances:

- **Digit strings** — PINs, credit-card numbers, SSNs, 2FA/OTP codes read aloud,
  IVR banking ("say your account number").
- **Wake-word + command grammar** — "Alexa, unlock the front door / disarm the
  alarm / read my last message."
- **Yes/no, hot-word presence, speaker presence/count.**

A 4-digit PIN recovered even at modest per-digit accuracy collapses the brute-force
space from 10,000 to a handful — quantifiable, headline-grade impact.

Our own evidence that open-vocab is a wall (keep it — it is the credibility anchor):
- M22/M23 open-vocab retrieval top-1 ≈ **3.8%** (25k words).
- The rhyme-collision failure grid (`M22_retrieval_generator/outputs/m22_wrong_grid.png`):
  generated mels match near-rhymes ("no"→"now", "were"→"door", "why"→"law") because
  the channel carries **envelope/prosody** (loudness + duration), shared across many
  words, not the fine phonetics that separate them.
- CHIRP verdict: controlled tones/chirps on 3 devices → no HF carrier, mains-comb
  only, content averaged out by the PSU. The channel is physically prosody-limited.
- **The context/language-model rescue fails honestly (M24).** A natural objection is
  "a language model should disambiguate rhymes from sentence context." We tested this
  to its conclusion. With TEACHER-FORCED ground-truth context, an LM rescorer looks
  strong (content-recall 0.12, ~2× a naive attack) — but that is an evaluation
  artifact. In the HONEST attack (joint decode, the surrounding words PREDICTED not
  given), recall collapses to **0.013 — below channel-only (0.020)**. Root cause: a
  **mutual-context bootstrap problem** — disambiguating content word A needs content
  words B, C, which are themselves unknown and equally unrecoverable, so there is no
  fixed point. Context can only help when it is given, and in a real attack it never
  is. Figure: `M24_context_decoder/outputs/m24v3_genM22_big.png`. This is *why*
  constrained vocab is the answer: digits/commands need NO context (each token is
  independent), sidestepping the bootstrap entirely.
  - **Confirmed with a proper JOINT BEAM SEARCH (M24 v4), not just greedy decoding.**
    A real beam search over candidate combinations + full-sentence bidirectional
    rescoring + a λ sweep still **exactly ties channel-only (0.020)** — adding the
    language model never beats raw retrieval; it only produces fluent *fabrications*
    ("EACH IDEA DEVELOPS A COUNTLESS PROGENY" → "the opera thinks the fifteenth
    cossacks": grammar recovered, meaning invented). Two independent decoders (ICM
    and beam) agree. Root limits, quantified: the true word is a candidate only ~23%
    of the time, and the channel is too weak for the true sentence to be the joint
    optimum among equally-fluent alternatives. The recovery ceiling is thus a
    **channel-discriminability** limit (needs a better front-end, not a better LM),
    and constrained vocab avoids it because ~10 tokens have no fluent-alternative
    collisions. Figure: `M24_context_decoder/outputs/m24v4_genM22_big.png`.

---

## The structure that wins (S&P / USENIX template)

1. **Discover + physically explain the channel.** Speaker current draw → PSU /
   rectifier → conducted emissions; envelope survives, phonetics averaged out. Our
   CHIRP-style controlled experiments are the crown jewel — reviewers reward a paper
   that explains *why* the leak exists.
2. **Bound it honestly.** Open-vocab is infeasible (M22/M23 numbers + failure grid).
   The "we do not overclaim" section.
3. **Weaponize the part that works.** Constrained-vocab attack on digits / commands,
   with real accuracy → real secret recovery.
4. **Defenses.** Power-line filtering / conditioning, PSU regulation, ferrite /
   isolation, dithering.

---

## The crux reviewers will attack — prioritized experiments

1. **Threat model / vantage-point realism.** Demonstrate at least one strong one:
   - (a) malicious USB charger or smart plug on the same strip — *most realistic,
     IoT supply-chain*.
   - (b) **cross-circuit / neighboring-room tap on shared building wiring** — *the
     dramatic escalation; through-a-wall recovery on the same phase is the headline*.
   - (c) compromised smart meter / breaker panel.
   No physical access to the victim device.
2. **Speaker independence.** Train on some speakers, test on unseen ones. The single
   most important robustness number — if it is speaker-dependent only, reviewers hammer it.
3. **Generalization** across devices (soundbar → TV → smart speaker → laptop),
   outlets, distance, and concurrent electrical noise (fridge compressor, dimmers,
   other loads on the circuit).
4. **Live, security-relevant data — not LibriSpeech narration.** Move to spoken-digit
   / command corpora (FSDD, Google Speech Commands, TIDIGITS, or TTS + human digits)
   played through real devices and captured on the line.

---

## Positioning vs. prior art (nail the delta or risk desk-reject)

- **Glowworm (CCS'21)** — audio from a device's power **LED** (optical, needs
  line-of-sight). *Our delta: conducted power line, no line-of-sight, works through
  walls on shared wiring — stealthier and remote.*
- **Accelerometer / gyroscope speech eavesdropping** (Gyrophone, Spearphone, AccEar)
  — on-device motion sensors. *Our delta: no code on the victim device at all — a
  purely electrical tap.*
- **PowerHammer / power-line exfil, EM keystroke leakage** — power line as a channel,
  but for *exfil / keystrokes*, not co-located **audio content**.
- Do a hard literature sweep for any prior *audio-onto-AC-line* leakage; if it
  exists, sharpen novelty via (i) capacity characterization, (ii) the
  generate-and-retrieve recovery method, (iii) the practical constrained-vocab attack.

---

## Secondary (methodological) contribution

The **train-a-generator-to-be-retrieval-optimal-from-a-lossy-physical-channel**
approach (M22: InfoNCE + correlation on flow-matching, open vocab) is transferable to
other degraded side channels. Not enough to carry a security paper alone, but a nice
depth/"systematization" note.

---

## Concrete next steps for THIS project

1. **Swap the corpus** LibriSpeech → spoken digits + a command grammar. Re-run the
   M20-style closed-set classifier and/or a CTC-over-envelope digit-sequence model.
   The jump over open-vocab 3.8% *is* the paper's main result.
2. **Run the speaker-independent split first** — make-or-break.
3. **Build the "malicious charger" bench demo**; attempt the **cross-outlet /
   same-circuit (through-wall)** capture. Working PIN recovery through a wall is
   Distinguished-Paper-caliber.
4. **Report information-theoretic capacity** — bits/utterance, per-digit accuracy,
   PIN-space reduction, wake-word ROC. Numbers reviewers can't argue with.
5. **Keep** the CHIRP mechanism analysis and the open-vocab negative result +
   rhyme-collision failure figure as credibility anchors.

**Do not** try to rescue open-vocab transcription — the data shows it's a wall. The
winning paper is: *"conducted power-line audio side channel → practical recovery of
spoken secrets (PINs / commands) from a same-circuit tap, with a rigorous physical
characterization of why open speech stays private."*

---

## Evidence already in hand (repo pointers)

- Channel physics / mechanism: `Capture_Analysis/`, CHIRP experiments.
- Open-vocab wall: `M22_retrieval_generator/` (fresh open-vocab retrieval generator,
  top-1 3.8% / top-5 11.6%), stats & failure figures in
  `M22_retrieval_generator/outputs/` (`m22_corr_stats.png`, `m22_wrong_grid.png`,
  `m22_corr_grid.png`).
- Retrieval / attack pipeline: `M23_crossmodal_word_retrieval/` (open-vocab attack,
  oracle ceiling analysis; M14- vs M22-generator variants).
- Closed-set prosody classifier baseline: `M20_powerline_word_classifier/`.
