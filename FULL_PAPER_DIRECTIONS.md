# Powerline Acoustic Side-Channel — Directions for a Full Top-Tier Paper

*Target venues: IEEE S&P (Oakland), USENIX Security, ACM CCS, NDSS.*
*Companion to `PAPER_DIRECTION.md` (the thesis) — this doc is the execution plan:
what a complete, submission-ready paper needs, what is DONE, what is UNTESTED, and
the module-by-module path to fill the gaps.*

*Written 2026-07-13. Convert relative dates before editing.*

---

## 0. One-paragraph state of the project

We have a **decisive, over-determined negative result**: open-vocabulary speech
content is physically absent from the conducted power line; only the loudness/prosody
envelope survives, in the 0–16 kHz mains-sideband band. This is confirmed by ~10
independent methods (end-to-end ASR, phonetic probe, retrieval, context-decoder,
word-pair physics, dual-band) and by controlled chirp/tone physics. **We do NOT yet
have the positive result the paper's thesis promises** — constrained-vocabulary
(digit / command) recovery has never actually been run, and the one closed-set
"positive" (M20/M21, 31–39%) has never been tested speaker-disjoint. A top-tier paper
needs the positive attack + robustness numbers + threat-model realism. Everything
below is oriented to closing that gap without re-litigating the (settled) wall.

---

## 1. Load-bearing assumptions that are STILL UNTESTED

These are the cracks a reviewer will find. Fix them before writing.

| # | Assumption the thesis rests on | Status | Risk if wrong |
|---|--------------------------------|--------|---------------|
| A1 | Constrained vocab (digits/commands) is recoverable | **NEVER RUN.** All positives are LibriSpeech *function words* — the worst case. | No positive result → paper is a negative-results note, not S&P. |
| A2 | The closed-set signal is word identity, not speaker identity | **NEVER tested speaker-disjoint.** All splits are chunk/time-disjoint. | If speaker-dependent, the one positive collapses; desk-reject risk. |
| A3 | The channel is device/setup-general | Only ONE soundbar, ONE session, ONE outlet characterized end-to-end. | "Works on one gadget" is not a channel. |
| A4 | A realistic attacker vantage point exists | Only bench capture on the device's own line. No charger / smart-plug / cross-circuit demo. | Reviewers reward the through-wall demo; without it the threat is theoretical. |
| A5 | Result survives real-world electrical noise | Captures are clean/isolated. No competing loads (fridge, dimmer, chargers). | Robustness is the crux reviewers attack. |

**Guiding rule:** the next modules exist to convert A1–A5 from assumption to measured
number. Do NOT spend more compute re-proving the open-vocab wall — it is the
credibility anchor and it is done.

---

## 2. The paper's spine (what every section must deliver)

1. **A new channel + why it exists.** Speaker current draw → PSU/rectifier → conducted
   emissions; envelope survives, phonetics averaged out by bulk caps. CHIRP/tone
   controlled experiments are the crown jewel. *(DONE — `Capture_Analysis/`, CHIRP.)*
2. **An honest capacity bound.** Open-vocab is infeasible; quantified and physically
   explained; LM-rescue fails honestly (bootstrap). *(DONE — M16–M27, M24.)*
3. **A real recovered secret.** Constrained-vocab digits/commands at usable accuracy →
   PIN-space collapse. *(NOT DONE — this is the headline gap.)*
4. **Robustness the threat model demands.** Speaker-independent, device-general,
   noise-tolerant, realistic vantage point. *(NOT DONE — A2–A5.)*
5. **Defenses.** Line filtering/conditioning, PSU regulation, ferrite/isolation,
   dithering; measure attack degradation under each. *(NOT DONE.)*

A submission needs all five. We have 1–2 airtight; 3–5 are the work ahead.

---

## 3. Directions, ranked by paper-impact-per-effort

### Direction I — The constrained-vocab attack (THE headline; do first)
The single experiment that decides whether a paper exists.

- **I-a (cheap, existing data, de-risk in a day):** mine `word_index.json` for the most
  acoustically *distinct* frequent words (max phonetic/duration spread — NOT
  the/that/which), run the M20 supervised classifier on that subset. A lift off the
  function-word floor is the go/no-go signal for the real capture. *(Proxy only —
  LibriSpeech has few spoken digits.)*
- **I-b (the real result, needs capture):** play a spoken-digit / command corpus
  (FSDD, TIDIGITS, Google Speech Commands) through the device, capture on the line,
  train a supervised classifier + a CTC-over-envelope digit-*sequence* model.
- **Deliverables:** per-class accuracy, confusion matrix, sequence-level PIN accuracy,
  **bits/utterance**, and **PIN-space reduction** (10,000 → k). These numbers are the
  paper.
- Evidence it may work: M26 K30 GEN-sup 0.41 (12× chance), demod which/this 0.71
  cross-recording — and digits are *far* more acoustically/prosodically separable than
  function words.

### Direction II — Speaker-independent split (make-or-break; runs alongside I)
- Train on one set of corpus speakers, test on held-out speakers. Report the gap vs the
  chunk-split number. Cheap to add to any classifier already written.
- If the closed-set signal survives speaker-disjoint → A2 confirmed, strong result.
  If it collapses → we learned the "positive" was speaker prosody; pivot to
  presence/count/speaker-ID framing (still publishable, different claim).

### Direction III — Threat-model realism / vantage point (the headline escalation)
Demonstrate at least one; (b) is Distinguished-Paper-caliber.
- (a) Malicious USB charger / smart plug on the same power strip — most realistic,
  IoT supply-chain.
- (b) **Cross-circuit / neighboring-room tap on shared building wiring** — through-a-wall
  recovery on the same phase is the headline figure.
- (c) Compromised smart meter / breaker-panel tap.
- No physical access to the victim device in any case.

### Direction IV — Generalization & robustness (the crux reviewers attack)
- Devices: soundbar → TV → smart speaker → laptop.
- Outlets, distance along the wiring, wire gauge / panel topology.
- Concurrent electrical noise: fridge compressor, dimmers, other chargers, LED drivers
  on the same circuit. Report accuracy vs SNR / vs number of competing loads.

### Direction V — Defenses (a section, not a footnote)
- Passive: power-line filter / EMI conditioner, ferrite chokes, isolation transformer.
- Active: PSU regulation quality, output dithering / spread-spectrum switching.
- Measure attack accuracy degradation under each → gives the paper a constructive close.

### Direction VI — Methodological systematization (secondary contribution)
- The **train-a-generator-to-be-retrieval-optimal-from-a-lossy-physical-channel**
  method (M22: InfoNCE + correlation on flow-matching, open vocab) generalizes to other
  degraded side channels. A depth/"systematization" note; not load-bearing alone.

---

## 4. Suggested module plan (M28+)

- **M28 — Constrained-vocab classifier (Direction I + II).**
  - Phase 1: I-a proxy on distinct LibriSpeech words + speaker-disjoint split. Go/no-go.
  - Phase 2 (if capture available): digit/command corpus through the device; supervised
    classifier + CTC sequence model; PIN-space-reduction headline number.
- **M29 — Device & noise generalization (Direction IV).** Multi-device capture matrix,
  competing-load ablations, accuracy-vs-SNR curves.
- **M30 — Vantage-point demo (Direction III).** Malicious-charger bench build; attempt
  cross-outlet / through-wall capture on shared wiring.
- **M31 — Defenses (Direction V).** Filter / ferrite / isolation ablations; attack
  degradation table.
- **M32 — Paper assembly.** Figures, capacity tables, positioning, artifact release.

Order rationale: M28 decides if there's a paper; do it before spending capture effort
on M29–M31.

---

## 5. Positioning vs prior art (nail the delta or risk desk-reject)

- **Glowworm (CCS'21)** — audio from device power **LED** (optical, line-of-sight).
  *Delta: conducted power line, no line-of-sight, through walls on shared wiring.*
- **Gyrophone / Spearphone / AccEar** — on-device motion sensors. *Delta: no code on the
  victim device; purely electrical tap.*
- **PowerHammer / EM keystroke leakage** — power line as channel, but for exfil /
  keystrokes, not co-located **audio content**.
- **Hard literature sweep required** for any prior *audio-onto-AC-line* leakage. If it
  exists, sharpen novelty via (i) capacity characterization, (ii) generate-and-retrieve
  method, (iii) practical constrained-vocab secret recovery.

---

## 6. Evidence already in hand (repo pointers)

- Channel physics / mechanism: `Capture_Analysis/`, `test_new_setup/` (CHIRP + tones,
  wideband DC→3.85 MHz carrier hunt — all envelope-only).
- Open-vocab wall: `M16`–`M18` (~100% WER), `M19` (PHON_r ≈ 0), `M22`/`M23` (retrieval
  top-1 3.8%, oracle 0.108), `M24` (LM-rescue collapses honestly), `M26` (frication
  physically absent), `M27` (dual-band adds zero).
- Rhyme-collision failure figures: `M22_retrieval_generator/outputs/m22_wrong_grid.png`,
  `m22_corr_stats.png`.
- Closed-set prosody baseline (to be made speaker-disjoint): `M20`, `M21`.
- Thesis / framing: `PAPER_DIRECTION.md`.

---

## 7. What NOT to do

- Do **not** try to rescue open-vocab transcription — proven a wall across the entire
  accessible spectrum and both coupling paths.
- Do **not** add another ~100% WER iteration or another generator variant on function
  words — diminishing returns; the wall is settled.
- Do **not** report any closed-set "positive" without the speaker-disjoint number
  beside it — an unqualified chunk-split accuracy is a reviewer trap.
