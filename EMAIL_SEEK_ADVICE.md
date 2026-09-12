Subject: Advice on the powerline side-channel project

Hi Professor [Name],

On the powerline audio side-channel work, we can now recover spoken digits and commands from
the power line for unseen speakers (~81% per digit, ~42% chance of breaking a 4-digit PIN),
while open-vocabulary speech stays a provable wall. I think it's a full paper for a top
security venue, and I'd like your read on the plan below.

Do you agree it's worth writing up? If so, here's how I'd strengthen it into a submission:

  - Generalization data. Collect from more audio sources and devices — spoken-digit/command
    corpora (e.g. FSDD, Google Speech Commands, TIDIGITS) played through a soundbar, TV,
    smart speaker, and laptop; across different outlets, distances, and with realistic
    competing loads on the circuit (fridge, dimmers, chargers). This device/robustness
    matrix is what reviewers will demand.
  - Threat-model realism. Demonstrate a believable attacker vantage point — a malicious USB
    charger or smart plug on the same strip, and ideally a cross-circuit / next-room tap on
    shared building wiring. Through-the-wall PIN recovery would be the headline.
  - Close the honest number. Re-train our shared front-end on training speakers only to
    remove a residual speaker-leak, and add a validation split for clean model selection.
  - Defenses + capacity. A short defenses section (line filtering, PSU regulation, ferrites)
    and an information-theoretic framing (per-digit accuracy, PIN-space reduction, ROC).

On venue, I'd target the top systems-security conferences rather than an ML venue —
IEEE S&P (Oakland), USENIX Security, ACM CCS, or NDSS — since the contribution is a new
side channel plus a practical attack. These run on rolling / multi-deadline cycles, so we
could aim at whichever deadline best fits how much of the generalization data we can gather
in time; I'd value your view on which to target and how it positions against prior work
(e.g. Glowworm and the sensor-based speech-eavesdropping line).

Could we grab 20 minutes to discuss? I'm happy to send a one-page summary first.

Thank you,
user
