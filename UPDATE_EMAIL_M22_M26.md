Subject: Powerline side-channel — why open-vocabulary speech is a wall

Hi [Advisor],

A short update on the open-vocabulary side of the powerline project. The takeaway: we've
established rigorously that recovering full, free-form speech from the power line is a hard
wall — and, importantly, we can explain why. This is the negative result that anchors the
paper's credibility and motivates the constrained-vocabulary attack.

We approached it from several angles and they all agree. We trained a generator specifically
to make word recovery work — mapping the powerline signal to a spectrogram optimized for
retrieval against a library of real words — and while that beat a plain reconstruction
model, absolute recovery stayed very low across an open vocabulary. Retrieval works for a
small closed set of words but collapses as the vocabulary grows, and what it does recover
turns out to be prosody and loudness rather than the actual phonetic content.

We then tested the natural rescue — using sentence context and a language model to
disambiguate. It doesn't hold up honestly: it looks strong when you feed it the true
surrounding words, but in a real attack those words are themselves unrecoverable, so there's
nothing to bootstrap from. We also ruled out that the ceiling was just a weak model — a
stronger generator didn't beat it — which points to the channel, not the method, as the
limit.

Finally, we traced the cause to the physics. The acoustic detail that distinguishes
similar-sounding words (high-frequency frication energy) is simply not present in the
capture; the power supply averages the audio down to a slow loudness envelope before it ever
reaches the wire. The models aren't failing — the information was never there.

Bottom line: the power line leaks the loudness/prosody envelope but not the fine phonetics
needed for open-vocabulary speech, and we can show that from multiple independent directions.
That's exactly why the attack targets short, closed-vocabulary secrets like digits and
commands, where recovery does work (separate update).

Best,
user
