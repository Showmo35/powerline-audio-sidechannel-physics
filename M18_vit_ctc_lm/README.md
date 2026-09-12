# M18_vit_ctc_lm — ViT reads the mel, LM assembles the words (null-subtracted)

The last test. M16 showed a small conv-CTC on M14's generated mels stays ~100 %
WER while its real-mel twin reaches 44.6 %. M18 asks whether a **stronger reader**
(a from-scratch ViT over the mel *image*) plus a **language-model back-end** can
recover words M16's tiny reader missed — built so it **cannot fool us**.

## Why the null control is the whole point

An LM on a signal that carries no words does not recover them, it **invents**
them fluently; and a pretrained LLM has memorised LibriSpeech's public-domain
texts, so it can *recite* the answer from almost nothing. So every condition is
run through the identical ViT + LM pipeline, and we read **differences**, not
absolutes:

| condition | input | role |
|---|---|---|
| **real** | reference mels | ceiling (pipeline works) |
| **null** | real mels, per-frame frequency **shuffle** | hallucination floor — energy envelope preserved *exactly*, phonetics destroyed |
| **gen** | M14-generated mels (reused from `M16/gen_mels`) | the reconstruction under test |

**Read:** `WER(gen) ≈ WER(null) ≫ WER(real)` → the LM only hallucinates; the
capture-wall verdict (M13–M17) is final. `WER(gen) ≪ WER(null)` → real residual
signal beyond the envelope — pursue.

## Components

- `models.py` — **ViTCTC**: Conv2d patchify the [80,T] mel (freq 16 × time 4) →
  separable 2-D pos-emb → 12-layer pre-norm ViT → mean-pool freq → CTC head
  (38 M params, from scratch; no ImageNet — spectrograms aren't natural images).
- `dataset.py` — gen / real / **null** sources; null = per-frame bin permutation
  (verified: per-frame energy identical to real → env floor). Same chunk split
  as M15/M16 **plus** utterance-level disjointness (drop train rows sharing an
  utt_id/text with any test chunk).
- `train.py` — CTC trainer, SpecAugment, greedy WER/CER each epoch, resume.
- `lm.py` + `train_lm.py` — from-scratch **CharLM** (trained only on train-split
  text) + CTC prefix beam search with shallow fusion.
- `decode.py` — greedy / beam / **beam+LM** WER per condition (+ dumps nbest).
- `llm_agent.py` — exploratory arm: a pretrained instruct LLM reconstructs the
  sentence from the noisy hyp; null-subtracted so memorisation is exposed.
- `summarize.py` — one table, conditions × decoders, with gen−null deltas.

## Jobs (nextgen A100)

```
6200638 vit real ─┐
6200639 vit null ─┼─ afterok ─► 6200662 analysis
6200640 vit gen  ─┘   (train_lm → decode ×3 → llm_agent ×3 → summarize)
```

## Run manually
```bash
cd "<REPO_ROOT>/M18_vit_ctc_lm"
sbatch run_train.slurm --input real --epochs 40 --batch 16 --out-dir outputs/vit_real
# ... null, gen ...
sbatch run_analysis.slurm     # after all three finish
```
