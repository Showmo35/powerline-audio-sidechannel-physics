# M22 — Open-Vocabulary Retrieval-Optimized Generator (fresh)

Train a powerline→mel generator **from scratch** whose *generated* word-mels are
**retrieval-optimal** for **every word** in the dataset: highest correlation with
the correct word's real mel and separated from other words, so kNN over a real-mel
gallery returns the correct word for **any** of the ~25k word types.

This is the generator that raises the retrieval ceiling for the M23 attack pipeline
(open-vocab attack found retrieval — not the LLM — is the bottleneck).

## Two things that make M22 different from a plain M14 generator
1. **Fresh training, no M14 checkpoint.** Weights are random-initialized; the DiT
   architecture (`models.py`) and IO readers (`data_io.py`) are vendored here so the
   module is self-contained. Log-mel mean/std are estimated from the data.
2. **Open vocabulary.** Trained on **all** word occurrences (24,990 word types,
   ~334k occurrences), not a 30-word closed set. Evaluated by open-vocab kNN over a
   gallery covering every word.

## Objective (two-phase, from scratch)
```
phase 1 (step < retr_start):  L = flow_matching(raw, real 4s mel)          # learn to generate
phase 2 (ramped in):          L = flow_matching
                                + w(step) * [ InfoNCE(gen → real, labels)   # separate words
                                            + (1 - corr(gen_i, real_i)) ]   # match target mel
```
- `w(step)` ramps 0→1 over `retr_ramp` steps starting at `retr_start` — the flow
  field must produce mels before backprop-through-sampling is meaningful.
- Retrieval terms live in the **kNN space**: time-norm → flatten → mean-center →
  L2-norm, so cosine == Pearson r.
- Generated word mel = few-step **differentiable** ODE rollout, sliced to the word.
- Open vocab ⇒ InfoNCE is effectively **instance-level** (retrieve your OWN real mel
  against the batch); same-word collisions add extra positives.

## Files (self-contained — no M14 dependency)
| file | role |
|------|------|
| `models.py` | PowerLine-Flow DiT architecture (vendored) |
| `data_io.py` | on-the-fly `.bin`/`.wav` window readers (vendored) |
| `config.py` | `PLFConfig` (generator/mel) + `RCFG` (retrieval, open-vocab) |
| `dataset_words.py` | one item = one word occurrence; `real_only` gallery path; mel-stats |
| `train_retrieval.py` | fresh init, mel-stat estimate, ramped loss, EMA, open-vocab kNN eval |
| `run_train.slurm` | A100 job (env `tf_gpu`), resumable |

## Run
```bash
sbatch run_train.slurm                          # 30k steps, batch 16, fresh
sbatch run_train.slurm --steps 60000 --batch 24
```
Outputs → `outputs/` (`best.pt` = best open-vocab top-1, `last.pt` resume, `train_log.json`).

## Metric
Open-vocab kNN **top-1 / top-5**: gallery = real train word-mels (≤ `gallery_per_word`
per word, covering all words), queries = generated test-mels. This is the number the
M23 attack needs to rise — compare against the frozen-M14 open-vocab retrieval
ceiling (attack oracle top-10 ≈ 0.108).

## Key knobs (`config.py` → `RCFG`)
- `min_occ` (1 = all words; raise to trim the long tail), `max_per_word` (train cap).
- `retr_start`, `retr_ramp` — when/how fast retrieval loss enters.
- `tau`, `lambda_con`, `lambda_corr` — retrieval loss shape.
- `train_steps` — differentiable rollout depth (memory driver; lower if OOM).
- `gallery_per_word`, `eval_queries` — eval sizing.

## Notes
- Backprop-through-sampling dominates memory once phase 2 starts; if you OOM, lower
  `--batch` or `RCFG.train_steps` (8 → 4), or push `retr_start` later.
- Words absent from the train gallery (e.g. a singleton that fell in the test split)
  are unretrievable by construction — inherent to honest open-vocab eval.
- Phonetics-vs-prosody attribution still needs M23's generator-side envelope-null
  control before claiming recovered phonetic content.
