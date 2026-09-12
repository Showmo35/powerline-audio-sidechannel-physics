# E2E Prediction Analysis Report

Date: March 30, 2026  
Project: `<REPO_ROOT>/baselines/E2E`

## 1. Scope

This report analyzes validation inference outputs from:

- `checkpoints_ctc/best_model.pt`
- `checkpoints_conditioned/best_model.pt`
- `checkpoints_joint/best_model.pt`

using the current inference result files:

- `inference_output_ctc/predictions.tsv`
- `inference_output_conditioned/predictions.tsv`
- `inference_output_joint/predictions.tsv`
- `inference_output_joint/transcription_compare.txt`

All three runs use the same 5 validation samples (`val_index` 0..4).

## 2. Metric Summary

### 2.1 Character Error Rate (CER)

| Model/Head | Avg CER | Std CER |
|---|---:|---:|
| CTC-only (CTC head) | 0.6454 | 0.0424 |
| Conditioned-only (Seq head) | 2.1887 | 0.4681 |
| Joint (CTC head) | **0.5988** | 0.0646 |
| Joint (Seq head) | 2.1566 | 0.4793 |

### 2.2 Word Error Rate (WER)

| Model/Head | Avg WER |
|---|---:|
| CTC-only (CTC head) | 1.1267 |
| Conditioned-only (Seq head) | 2.9994 |
| Joint (CTC head) | **1.0198** |
| Joint (Seq head) | 2.4798 |

### 2.3 Output Length Behavior (mean characters)

| Model | Ref length | Pred CTC length | Pred Seq length |
|---|---:|---:|---:|
| CTC-only | 75.6 | 51.4 | 0.0 |
| Conditioned-only | 75.6 | 0.0 | 199.4 |
| Joint | 75.6 | 58.0 | 199.8 |

Interpretation:
- CTC outputs are shorter than reference and phonetically approximate.
- Seq outputs are near max length and strongly over-generate.

## 3. Sample-by-Sample Comparison

| idx | CTC CER | Joint CTC CER | Conditioned Seq CER | Joint Seq CER |
|---:|---:|---:|---:|---:|
| 0 | 0.6182 | **0.5455** | 2.9455 | **2.8909** |
| 1 | 0.6105 | **0.5368** | 1.5684 | **1.5579** |
| 2 | 0.7027 | **0.6622** | 2.1486 | 2.1486 |
| 3 | 0.6047 | **0.5581** | 1.8837 | **1.7442** |
| 4 | 0.6912 | 0.6912 | **2.3971** | 2.4412 |

Findings:
- Joint CTC improved over CTC-only on 4/5 samples; 1 sample unchanged.
- Joint Seq improved on 3/5, tied 1/5, worse 1/5; still very poor overall.

## 4. Qualitative Error Analysis

## 4.1 CTC Head Behavior (CTC-only and Joint CTC)

Observed pattern:
- Retains rough speech rhythm and some lexical anchors.
- Produces partial phonetic strings and many substitutions/deletions.
- Better grounding to acoustics than seq head.

Example (sample 0):
- Ref: `oh so bill's got to come down the chimney house he said`
- Joint CTC: `h bo billt go to bo ot to toe so th chiy od oed`

Comments:
- `bill's` and some function words are partially preserved.
- Many consonant clusters and final consonants are degraded.

## 4.2 Seq Head Behavior (Conditioned-only and Joint Seq)

Observed pattern:
- Generates long fluent text that resembles corpus style (Alice prose).
- Frequently semantically unrelated to reference.
- Repetitions and loops (`...alalal...`, repeated fragments) indicate weak acoustic control.

Example (sample 2):
- Ref: `again i wonder if she'll fall right through the earth how funny it'll seem`
- Joint Seq: starts with unrelated narrative about a jar and orange marmalade.

Comments:
- The seq decoder is acting more like a language prior sampler than a strict transcription decoder.

## 5. Audio Output Notes

Per-sample audio pairs were generated in each inference directory:

- `sample_XXXX_actual.wav`
- `sample_XXXX_pred.wav`

These are reconstructed from mel spectrograms via Griffin-Lim inversion (24 iterations), so quality reflects both model error and inversion artifacts.

Available side-by-side artifacts per sample:
- text comparison: `transcription_compare.txt`
- spectrograms: `sample_XXXX_mel.png`
- audio pair: `sample_XXXX_actual.wav`, `sample_XXXX_pred.wav`

## 6. Why 3 Models Were Useful

The 3 training modes answered three different questions:

1. Can direct envelope->text work? (`ctc`)
2. Can envelope-constrained LM decoding work? (`conditioned`)
3. Does multitask training help either head? (`joint`)

Current answer:
- Yes, multitask helps the CTC path.
- Seq path remains under-constrained for faithful transcription.

## 7. Conclusions

1. Best current transcription path is **Joint CTC** (`avg CER 0.5988`), followed by CTC-only.
2. Seq-based decoding is currently unsuitable for accurate transcription (`avg CER > 2`).
3. Model has useful coarse phonetic recovery but still large substitution/deletion rates.
4. Inference sample size is small (`n=5`), so this is directional, not final.

## 8. Recommended Next Steps (priority order)

1. Evaluate on larger validation subset (`n>=100`) with same scripts to stabilize metrics.
2. Use CTC output as primary prediction in reporting and downstream tasks.
3. Constrain seq decoding:
   - lower max decode length
   - add EOS token and stop criterion
   - length penalty and repetition penalty
4. Improve acoustic alignment signal:
   - stronger CTC weight in joint mode
   - curriculum (CTC pretrain -> joint finetune)
5. Add beam search + lexicon constraints for CTC decoding.
6. Add confidence and per-word error analysis for targeted model debugging.

## 9. File References

- `<REPO_ROOT>/baselines/E2E/inference_output_ctc/predictions.tsv`
- `<REPO_ROOT>/baselines/E2E/inference_output_conditioned/predictions.tsv`
- `<REPO_ROOT>/baselines/E2E/inference_output_joint/predictions.tsv`
- `<REPO_ROOT>/baselines/E2E/inference_output_joint/transcription_compare.txt`
