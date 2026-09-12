# Powerline ASR — To-Do

## 1. Fix the Denoising Bottleneck (highest priority)

- [ ] Generate synthetic training data: add simulated powerline interference (60/120 Hz harmonics + transmission noise) to LibriSpeech or Common Voice
- [ ] Retrain U-Net on synthetic + real powerline data
- [ ] Evaluate CER after retraining — target below 0.40

## 2. Module 7 — Debug FullSubNet

- [ ] Investigate zero-improvement result (MSE/MAE unchanged, NaN correlation)
- [ ] Check if model is outputting constant/identity values
- [ ] Consider increasing model capacity (fb_hidden, sb_hidden) or fixing the evaluation script

## 3. Module 8 — Complete Inference

- [ ] Re-run inference for Module 8a (frozen encoder) and 8b (full model) with a longer job time limit
- [ ] Save transcription files and final CER/WER tables

## 4. LLM Post-Correction Pipeline

- [ ] Write a post-correction script using the Claude API on Module 3 beam+LM predictions
- [ ] Evaluate on current predictions (CER ~0.70) to establish a baseline correction gain
- [ ] Expand training data to more audiobooks (10–50 titles) to move from retrieval toward generalization
- [ ] Re-evaluate LLM correction after U-Net is improved (expect larger gains at CER ~0.35)

## 5. Module 3 — Transcription Quality Analysis

### Finding
There are **no meaningfully correct transcriptions** in Module 3 at current CER ~0.70.
The best predictions recover only short function words ("the", "and", "it") and rough
consonant shapes. No content words are correctly recovered in any sample.

Analysis method: ranked all 590 val samples by (1) word overlap between REF and BEAM
prediction, and (2) number of real words (≥4 chars) in the prediction.

### Best samples by CER (beam+LM)

| Sample | CER | REF | BEAM prediction |
|--------|-----|-----|-----------------|
| 216 | 0.566 | and peeped over the edge of the mushroom and her eyes | t the o e e e o te ano on and the to |
| 520 | 0.577 | settle the question and they repeated their arguments to her though as they all spoke at once she found it very | a it the ine and the e thet hie the he ae he he e he he ae he e e e e t e |
| 425 | 0.585 | 'the last time she saw them they were trying to put the door mouse into the teapot | 'ie ho the tho o en ha he ha the the he oe o e ho the thee |
| 442 | 0.600 | 'just as she said this she noticed that one of the trees had | 'ue whe tho to ' the hae the o ha |
| 10 | 0.604 | eye fell on a little glass box that was lying under the table she opened it and found in it | ae hit i lit et het het e e in an her e he she the et tet hut hit ettit |

### Best sample by word overlap (max overlap = 2 function words)

```
Sample 216  CER=0.566
  REF : and peeped over the edge of the mushroom and her eyes
  BEAM: t the o e e e o te ano on and the to
  → recovered: "and", "the"  — function words only

Sample 433  CER=0.611  (most real words in any prediction)
  REF : there again' said alice as she picked her way through the wood 'it's the
  BEAM: to of oen sand the ta hale hu ue te the mee' 'hthe e
  → recovered: "of", "the"  — "sand" is a near-miss for "said"
```

### Interpretation
The model is recovering **phoneme-level patterns** — consonant clusters, syllable lengths,
short function words — but cannot reconstruct content words. WER > 1.0 means word-level
output is worse than random alignment. Root cause: denoised spectrograms do not preserve
enough phonetic detail for any decoder to produce intelligible text at this stage.
The bottleneck is the U-Net front-end, not the ASR decoder (confirmed by Conformer
achieving the same CER as BiGRU).

- [ ] After U-Net improvement (target CER < 0.40), re-run this analysis and compare

---

## 6. Report Figures

- [ ] Generate training curve plots from `history.json` for Modules 3, 5, 6, 8, 9
- [ ] Create CER/WER grouped bar chart across all modules
- [ ] Draw system pipeline diagram
- [ ] Add all figures to `report_figures/`
