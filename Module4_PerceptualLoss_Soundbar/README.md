# Module 4 Perceptual Loss - Soundbar

This folder adapts `Module4_PerceptualLoss` to the Soundbar leakage dataset.

What changes:
- Input: `Powerline_Data_Captures/Soundbar/LibriSpeech/chunk_*.bin`
- Target: `Powerline_Data_Captures/audio_chunks/chunk_*.wav`
- Alignment: sequential mapping anchored at the last full audio chunk, following `SOUNDBAR_ALIGNMENT_SUMMARY.md`
- Data preparation: creates paired noisy/clean mel spectrogram windows from aligned chunk pairs

What is reused:
- The original perceptual UNet, CTC perceptual encoder, and inference/training logic from `Module4_PerceptualLoss`

Typical workflow:
1. Prepare data
2. Train the perceptual UNet
3. Run inference and compare against the clean audio targets

Example:
```bash
cd "<REPO_ROOT>/Module4_PerceptualLoss_Soundbar"
python prepare_data.py
python train.py
python inference.py
```

Default paths point to:
- `Powerline_Data_Captures/Soundbar/LibriSpeech`
- `Powerline_Data_Captures/audio_chunks`
- `Module4_PerceptualLoss_Soundbar/data`
- `Module4_PerceptualLoss_Soundbar/checkpoints`
