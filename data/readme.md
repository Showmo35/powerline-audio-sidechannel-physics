# Soundbar Data Guide: Audio Chunks and Alignment
## 1) Build clean audio chunks

Use:

- `Powerline_Data_Captures/build_audio_chunks.py`

What it does:

- Reads LibriSpeech `.flac` files recursively in sorted order.
- Concatenates audio into fixed-length `.wav` chunks (default: 30 minutes).
- Converts output to 16 kHz mono.
- Saves chunks as `chunk_001.wav`, `chunk_002.wav`, ... in `audio_chunks/`.
- Supports resume with `build_progress.json` if interrupted.

Typical command:

```bash
python Powerline_Data_Captures/build_audio_chunks.py \
	--flac-root /path/to/LibriSpeech/train-clean-100 \
	--output-dir /path/to/audio_chunks
```

## 2) Bin/Wav alignment rule

Data sources:

- Leakage signal: `Soundbar/LibriSpeech/chunk_XXX.bin`
- Clean audio target: `audio_chunks/chunk_YYY.wav`

Alignment principle:

- Chunks are aligned by sequence order.
- For each matched bin/wav pair, use only the last 30 minutes from both signals.
