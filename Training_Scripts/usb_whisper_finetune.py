#!/usr/bin/env python3
import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
from datasets import Dataset
from scipy.signal import butter, resample_poly, sosfiltfilt
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import whisper
import tempfile
from scipy.io import wavfile


def _play_or_write_audio(array: np.ndarray, sr: int, out_dir: str, name_prefix: str) -> str:
    """Try to play audio with simpleaudio; if unavailable write a WAV and return path.

    Returns path to WAV written (may be in tempdir) so caller can reference it.
    """
    # Normalize floats to int16
    try:
        import simpleaudio as sa  # type: ignore
        have_simpleaudio = True
    except Exception:
        have_simpleaudio = False

    int16 = np.clip((array * 32767.0), -32768, 32767).astype(np.int16)

    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{name_prefix}.wav")
    wavfile.write(wav_path, sr, int16)

    if have_simpleaudio:
        try:
            play_obj = sa.play_buffer(int16.tobytes(), 1, 2, sr)
            play_obj.wait_done()
        except Exception:
            # fallback: just return written path
            pass

    return wav_path


# -----------------------------
# Signal processing utilities
# -----------------------------
def bandpass_sos(x: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def demodulate_usb_to_16k(
    usb_data: np.ndarray,
    sample_rate: int,
    center_freq: float,
    bandwidth: float,
    baseband_lpf_cutoff: float,
    asr_sample_rate: int,
) -> np.ndarray:
    lowcut = center_freq - bandwidth / 2
    highcut = center_freq + bandwidth / 2

    usb_band = bandpass_sos(usb_data, sample_rate, lowcut, highcut, order=4)

    t = np.arange(len(usb_band), dtype=np.float64) / float(sample_rate)
    mixed_down = usb_band * np.cos(2 * np.pi * center_freq * t)

    nyq = sample_rate / 2
    lpf_norm = min(baseband_lpf_cutoff / nyq, 0.999999)
    sos_lpf = butter(6, lpf_norm, btype="low", output="sos")
    demod_baseband = sosfiltfilt(sos_lpf, mixed_down)

    demod_16k = resample_poly(demod_baseband, asr_sample_rate, sample_rate).astype(np.float32)
    peak = float(np.max(np.abs(demod_16k)) + 1e-12)
    return demod_16k / peak


# -----------------------------
# Data discovery and alignment
# -----------------------------
def extract_chapter_num(path: str) -> Optional[int]:
    name = os.path.basename(path).lower()
    match = re.search(r"chap[_-]?(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def find_usb_files(data_roots: List[str]) -> List[str]:
    usb_files: List[str] = []
    for root in data_roots:
        usb_files.extend(glob.glob(os.path.join(root, "*_img.bin")))
    usb_files = sorted(set(usb_files))
    return usb_files


def find_mp3_for_chapter(mp3_root: str, chapter_num: int) -> Optional[str]:
    candidates = [
        os.path.join(mp3_root, f"Alice_In_Wonderland_ch_{chapter_num:02d}.mp3"),
        os.path.join(mp3_root, f"Alice_In_Wonderland_ch_{chapter_num}.mp3"),
        os.path.join(mp3_root, f"*{chapter_num:02d}*.mp3"),
        os.path.join(mp3_root, f"*{chapter_num}*.mp3"),
    ]

    for c in candidates[:2]:
        if os.path.exists(c):
            return c

    for patt in candidates[2:]:
        matches = glob.glob(patt)
        if matches:
            return sorted(matches)[0]

    return None


def transcribe_mp3_with_timestamps(mp3_path: str, teacher_model_name: str, language: str) -> List[Dict]:
    # Load MP3 with librosa to avoid ffmpeg dependency
    audio_array, sr = librosa.load(mp3_path, sr=16000, mono=True)
    
    teacher = whisper.load_model(teacher_model_name)
    result = teacher.transcribe(
        audio_array,
        language=language,
        task="transcribe",
        fp16=torch.cuda.is_available(),
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )
    segments = result.get("segments", [])
    return [s for s in segments if s.get("text", "").strip()]


def build_usb_text_examples(
    usb_paths: List[str],
    mp3_root: str,
    teacher_model_name: str,
    language: str,
    sample_rate: int,
    center_freq: float,
    bandwidth: float,
    baseband_lpf_cutoff: float,
    asr_sample_rate: int,
    max_seconds_per_usb: Optional[float],
    chapter_start_offset_sec: float,
) -> List[Dict]:
    examples: List[Dict] = []

    for usb_path in usb_paths:
        chapter = extract_chapter_num(usb_path)
        if chapter is None:
            print(f"[skip] Could not infer chapter from USB filename: {usb_path}")
            continue

        mp3_path = find_mp3_for_chapter(mp3_root, chapter)
        if mp3_path is None:
            print(f"[skip] No matching MP3 found for chapter {chapter:02d}")
            continue

        print(f"\n[chapter {chapter:02d}] USB={os.path.basename(usb_path)} | MP3={os.path.basename(mp3_path)}")

        usb_data = np.fromfile(usb_path, dtype=np.float32)
        if max_seconds_per_usb is not None:
            max_samples = int(max_seconds_per_usb * sample_rate)
            usb_data = usb_data[:max_samples]

        # Use the raw USB signal (no bandpass/demodulation). Resample to ASR rate and normalize.
        raw_16k = resample_poly(usb_data, asr_sample_rate, sample_rate).astype(np.float32)
        peak = float(np.max(np.abs(raw_16k)) + 1e-12)
        raw_16k = raw_16k / peak

        segments = transcribe_mp3_with_timestamps(mp3_path, teacher_model_name=teacher_model_name, language=language)
        print(f"  MP3 transcript segments: {len(segments)}")

        chapter_examples = 0
        for seg in segments:
            start_sec = float(seg["start"]) + chapter_start_offset_sec
            end_sec = float(seg["end"]) + chapter_start_offset_sec
            text = seg["text"].strip()

            start_idx = max(0, int(start_sec * asr_sample_rate))
            end_idx = min(len(raw_16k), int(end_sec * asr_sample_rate))

            if end_idx - start_idx < int(0.25 * asr_sample_rate):
                continue

            audio_chunk = raw_16k[start_idx:end_idx]
            if audio_chunk.size == 0:
                continue

            examples.append(
                {
                    "audio": {
                        "array": audio_chunk.astype(np.float32),
                        "sampling_rate": asr_sample_rate,
                    },
                    "text": text,
                    "chapter": chapter,
                    "usb_file": os.path.basename(usb_path),
                    "mp3_file": os.path.basename(mp3_path),
                }
            )
            chapter_examples += 1

        print(f"  Added aligned examples: {chapter_examples}")

    print(f"\nTotal training examples: {len(examples)}")
    return examples


# -----------------------------
# Trainer utilities
# -----------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Convert to tensor and mask padding tokens to -100 for loss computation
        attention_mask = labels_batch.get("attention_mask")
        if attention_mask is None:
            # If tokenizer did not produce attention mask (very short sequences), create one
            attention_mask = (labels_batch["input_ids"] != self.processor.tokenizer.pad_token_id).long()

        labels = labels_batch["input_ids"].masked_fill(torch.tensor(attention_mask).ne(1), -100)

        batch["labels"] = labels
        return batch


def prepare_dataset(batch: Dict, processor: WhisperProcessor, language: str) -> Dict:
    audio = batch["audio"]
    feats = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
    # feature_extractor returns a BatchFeature-like dict with key 'input_features'
    batch["input_features"] = feats["input_features"][0]
    tokenized = processor.tokenizer(batch["text"])
    batch["labels"] = tokenized["input_ids"]
    batch["language"] = language
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on USB-demodulated speech with MP3-derived transcription targets.")

    parser.add_argument("--data-roots", nargs="+", default=[
        "<REPO_ROOT>/May29_Alice",
        "<REPO_ROOT>/July10_Podcasts",
    ])
    parser.add_argument("--mp3-root", default="<REPO_ROOT>/Alice_In_Wonderland_mp3")
    parser.add_argument("--sample-rate", type=int, default=200000)
    parser.add_argument("--asr-sample-rate", type=int, default=16000)

    parser.add_argument("--center-freq", type=float, default=20000.0)
    parser.add_argument("--bandwidth", type=float, default=2500.0)
    parser.add_argument("--baseband-lpf-cutoff", type=float, default=2500.0)

    parser.add_argument("--chapter-start-offset-sec", type=float, default=0.0,
                        help="Optional offset to align MP3 segment timestamps onto USB timeline.")
    parser.add_argument("--max-seconds-per-usb", type=float, default=None)

    parser.add_argument("--teacher-model", default="base", help="OpenAI Whisper model used to create MP3 segment targets.")
    parser.add_argument("--base-model", default="openai/whisper-small.en", help="HF Whisper model to fine-tune.")
    parser.add_argument("--language", default="en")

    parser.add_argument("--output-dir", default="<REPO_ROOT>/output/usb_whisper_finetuned")
    parser.add_argument("--predict-only", action="store_true", help="Only run prediction using a model in --output-dir; skip training.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    usb_paths = find_usb_files(args.data_roots)
    if not usb_paths:
        raise RuntimeError("No USB *_img.bin files found under --data-roots")

    print(f"Found {len(usb_paths)} USB files")

    examples = build_usb_text_examples(
        usb_paths=usb_paths,
        mp3_root=args.mp3_root,
        teacher_model_name=args.teacher_model,
        language=args.language,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        bandwidth=args.bandwidth,
        baseband_lpf_cutoff=args.baseband_lpf_cutoff,
        asr_sample_rate=args.asr_sample_rate,
        max_seconds_per_usb=args.max_seconds_per_usb,
        chapter_start_offset_sec=args.chapter_start_offset_sec,
    )

    if len(examples) < 8:
        raise RuntimeError(
            f"Only {len(examples)} examples were created. Need more aligned data for stable fine-tuning."
        )

    # Create train/test split: use all chapters, random 80/20 split
    raw_ds = Dataset.from_list(examples)
    split = raw_ds.train_test_split(test_size=0.2, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    # Keep raw_eval_examples aligned with eval_ds for playback purposes by constructing list
    raw_eval_examples = [examples[i] for i in split["test_indices"]] if "test_indices" in split else list(eval_ds)

    # If user requested predict-only, load model + processor from output_dir and run prediction
    if args.predict_only:
        if not os.path.isdir(args.output_dir):
            raise RuntimeError(f"--output-dir does not exist or has no model: {args.output_dir}")

        print("Loading model and processor from output_dir for prediction...")
        processor = WhisperProcessor.from_pretrained(args.output_dir)
        model = WhisperForConditionalGeneration.from_pretrained(args.output_dir)

        # Prepare the evaluation dataset so it contains `input_features` and `labels`
        # (same preprocessing used during training).
        eval_ds = eval_ds.map(lambda b: prepare_dataset(b, processor, args.language), remove_columns=eval_ds.column_names)

        # Minimal training args for building a trainer instance to use predict
        pred_training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.batch_size,
            do_train=False,
            report_to="none",
            predict_with_generate=True,
            generation_max_length=128,
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        trainer = Seq2SeqTrainer(
            args=pred_training_args,
            model=model,
            data_collator=data_collator,
            eval_dataset=eval_ds,
        )

        print("Running prediction on evaluation dataset...")
        pred_output = trainer.predict(eval_ds)
        preds = pred_output.predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)

        label_ids = pred_output.label_ids
        if label_ids is not None:
            label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
            decoded_labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        else:
            decoded_labels = [""] * len(decoded_preds)

        import json

        pred_lines = []
        for ref, pred in zip(decoded_labels, decoded_preds):
            pred_lines.append(json.dumps({"reference": ref, "prediction": pred}, ensure_ascii=False))

        pred_file = os.path.join(args.output_dir, "eval_predictions.jsonl")
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write("\n".join(pred_lines))

        print(f"Saved {len(pred_lines)} predictions to: {pred_file}")

        # Play or save evaluation audio and show prediction vs. reference
        out_clips_dir = os.path.join(args.output_dir, "eval_audio_clips")
        os.makedirs(out_clips_dir, exist_ok=True)
        n_show = min(len(decoded_preds), len(raw_eval_examples))
        for i in range(n_show):
            ref = decoded_labels[i]
            pred = decoded_preds[i]
            example = raw_eval_examples[i]
            usb_file = example.get("usb_file")
            mp3_file = example.get("mp3_file")
            print("\n--- Eval Example %d ---" % i)
            print(f"USB: {usb_file} | MP3: {mp3_file} | chapter: {example.get('chapter')}")
            print("Reference:", ref)
            print("Prediction:", pred)

            audio = example["audio"]
            wav_path = _play_or_write_audio(audio["array"], audio["sampling_rate"], out_clips_dir, f"eval_{i}_chap{example.get('chapter')}")
            print(f"Audio (wav) at: {wav_path}")
        return

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    # Newer transformers versions expect generation parameters in `generation_config`.
    # Move suppress_tokens there to avoid ValueError when saving the model.
    try:
        gen_cfg = model.generation_config
    except Exception:
        from transformers import GenerationConfig

        gen_cfg = GenerationConfig()
        model.generation_config = gen_cfg

    model.generation_config.suppress_tokens = []

    train_ds = train_ds.map(lambda b: prepare_dataset(b, processor, args.language), remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda b: prepare_dataset(b, processor, args.language), remove_columns=eval_ds.column_names)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        max_steps=-1,
        num_train_epochs=5,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        predict_with_generate=True,
        generation_max_length=128,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")

    # Run prediction on the evaluation dataset and save decoded outputs
    print("Running prediction on evaluation dataset...")
    pred_output = trainer.predict(eval_ds)
    preds = pred_output.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)

    label_ids = pred_output.label_ids
    if label_ids is not None:
        # Replace -100 with pad token id so tokenizer can decode
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        decoded_labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    else:
        decoded_labels = [""] * len(decoded_preds)

    import json

    pred_lines = []
    for ref, pred in zip(decoded_labels, decoded_preds):
        pred_lines.append(json.dumps({"reference": ref, "prediction": pred}, ensure_ascii=False))

    pred_file = os.path.join(args.output_dir, "eval_predictions.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines))

    print(f"Saved {len(pred_lines)} predictions to: {pred_file}")
    # Play or save evaluation audio and show prediction vs. reference (post-training)
    out_clips_dir = os.path.join(args.output_dir, "eval_audio_clips")
    os.makedirs(out_clips_dir, exist_ok=True)
    n_show = min(len(decoded_preds), len(raw_eval_examples))
    for i in range(n_show):
        ref = decoded_labels[i]
        pred = decoded_preds[i]
        example = raw_eval_examples[i]
        usb_file = example.get("usb_file")
        mp3_file = example.get("mp3_file")
        print("\n--- Eval Example %d ---" % i)
        print(f"USB: {usb_file} | MP3: {mp3_file} | chapter: {example.get('chapter')}")
        print("Reference:", ref)
        print("Prediction:", pred)

        audio = example["audio"]
        wav_path = _play_or_write_audio(audio["array"], audio["sampling_rate"], out_clips_dir, f"eval_{i}_chap{example.get('chapter')}")
        print(f"Audio (wav) at: {wav_path}")


if __name__ == "__main__":
    main()
