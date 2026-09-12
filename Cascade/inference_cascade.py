"""
inference_cascade.py
--------------------
Run validation-set inference for the cascade model and save per-sample outputs.

Outputs:
  - predictions.tsv           : ref/pred text + CER per sample
  - transcription_compare.txt : readable side-by-side text
  - sample_*_actual.wav       : ground-truth audio from clean mel
  - sample_*_pred.wav         : predicted audio from UNet-cleaned mel
  - sample_*_mel.png          : noisy / clean / predicted mel plots
  - summary.json              : aggregate metrics
"""

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import BLANK_IDX, CascadeModel, IDX_TO_CHAR, load_pretrained_unet

SR = 16_000
N_FFT = 400
HOP = 160
N_MELS = 80


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokens_to_text(tokens):
    chars = []
    for t in tokens:
        t = int(t)
        if t == BLANK_IDX:
            continue
        chars.append(IDX_TO_CHAR.get(t, ""))
    return "".join(chars).strip()


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+cost)
    return int(dp[n, m])


def cer(pred, ref):
    return edit_distance(pred, ref) / max(1, len(ref))


def wer(pred, ref):
    return edit_distance(pred.split(), ref.split()) / max(1, len(ref.split()))


def ctc_greedy_decode(logits_tbv):
    ids_bt = torch.argmax(logits_tbv, dim=-1).transpose(0, 1)
    out = []
    for seq in ids_bt:
        collapsed = []
        prev = None
        for t in seq.tolist():
            if t == prev:
                continue
            prev = t
            if t != BLANK_IDX:
                collapsed.append(t)
        out.append(tokens_to_text(collapsed))
    return out


def load_model(ckpt_path, device, args):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    model = CascadeModel(
        unet_base_ch=ckpt_args.get("unet_base_ch", args.unet_base_ch),
        gru_hidden=ckpt_args.get("gru_hidden", args.gru_hidden),
        gru_layers=ckpt_args.get("gru_layers", args.gru_layers),
        dropout=ckpt_args.get("dropout", args.dropout),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded cascade model from {ckpt_path} "
          f"(phase={ckpt.get('phase', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.6f}, "
          f"val_cer={ckpt.get('val_cer', 0):.4f})")
    return model


def save_mel_plot(noisy, clean, pred, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = min(noisy.min(), clean.min(), pred.min())
    vmax = max(noisy.max(), clean.max(), pred.max())
    kw = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
              cmap="inferno")
    axes[0].imshow(noisy, **kw)
    axes[0].set_title("Noisy (Bandpass)")
    axes[1].imshow(clean, **kw)
    axes[1].set_title("Clean (GT)")
    axes[2].imshow(pred, **kw)
    axes[2].set_title("Predicted (Cascade)")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def denorm_log_mel_to_power_mel(mel_norm):
    log_spec = (mel_norm * 4.0) - 4.0
    mel_power = np.power(10.0, log_spec, dtype=np.float32)
    return np.maximum(mel_power, 1e-10)


def mel_to_audio_approx(mel_norm, sr=SR, n_fft=N_FFT, hop=HOP, n_iter=32):
    mel_power = denorm_log_mel_to_power_mel(mel_norm)
    audio = librosa.feature.inverse.mel_to_audio(
        M=mel_power, sr=sr, n_fft=n_fft, hop_length=hop,
        win_length=n_fft, power=2.0, n_iter=n_iter,
        center=False, htk=False,
    )
    target_len = max(n_fft, (mel_norm.shape[1] - 1) * hop + n_fft)
    audio = librosa.util.fix_length(audio, size=target_len)
    peak = np.max(np.abs(audio)) + 1e-9
    audio = 0.95 * (audio / peak)
    return audio.astype(np.float32)


def save_wav(path, audio, sr=SR):
    wav_write(path, sr, audio.astype(np.float32))


def write_transcription_report(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"Sample {r['val_index']}\n")
            f.write(f"  Actual:    {r['ref']}\n")
            f.write(f"  Predicted: {r['pred']}\n")
            f.write(f"  CER:       {r['cer']:.4f}\n")
            f.write(f"  WER:       {r['wer']:.4f}\n")
            f.write(f"  Mel plot:  {r['mel_plot']}\n")
            if r.get('actual_wav'):
                f.write(f"  GT audio:  {r['actual_wav']}\n")
                f.write(f"  Pred audio:{r['pred_wav']}\n")
            f.write("\n")


@torch.no_grad()
def run(args):
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.ckpt, device, args)

    data = np.load(args.data)
    noisy_val    = data["noisy_val"]
    clean_val    = data["clean_val"]
    text_val     = data["text_val"]
    text_len_val = data["text_len_val"]

    n_val = len(noisy_val)
    n = min(args.n_samples, n_val)
    if args.sample_mode == "random":
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(n_val, size=n, replace=False)
    else:
        idxs = np.arange(n)

    rows = []
    all_cers = []
    all_wers = []

    for start in range(0, len(idxs), args.batch):
        sub = idxs[start:start + args.batch]
        noisy = torch.from_numpy(noisy_val[sub]).float().to(device)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred_mel, ctc_logits = model(noisy)
        else:
            pred_mel, ctc_logits = model(noisy)

        pred_mel_np = pred_mel[:, 0].detach().cpu().numpy()
        pred_texts = ctc_greedy_decode(ctc_logits)

        for bi, idx in enumerate(sub):
            ref_text = tokens_to_text(
                text_val[idx, :int(text_len_val[idx])].tolist())
            pred_text = pred_texts[bi]

            c = cer(pred_text, ref_text)
            w = wer(pred_text, ref_text)
            all_cers.append(c)
            all_wers.append(w)

            sample_name = f"sample_{int(idx):04d}"
            png_path = os.path.join(args.out, f"{sample_name}_mel.png")
            save_mel_plot(
                noisy=noisy_val[idx, 0],
                clean=clean_val[idx, 0],
                pred=pred_mel_np[bi],
                out_png=png_path,
            )

            actual_wav_name = ""
            pred_wav_name = ""
            if not args.no_audio:
                actual_audio = mel_to_audio_approx(clean_val[idx, 0],
                                                   n_iter=args.gl_iters)
                pred_audio = mel_to_audio_approx(pred_mel_np[bi],
                                                 n_iter=args.gl_iters)
                actual_wav_name = f"{sample_name}_actual.wav"
                pred_wav_name   = f"{sample_name}_pred.wav"
                save_wav(os.path.join(args.out, actual_wav_name), actual_audio)
                save_wav(os.path.join(args.out, pred_wav_name), pred_audio)

            rows.append({
                "val_index": int(idx),
                "ref": ref_text,
                "pred": pred_text,
                "cer": float(c),
                "wer": float(w),
                "mel_plot": os.path.basename(png_path),
                "actual_wav": actual_wav_name,
                "pred_wav": pred_wav_name,
            })

    # Save predictions TSV
    tsv_path = os.path.join(args.out, "predictions.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("val_index\tcer\twer\tref\tpred\tactual_wav\tpred_wav\tmel_plot\n")
        for r in rows:
            f.write(f"{r['val_index']}\t{r['cer']:.6f}\t{r['wer']:.6f}\t"
                    f"{r['ref']}\t{r['pred']}\t"
                    f"{r['actual_wav']}\t{r['pred_wav']}\t{r['mel_plot']}\n")

    # Save readable report
    report_path = os.path.join(args.out, "transcription_compare.txt")
    write_transcription_report(report_path, rows)

    # Save summary JSON
    summary = {
        "ckpt": args.ckpt,
        "n_val_total": int(n_val),
        "n_samples_used": int(len(idxs)),
        "indices": [int(i) for i in idxs.tolist()],
        "avg_cer": float(np.mean(all_cers)),
        "avg_wer": float(np.mean(all_wers)),
        "std_cer": float(np.std(all_cers)),
        "std_wer": float(np.std(all_wers)),
        "device": str(device),
        "audio_saved": not args.no_audio,
        "griffin_lim_iters": int(args.gl_iters),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(os.path.dirname(here), "E2E")

    parser = argparse.ArgumentParser(
        description="Cascade model validation inference")
    parser.add_argument("--data",
        default=os.path.join(e2e_dir, "data", "e2e_data.npz"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sample_mode",
        choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device",
        choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--gl_iters", type=int, default=24)

    # Architecture defaults (used if not in checkpoint)
    parser.add_argument("--unet_base_ch", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    run(args)
