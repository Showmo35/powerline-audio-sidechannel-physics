"""
inference_e2e.py
----------------
Run validation-set inference for E2E models and save per-sample outputs.

Outputs:
  - predictions.tsv          : ref/pred text + CER per sample
  - transcription_compare.txt: readable actual vs predicted text
  - sample_*_actual.wav      : reconstructed ground-truth audio from clean mel
  - sample_*_pred.wav        : reconstructed predicted audio from predicted mel
  - summary.json             : aggregate metrics and run metadata
  - mel_*.png                : noisy/clean/predicted mel plots per sample
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

from model import BLANK_IDX, E2EModel, IDX_TO_CHAR

SR = 16_000
N_FFT = 400
HOP = 160
N_MELS = 80


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
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
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def cer(pred, ref):
    return edit_distance(pred, ref) / max(1, len(ref))


def ctc_greedy_decode(logits_tbv):
    # logits_tbv: (T, B, V)
    ids_bt = torch.argmax(logits_tbv, dim=-1).transpose(0, 1)  # (B, T)
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


@torch.no_grad()
def seq_autoregressive_decode(model, predicted_mel, max_len):
    # predicted_mel: (B,1,80,T)
    bsz = predicted_mel.shape[0]
    device = predicted_mel.device
    tokens = torch.zeros((bsz, 1), dtype=torch.long, device=device)  # start token = blank
    for _ in range(max_len):
        logits = model.cond_decoder(predicted_mel, tokens)  # (B, L, V)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B,1)
        tokens = torch.cat([tokens, next_tok], dim=1)
    # drop start token
    return [tokens_to_text(seq.tolist()) for seq in tokens[:, 1:]]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    model = E2EModel(
        base_ch=args.get("base_ch", 64),
        gru_hidden=args.get("gru_hidden", 256),
        gru_layers=args.get("gru_layers", 2),
        dropout=args.get("dropout", 0.1),
        dec_d_model=args.get("dec_d_model", 256),
        dec_heads=args.get("dec_heads", 4),
        dec_layers=args.get("dec_layers", 3),
        dec_max_text_len=args.get("max_text_len", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mode = ckpt.get("mode", args.get("mode", "joint"))
    return model, mode, args


def save_mel_plot(noisy, clean, pred, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = min(noisy.min(), clean.min(), pred.min())
    vmax = max(noisy.max(), clean.max(), pred.max())
    kw = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
    axes[0].imshow(noisy, **kw)
    axes[0].set_title("Noisy")
    axes[1].imshow(clean, **kw)
    axes[1].set_title("Clean (GT)")
    axes[2].imshow(pred, **kw)
    axes[2].set_title("Predicted")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def denorm_log_mel_to_power_mel(mel_norm):
    """
    Undo prepare_data.py normalization approximately:
      log_spec_norm = (log10(mel_power) + 4.0) / 4.0
      mel_power     = 10 ** (log_spec_norm*4 - 4)
    """
    log_spec = (mel_norm * 4.0) - 4.0
    mel_power = np.power(10.0, log_spec, dtype=np.float32)
    return np.maximum(mel_power, 1e-10)


def mel_to_audio_approx(mel_norm, sr=SR, n_fft=N_FFT, hop=HOP, n_iter=32):
    mel_power = denorm_log_mel_to_power_mel(mel_norm)
    audio = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        power=2.0,
        n_iter=n_iter,
        center=False,
        htk=False,
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
            f.write(f"Actual:       {r['ref']}\n")
            f.write(f"Pred (CTC):   {r['pred_ctc']}\n")
            f.write(f"Pred (Seq):   {r['pred_seq']}\n")
            f.write(f"CER CTC:      {r['cer_ctc']:.6f}\n")
            f.write(f"CER Seq:      {r['cer_seq']:.6f}\n")
            f.write(f"Actual Audio: {r.get('actual_wav', '')}\n")
            f.write(f"Pred Audio:   {r.get('pred_wav', '')}\n")
            f.write(f"Mel Plot:     {r['mel_plot']}\n")
            f.write("\n")


@torch.no_grad()
def run(args):
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model, mode, ckpt_args = load_model(args.ckpt, device)
    data = np.load(args.data)
    noisy_val = data["noisy_val"]
    clean_val = data["clean_val"]
    text_val = data["text_val"]
    text_len_val = data["text_len_val"]

    n_val = len(noisy_val)
    n = min(args.n_samples, n_val)
    if args.sample_mode == "random":
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(n_val, size=n, replace=False)
    else:
        idxs = np.arange(n)

    run_mode = args.mode if args.mode != "auto" else mode
    do_ctc = run_mode in ("ctc", "joint")
    do_seq = run_mode in ("conditioned", "joint")
    max_text_len = int(args.max_text_len or ckpt_args.get("max_text_len", 200))

    rows = []
    ctc_cers = []
    seq_cers = []

    for start in range(0, len(idxs), args.batch):
        sub = idxs[start:start + args.batch]

        noisy = torch.from_numpy(noisy_val[sub]).float().to(device)
        clean = clean_val[sub]
        text = text_val[sub]
        text_len = text_len_val[sub]

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred_mel, ctc_logits, _ = model(noisy, None)
        else:
            pred_mel, ctc_logits, _ = model(noisy, None)

        pred_mel_np = pred_mel[:, 0].detach().cpu().numpy()
        ctc_texts = ctc_greedy_decode(ctc_logits) if do_ctc else [""] * len(sub)
        seq_texts = seq_autoregressive_decode(model, pred_mel, max_text_len) if do_seq else [""] * len(sub)

        for bi, idx in enumerate(sub):
            ref = tokens_to_text(text[bi, : int(text_len[bi])].tolist())
            pred_ctc = ctc_texts[bi]
            pred_seq = seq_texts[bi]

            cer_ctc = cer(pred_ctc, ref) if do_ctc else -1.0
            cer_seq = cer(pred_seq, ref) if do_seq else -1.0
            if do_ctc:
                ctc_cers.append(cer_ctc)
            if do_seq:
                seq_cers.append(cer_seq)

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
                actual_audio = mel_to_audio_approx(clean_val[idx, 0], n_iter=args.gl_iters)
                pred_audio = mel_to_audio_approx(pred_mel_np[bi], n_iter=args.gl_iters)
                actual_wav_name = f"{sample_name}_actual.wav"
                pred_wav_name = f"{sample_name}_pred.wav"
                save_wav(os.path.join(args.out, actual_wav_name), actual_audio)
                save_wav(os.path.join(args.out, pred_wav_name), pred_audio)

            rows.append(
                {
                    "val_index": int(idx),
                    "ref": ref,
                    "pred_ctc": pred_ctc,
                    "pred_seq": pred_seq,
                    "cer_ctc": float(cer_ctc),
                    "cer_seq": float(cer_seq),
                    "mel_plot": os.path.basename(png_path),
                    "actual_wav": actual_wav_name,
                    "pred_wav": pred_wav_name,
                }
            )

    tsv_path = os.path.join(args.out, "predictions.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(
            "val_index\tcer_ctc\tcer_seq\tref\tpred_ctc\tpred_seq\t"
            "actual_wav\tpred_wav\tmel_plot\n"
        )
        for r in rows:
            f.write(
                f"{r['val_index']}\t{r['cer_ctc']:.6f}\t{r['cer_seq']:.6f}\t"
                f"{r['ref']}\t{r['pred_ctc']}\t{r['pred_seq']}\t"
                f"{r['actual_wav']}\t{r['pred_wav']}\t{r['mel_plot']}\n"
            )

    report_path = os.path.join(args.out, "transcription_compare.txt")
    write_transcription_report(report_path, rows)

    summary = {
        "ckpt": args.ckpt,
        "mode_from_ckpt": mode,
        "mode_used": run_mode,
        "n_val_total": int(n_val),
        "n_samples_used": int(len(idxs)),
        "indices": [int(i) for i in idxs.tolist()],
        "avg_cer_ctc": float(np.mean(ctc_cers)) if ctc_cers else None,
        "avg_cer_seq": float(np.mean(seq_cers)) if seq_cers else None,
        "device": str(device),
        "predictions_tsv": os.path.basename(tsv_path),
        "transcription_compare_txt": os.path.basename(report_path),
        "audio_saved": not args.no_audio,
        "griffin_lim_iters": int(args.gl_iters),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved -> {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run val-set inference for E2E checkpoints")
    parser.add_argument("--data", default=os.path.join(here, "data", "e2e_data.npz"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["auto", "ctc", "conditioned", "joint"], default="auto")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max_text_len", type=int, default=0, help="0 = use checkpoint max_text_len")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_audio", action="store_true", help="Disable audio reconstruction and wav saving")
    parser.add_argument("--gl_iters", type=int, default=24, help="Griffin-Lim iterations for mel inversion")
    args = parser.parse_args()
    run(args)
