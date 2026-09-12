"""
inference_all.py
----------------
Run inference for all experiments and the E2E baseline, print comparison.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP_MODEL = _load_module_from_path(
    "experiments_model",
    os.path.join(HERE, "model.py"),
)
E2E_MODEL = _load_module_from_path(
    "e2e_model",
    os.path.join(ROOT, "E2E", "model.py"),
)
CASCADE_MODEL = _load_module_from_path(
    "cascade_model",
    os.path.join(ROOT, "Cascade", "model.py"),
)

BLANK_IDX = EXP_MODEL.BLANK_IDX
IDX_TO_CHAR = EXP_MODEL.IDX_TO_CHAR
PureCTCEncoder = EXP_MODEL.PureCTCEncoder


def tokens_to_text(tokens):
    return "".join(IDX_TO_CHAR.get(int(t), "") for t in tokens
                   if t != BLANK_IDX).strip()


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
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
            if t == prev: continue
            prev = t
            if t != BLANK_IDX: collapsed.append(t)
        out.append(tokens_to_text(collapsed))
    return out


def pick_device(d):
    if d == "cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_exp1(ckpt_path, data, idxs, device):
    """Pure CTC model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = PureCTCEncoder(
        hidden=args.get("gru_hidden", 256),
        n_layers=args.get("gru_layers", 2),
        dropout=args.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    noisy_val = data["noisy_val"]
    noisy = torch.from_numpy(noisy_val[idxs]).float().to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            logits = model(noisy)
    else:
        logits = model(noisy)
    return ctc_greedy_decode(logits)


@torch.no_grad()
def run_exp2(ckpt_path, data, idxs, device):
    """E2E model (UNet + CTC) with CTC-weighted loss."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = E2E_MODEL.E2EModel(
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

    noisy_val = data["noisy_val"]
    noisy = torch.from_numpy(noisy_val[idxs]).float().to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            _, ctc_logits, _ = model(noisy, None)
    else:
        _, ctc_logits, _ = model(noisy, None)
    return ctc_greedy_decode(ctc_logits)


@torch.no_grad()
def run_exp3(ckpt_path, data, idxs, device):
    """Cascade model with larger CTC encoder."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = CASCADE_MODEL.CascadeModel(
        unet_base_ch=args.get("unet_base_ch", 64),
        gru_hidden=args.get("gru_hidden", 512),
        gru_layers=args.get("gru_layers", 3),
        dropout=args.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    noisy_val = data["noisy_val"]
    noisy = torch.from_numpy(noisy_val[idxs]).float().to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            _, ctc_logits = model(noisy)
    else:
        _, ctc_logits = model(noisy)
    return ctc_greedy_decode(ctc_logits)


def main(args):
    device = pick_device(args.device)
    data = np.load(args.data)
    text_val = data["text_val"]
    text_len_val = data["text_len_val"]

    n_val = len(data["noisy_val"])
    n = min(args.n_samples, n_val)
    idxs = np.arange(n)

    refs = [tokens_to_text(text_val[i, :int(text_len_val[i])].tolist())
            for i in idxs]

    results = {}
    experiments = [
        ("exp1_pure_ctc", args.exp1_ckpt, run_exp1),
        ("exp2_ctc_weighted", args.exp2_ckpt, run_exp2),
        ("exp3_larger_ctc", args.exp3_ckpt, run_exp3),
    ]

    for name, ckpt, run_fn in experiments:
        if not ckpt or not os.path.exists(ckpt):
            print(f"SKIP {name}: checkpoint not found ({ckpt})")
            continue
        try:
            preds = run_fn(ckpt, data, idxs, device)
            cers = [cer(p, r) for p, r in zip(preds, refs)]
            wers = [wer(p, r) for p, r in zip(preds, refs)]
            results[name] = {
                "preds": preds, "cers": cers, "wers": wers,
                "avg_cer": float(np.mean(cers)),
                "avg_wer": float(np.mean(wers)),
            }
            print(f"{name}: avg_cer={np.mean(cers):.4f}  avg_wer={np.mean(wers):.4f}")
        except Exception as e:
            print(f"ERROR {name}: {e}")

    # Save comparison report
    os.makedirs(args.out, exist_ok=True)

    report_path = os.path.join(args.out, "comparison.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  EXPERIMENT COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Model':<25} {'Avg CER':>10} {'Avg WER':>10}\n")
        f.write("-" * 50 + "\n")
        for name, res in results.items():
            f.write(f"{name:<25} {res['avg_cer']:>10.4f} {res['avg_wer']:>10.4f}\n")
        f.write("\n")

        for si in range(n):
            f.write(f"--- Sample {idxs[si]} ---\n")
            f.write(f"  Reference: {refs[si]}\n")
            for name, res in results.items():
                f.write(f"  {name}: {res['preds'][si]}  "
                        f"(CER={res['cers'][si]:.4f})\n")
            f.write("\n")

    # Save JSON summary
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "avg_cer": res["avg_cer"],
            "avg_wer": res["avg_wer"],
        }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> {args.out}")
    print(open(report_path).read())


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Compare all experiments")
    parser.add_argument("--data",
        default=os.path.join(os.path.dirname(here), "E2E", "data", "e2e_data.npz"))
    parser.add_argument("--out",
        default=os.path.join(here, "results"))
    parser.add_argument("--exp1_ckpt",
        default=os.path.join(here, "exp1_pure_ctc", "best_model.pt"))
    parser.add_argument("--exp2_ckpt",
        default=os.path.join(here, "exp2_ctc_weighted", "best_model.pt"))
    parser.add_argument("--exp3_ckpt",
        default=os.path.join(here, "exp3_larger_ctc", "best_model.pt"))
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
        default="auto")
    args = parser.parse_args()
    main(args)
