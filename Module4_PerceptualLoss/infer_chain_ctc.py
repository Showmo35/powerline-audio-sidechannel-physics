"""
infer_chain_ctc.py
------------------
Chain Module 4 (Perceptual-Loss UNet) -> Module 3 (BiGRU CTC).

Loads noisy_val from Module 3's transcribe_data.npz, runs each sample
through the Module 4 UNet, then feeds the enhanced mel into the Module 3
CTC model for greedy decoding.  Computes CER/WER vs ground-truth labels
and compares against the baseline (Module 4 UNet trained with plain recon
loss, i.e. the predicted_val already stored in the data file).

Outputs:
  inference_output_chain/results.json
  inference_output_chain/transcriptions.txt
  inference_output_chain/cer_distribution.png
"""

import os, sys, json, time, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
M3_DIR = os.path.join(ROOT, 'Module3_BeamSearchLM')
M4_DIR = HERE

DATA_PATH   = os.path.join(M3_DIR, 'data', 'transcribe_data.npz')
UNET_CKPT   = os.path.join(M4_DIR, 'checkpoints', 'best_model.pt')
CTC_CKPT    = os.path.join(M3_DIR, 'checkpoints', 'best_cer_model.pt')
OUT_DIR     = os.path.join(M4_DIR, 'inference_output_chain')
BATCH       = 16

# ── model imports via importlib to avoid name collision ───────────────────────
def _load(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_m4_model = _load('m4_model', os.path.join(M4_DIR, 'model.py'))
_m3_model = _load('m3_model', os.path.join(M3_DIR, 'model.py'))

PowerlineUNet = _m4_model.PowerlineUNet
CTCEncoder    = _m3_model.CTCEncoder
greedy_decode = _m3_model.greedy_decode
IDX_TO_CHAR   = _m3_model.IDX_TO_CHAR
BLANK_IDX     = _m3_model.BLANK_IDX


# ── helpers ──────────────────────────────────────────────────────────────────
def decode_labels(texts, text_lens):
    out = []
    for t, tl in zip(texts, text_lens):
        chars = [IDX_TO_CHAR.get(int(c), '') for c in t[:tl]]
        out.append(''.join(chars))
    return out


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def compute_cer(preds, refs):
    total_d, total_l = 0, 0
    for p, r in zip(preds, refs):
        total_d += edit_distance(p, r)
        total_l += max(len(r), 1)
    return total_d / max(total_l, 1)


def compute_wer(preds, refs):
    total_d, total_l = 0, 0
    for p, r in zip(preds, refs):
        total_d += edit_distance(p.split(), r.split())
        total_l += max(len(r.split()), 1)
    return total_d / max(total_l, 1)


def run_inference(mels_np, unet, ctc, device, label):
    """Run mels through unet (optional) then ctc; return (preds, elapsed_s)."""
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(mels_np), BATCH):
            batch = torch.from_numpy(mels_np[i:i+BATCH]).float().to(device)
            if unet is not None:
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        batch = unet(batch)
                else:
                    batch = unet(batch)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = ctc(batch)
            else:
                logits = ctc(batch)
            preds.extend(greedy_decode(logits.cpu()))
    elapsed = time.time() - t0
    print(f"  [{label}] done in {elapsed:.1f}s")
    return preds, elapsed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data: {DATA_PATH}")
    d = np.load(DATA_PATH)
    noisy_val     = d['noisy_val']        # (590, 1, 80, 498) raw noisy mel
    predicted_val = d['predicted_val']    # (590, 1, 80, 498) plain UNet enhanced
    text_val      = d['text_val']
    text_len_val  = d['text_len_val']
    refs = decode_labels(text_val, text_len_val)
    N = len(noisy_val)
    print(f"Val samples : {N}  |  mel shape: {noisy_val.shape[1:]}")

    # ── load Perceptual-Loss UNet (Module 4) ──────────────────────────────────
    print(f"\nLoading Perceptual-Loss UNet: {UNET_CKPT}")
    unet_ckpt = torch.load(UNET_CKPT, map_location=device, weights_only=False)
    unet = PowerlineUNet(
        base_ch=unet_ckpt.get('args', {}).get('base_ch', 64)
    ).to(device)
    unet.load_state_dict(unet_ckpt['model_state'])
    unet.eval()
    print(f"  epoch {unet_ckpt.get('epoch','?')}, "
          f"val_loss={unet_ckpt.get('val_loss',0):.5f}")

    # ── load CTC encoder (Module 3 best-CER) ─────────────────────────────────
    print(f"\nLoading CTC encoder: {CTC_CKPT}")
    ctc_ckpt = torch.load(CTC_CKPT, map_location=device, weights_only=False)
    ctc_args = ctc_ckpt.get('args', {})
    ctc = CTCEncoder(
        n_mels=80,
        hidden=ctc_args.get('gru_hidden', 256),
        n_layers=ctc_args.get('gru_layers', 2),
        dropout=0.0,
    ).to(device)
    ctc.load_state_dict(ctc_ckpt['model_state'])
    ctc.eval()
    print(f"  epoch {ctc_ckpt.get('epoch','?')}, "
          f"val_cer={ctc_ckpt.get('val_cer',0):.4f}")

    # ── run all three conditions ──────────────────────────────────────────────
    print("\n--- Condition 1: Noisy mel -> CTC (no enhancement) ---")
    noisy_preds, t_noisy = run_inference(noisy_val, None, ctc, device, 'noisy->CTC')
    noisy_cer = compute_cer(noisy_preds, refs)
    noisy_wer = compute_wer(noisy_preds, refs)
    print(f"  CER: {noisy_cer:.4f}  WER: {noisy_wer:.4f}")

    print("\n--- Condition 2: Plain-recon UNet predicted_val -> CTC ---")
    recon_preds, t_recon = run_inference(predicted_val, None, ctc, device, 'plainUNet->CTC')
    recon_cer = compute_cer(recon_preds, refs)
    recon_wer = compute_wer(recon_preds, refs)
    print(f"  CER: {recon_cer:.4f}  WER: {recon_wer:.4f}")

    print("\n--- Condition 3: Perceptual-Loss UNet -> CTC (THIS is the new experiment) ---")
    perc_preds, t_perc = run_inference(noisy_val, unet, ctc, device, 'percUNet->CTC')
    perc_cer = compute_cer(perc_preds, refs)
    perc_wer = compute_wer(perc_preds, refs)
    print(f"  CER: {perc_cer:.4f}  WER: {perc_wer:.4f}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {'Condition':<35} {'CER':>8} {'WER':>8} {'Time':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Noisy -> CTC':<35} {noisy_cer:>8.4f} {noisy_wer:>8.4f} {t_noisy:>7.1f}s")
    print(f"  {'Plain-Recon UNet -> CTC':<35} {recon_cer:>8.4f} {recon_wer:>8.4f} {t_recon:>7.1f}s")
    print(f"  {'Perceptual-Loss UNet -> CTC':<35} {perc_cer:>8.4f} {perc_wer:>8.4f} {t_perc:>7.1f}s")
    print(f"{'='*62}")

    # ── per-sample analysis ───────────────────────────────────────────────────
    sample_results = []
    for i, ref in enumerate(refs):
        sample_results.append({
            'id': i,
            'ref': ref,
            'noisy_pred': noisy_preds[i],
            'noisy_cer': edit_distance(noisy_preds[i], ref) / max(len(ref), 1),
            'recon_pred': recon_preds[i],
            'recon_cer': edit_distance(recon_preds[i], ref) / max(len(ref), 1),
            'perc_pred': perc_preds[i],
            'perc_cer': edit_distance(perc_preds[i], ref) / max(len(ref), 1),
        })
    sample_results.sort(key=lambda x: x['perc_cer'])

    # ── save transcriptions.txt ───────────────────────────────────────────────
    txt_path = os.path.join(OUT_DIR, 'transcriptions.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Noisy->CTC          CER: {noisy_cer:.4f} | WER: {noisy_wer:.4f}\n")
        f.write(f"PlainRecon->CTC     CER: {recon_cer:.4f} | WER: {recon_wer:.4f}\n")
        f.write(f"PercLoss->CTC       CER: {perc_cer:.4f} | WER: {perc_wer:.4f}\n")
        f.write('=' * 70 + '\n\n')

        f.write("--- WORST 10 (by Perceptual UNet CER) ---\n\n")
        for r in sample_results[-10:]:
            f.write(f"Sample {r['id']} "
                    f"(noisy={r['noisy_cer']:.3f} recon={r['recon_cer']:.3f} perc={r['perc_cer']:.3f})\n")
            f.write(f"  REF  : {r['ref']}\n")
            f.write(f"  NOISY: {r['noisy_pred']}\n")
            f.write(f"  RECON: {r['recon_pred']}\n")
            f.write(f"  PERC : {r['perc_pred']}\n\n")

        f.write("--- BEST 10 (by Perceptual UNet CER) ---\n\n")
        for r in sample_results[:10]:
            f.write(f"Sample {r['id']} "
                    f"(noisy={r['noisy_cer']:.3f} recon={r['recon_cer']:.3f} perc={r['perc_cer']:.3f})\n")
            f.write(f"  REF  : {r['ref']}\n")
            f.write(f"  NOISY: {r['noisy_pred']}\n")
            f.write(f"  RECON: {r['recon_pred']}\n")
            f.write(f"  PERC : {r['perc_pred']}\n\n")

    # ── save results.json ─────────────────────────────────────────────────────
    results = {
        'n_val': N,
        'noisy_ctc': {'cer': noisy_cer, 'wer': noisy_wer, 'time_s': t_noisy},
        'plain_recon_ctc': {'cer': recon_cer, 'wer': recon_wer, 'time_s': t_recon},
        'perceptual_loss_ctc': {'cer': perc_cer, 'wer': perc_wer, 'time_s': t_perc},
        'unet_ckpt': UNET_CKPT,
        'ctc_ckpt': CTC_CKPT,
    }
    json_path = os.path.join(OUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ── CER distribution plot ─────────────────────────────────────────────────
    noisy_cers = [r['noisy_cer'] for r in sample_results]
    recon_cers = [r['recon_cer'] for r in sample_results]
    perc_cers  = [r['perc_cer']  for r in sample_results]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(noisy_cers, bins=30, alpha=0.5,
            label=f'Noisy->CTC (mean={np.mean(noisy_cers):.3f})')
    ax.hist(recon_cers, bins=30, alpha=0.5,
            label=f'PlainRecon->CTC (mean={np.mean(recon_cers):.3f})')
    ax.hist(perc_cers,  bins=30, alpha=0.5,
            label=f'PercLoss->CTC (mean={np.mean(perc_cers):.3f})')
    ax.set_xlabel('CER per sample')
    ax.set_ylabel('Count')
    ax.set_title('CER Distribution: Noisy vs Plain-Recon UNet vs Perceptual-Loss UNet')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cer_distribution.png'), dpi=150)
    plt.close()

    print(f"\nSaved to: {OUT_DIR}")
    print(f"  results.json, transcriptions.txt, cer_distribution.png")


if __name__ == '__main__':
    main()
