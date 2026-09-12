"""
inference.py
------------
Evaluate UNet reconstruction quality on validation set.

Computes MSE, MAE, Pearson correlation with clean spectrograms.
Saves spectrogram comparison plots and metrics to results.json.

Usage:
    python inference.py --data data/train_data.npz --ckpt checkpoints/best_model.pt
"""

import os, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import PowerlineUNet


def to_jsonable(obj):
    """Recursively convert NumPy types to JSON-serializable Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    return obj


def compute_metrics(pred, clean):
    """Compute per-sample MSE, MAE, Pearson correlation."""
    mse = np.mean((pred - clean) ** 2)
    mae = np.mean(np.abs(pred - clean))

    # Pearson correlation (flatten to 1-D)
    p_flat = pred.flatten()
    c_flat = clean.flatten()
    p_mean = p_flat.mean()
    c_mean = c_flat.mean()
    num = np.sum((p_flat - p_mean) * (c_flat - c_mean))
    den = np.sqrt(np.sum((p_flat - p_mean)**2) * np.sum((c_flat - c_mean)**2)) + 1e-12
    corr = num / den

    return mse, mae, corr


def plot_comparison(noisy, clean, pred, idx, metrics, out_dir):
    """Save a 3-panel spectrogram comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    vmin = min(noisy.min(), clean.min(), pred.min())
    vmax = max(noisy.max(), clean.max(), pred.max())
    kw = dict(aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')

    axes[0].imshow(noisy, **kw)
    axes[0].set_title('Noisy (Bandpass 50-4kHz)')

    axes[1].imshow(clean, **kw)
    axes[1].set_title('Clean (MP3 Ground Truth)')

    axes[2].imshow(pred, **kw)
    axes[2].set_title('UNet Predicted (Perceptual Loss)')

    for ax in axes:
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Mel bin')

    mse, mae, corr = metrics
    fig.suptitle(f"Sample {idx}  |  MSE={mse:.5f}  MAE={mae:.4f}  Corr={corr:.4f}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparison_{idx:04d}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def load_val_data(path):
    """Load validation arrays from legacy dense npz or sharded descriptor npz."""
    d = np.load(path, allow_pickle=True)
    if 'format' in d.files and str(d['format']) == 'sharded':
        shard_dir = str(d['shard_dir'])
        d.close()

        val_files = sorted(
            os.path.join(shard_dir, n)
            for n in os.listdir(shard_dir)
            if n.startswith('val_') and n.endswith('.npz')
        )
        if not val_files:
            raise RuntimeError(f"No val shards found in {shard_dir}")

        noisy_parts = []
        clean_parts = []
        for fp in val_files:
            with np.load(fp, mmap_mode='r') as sd:
                noisy_parts.append(sd['noisy'])
                clean_parts.append(sd['clean'])

        noisy_val = np.concatenate(noisy_parts, axis=0)
        clean_val = np.concatenate(clean_parts, axis=0)
        return noisy_val, clean_val

    noisy_val = d['noisy_val']
    clean_val = d['clean_val']
    return noisy_val, clean_val


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data} ...")
    noisy_val, clean_val = load_val_data(args.data)
    print(f"Val samples: {len(noisy_val)}")
    print(f"Spec shape: {noisy_val.shape[1:]}")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    model = PowerlineUNet(
        base_ch=ckpt_args.get('base_ch', 64)
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.5f})")
    print(f"Params: {model.count_params():,}")

    # Run inference on all val samples
    all_preds = []
    batch_size = args.batch
    print(f"\nRunning inference on {len(noisy_val)} val samples ...")

    with torch.no_grad():
        for i in range(0, len(noisy_val), batch_size):
            batch = torch.from_numpy(noisy_val[i:i+batch_size]).float().to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(batch)
            else:
                pred = model(batch)
            all_preds.append(pred.cpu().numpy())

    predicted = np.concatenate(all_preds, axis=0)
    print(f"Predictions shape: {predicted.shape}")

    # Compute metrics for each sample
    sample_metrics = []
    for i in range(len(predicted)):
        mse, mae, corr = compute_metrics(predicted[i], clean_val[i])
        sample_metrics.append({'idx': i, 'mse': mse, 'mae': mae, 'corr': corr})

    # Aggregate metrics
    mse_vals = [m['mse'] for m in sample_metrics]
    mae_vals = [m['mae'] for m in sample_metrics]
    corr_vals = [m['corr'] for m in sample_metrics]

    avg_mse  = np.mean(mse_vals)
    avg_mae  = np.mean(mae_vals)
    avg_corr = np.mean(corr_vals)
    std_mse  = np.std(mse_vals)
    std_mae  = np.std(mae_vals)
    std_corr = np.std(corr_vals)

    print(f"\n{'='*60}")
    print(f"  Reconstruction Quality (val set, N={len(predicted)})")
    print(f"  MSE  : {avg_mse:.6f} +/- {std_mse:.6f}")
    print(f"  MAE  : {avg_mae:.5f} +/- {std_mae:.5f}")
    print(f"  Corr : {avg_corr:.4f} +/- {std_corr:.4f}")
    print(f"{'='*60}\n")

    # Also compute metrics for noisy vs clean (baseline)
    noisy_metrics = []
    for i in range(len(noisy_val)):
        mse, mae, corr = compute_metrics(noisy_val[i], clean_val[i])
        noisy_metrics.append({'mse': mse, 'mae': mae, 'corr': corr})

    noisy_avg_mse  = np.mean([m['mse'] for m in noisy_metrics])
    noisy_avg_mae  = np.mean([m['mae'] for m in noisy_metrics])
    noisy_avg_corr = np.mean([m['corr'] for m in noisy_metrics])

    print(f"  Baseline (noisy vs clean):")
    print(f"  MSE  : {noisy_avg_mse:.6f}")
    print(f"  MAE  : {noisy_avg_mae:.5f}")
    print(f"  Corr : {noisy_avg_corr:.4f}")
    print(f"{'='*60}\n")

    # Save comparison plots
    n_plot = min(args.n_samples, len(predicted))
    print(f"Saving {n_plot} comparison plots ...")
    for i in range(n_plot):
        m = (sample_metrics[i]['mse'], sample_metrics[i]['mae'],
             sample_metrics[i]['corr'])
        plot_comparison(
            noisy_val[i, 0], clean_val[i, 0], predicted[i, 0],
            i, m, args.out)

    # Save metrics distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(mse_vals, bins=30, alpha=0.7, color='steelblue')
    axes[0].axvline(avg_mse, color='red', linestyle='--', label=f'Mean={avg_mse:.5f}')
    axes[0].set_title('MSE Distribution'); axes[0].legend()

    axes[1].hist(mae_vals, bins=30, alpha=0.7, color='steelblue')
    axes[1].axvline(avg_mae, color='red', linestyle='--', label=f'Mean={avg_mae:.4f}')
    axes[1].set_title('MAE Distribution'); axes[1].legend()

    axes[2].hist(corr_vals, bins=30, alpha=0.7, color='steelblue')
    axes[2].axvline(avg_corr, color='red', linestyle='--', label=f'Mean={avg_corr:.4f}')
    axes[2].set_title('Correlation Distribution'); axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "metrics_distribution.png"), dpi=130)
    plt.close()

    # Save results.json
    results = {
        'checkpoint': args.ckpt,
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_val_loss': ckpt.get('val_loss'),
        'n_val_samples': len(predicted),
        'reconstruction': {
            'mse_mean': float(avg_mse),
            'mse_std': float(std_mse),
            'mae_mean': float(avg_mae),
            'mae_std': float(std_mae),
            'corr_mean': float(avg_corr),
            'corr_std': float(std_corr),
        },
        'baseline_noisy': {
            'mse_mean': float(noisy_avg_mse),
            'mae_mean': float(noisy_avg_mae),
            'corr_mean': float(noisy_avg_corr),
        },
        'per_sample': sample_metrics,
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(to_jsonable(results), f, indent=2)

    print(f"Saved to: {args.out}")
    print(f"  results.json, metrics_distribution.png, {n_plot} comparison plots")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate UNet reconstruction with perceptual loss")
    parser.add_argument('--data', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'train_data.npz'))
    parser.add_argument('--ckpt', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best_model.pt'))
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'inference_output'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
