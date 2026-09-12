"""
inference.py
------------
Evaluate FullSubNet+ enhancement quality on validation set.
Computes MSE, MAE, Pearson correlation. Saves spectrogram plots.

Usage:
    python inference.py --data data/train_data.npz --ckpt checkpoints/best_model.pt
"""

import os, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import EfficientFullSubNetEnhancer


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    d = np.load(args.data)
    noisy_val = d['noisy_val']
    clean_val = d['clean_val']
    print(f"Val samples: {len(noisy_val)}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    model = EfficientFullSubNetEnhancer(
        n_freqs=80,
        fb_hidden=ckpt_args.get('fb_hidden', 64),
        sb_hidden=ckpt_args.get('sb_hidden', 32),
        fb_layers=ckpt_args.get('fb_layers', 2),
        sb_layers=ckpt_args.get('sb_layers', 2),
        n_neighbor=ckpt_args.get('n_neighbor', 3),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.5f})")

    # Run inference on all val
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(noisy_val), args.batch):
            batch = torch.from_numpy(noisy_val[i:i+args.batch]).float().to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(batch)
            else:
                pred = model(batch)
            all_preds.append(pred.float().cpu().numpy())

    predicted = np.concatenate(all_preds, axis=0)

    # Compute metrics
    mse_per_sample = np.mean((predicted - clean_val) ** 2, axis=(1, 2, 3))
    mae_per_sample = np.mean(np.abs(predicted - clean_val), axis=(1, 2, 3))

    # Baseline: noisy vs clean
    base_mse = np.mean((noisy_val - clean_val) ** 2, axis=(1, 2, 3))
    base_mae = np.mean(np.abs(noisy_val - clean_val), axis=(1, 2, 3))

    # Pearson correlation
    def corr(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.corrcoef(a_flat, b_flat)[0, 1]

    corrs_pred  = [corr(predicted[i], clean_val[i]) for i in range(len(clean_val))]
    corrs_noisy = [corr(noisy_val[i], clean_val[i]) for i in range(len(clean_val))]

    print(f"\n{'='*60}")
    print(f"  {'Metric':<20} {'Noisy':>10} {'FullSubNet':>12} {'Improvement':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'MSE':<20} {np.mean(base_mse):>10.5f} {np.mean(mse_per_sample):>12.5f} "
          f"{(np.mean(base_mse) - np.mean(mse_per_sample)):>+12.5f}")
    print(f"  {'MAE':<20} {np.mean(base_mae):>10.5f} {np.mean(mae_per_sample):>12.5f} "
          f"{(np.mean(base_mae) - np.mean(mae_per_sample)):>+12.5f}")
    print(f"  {'Correlation':<20} {np.mean(corrs_noisy):>10.4f} "
          f"{np.mean(corrs_pred):>12.4f} "
          f"{(np.mean(corrs_pred) - np.mean(corrs_noisy)):>+12.4f}")
    print(f"{'='*60}")

    # Save metrics
    summary = {
        'n_val': len(clean_val),
        'fullsubnet': {
            'mse_mean': float(np.mean(mse_per_sample)),
            'mse_std': float(np.std(mse_per_sample)),
            'mae_mean': float(np.mean(mae_per_sample)),
            'corr_mean': float(np.mean(corrs_pred)),
        },
        'baseline_noisy': {
            'mse_mean': float(np.mean(base_mse)),
            'mae_mean': float(np.mean(base_mae)),
            'corr_mean': float(np.mean(corrs_noisy)),
        },
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot sample comparisons
    n_plot = min(args.n_samples, len(noisy_val))
    for i in range(n_plot):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        n = noisy_val[i, 0]
        c = clean_val[i, 0]
        p = predicted[i, 0]
        vmin = min(n.min(), c.min(), p.min())
        vmax = max(n.max(), c.max(), p.max())
        kw = dict(aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')
        axes[0].imshow(n, **kw); axes[0].set_title('Noisy (Bandpass)')
        axes[1].imshow(c, **kw); axes[1].set_title('Clean (MP3)')
        axes[2].imshow(p, **kw); axes[2].set_title('FullSubNet Enhanced')
        for ax in axes:
            ax.set_xlabel('Time'); ax.set_ylabel('Mel bin')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'sample_{i}.png'), dpi=150)
        plt.close()

    # Metric distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mse_per_sample, bins=30, alpha=0.6, label='FullSubNet')
    axes[0].hist(base_mse, bins=30, alpha=0.6, label='Noisy baseline')
    axes[0].set_xlabel('MSE'); axes[0].set_title('MSE Distribution')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(corrs_pred, bins=30, alpha=0.6, label='FullSubNet')
    axes[1].hist(corrs_noisy, bins=30, alpha=0.6, label='Noisy baseline')
    axes[1].set_xlabel('Correlation'); axes[1].set_title('Correlation Distribution')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'metric_distributions.png'), dpi=150)
    plt.close()

    print(f"Saved to: {args.out}")


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(_HERE, 'data', 'train_data.npz'))
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', default=os.path.join(_HERE, 'inference_output'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)
