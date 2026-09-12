"""
model.py
--------
Module 7: FullSubNet+-style mel spectrogram enhancement.

Dual-path architecture:
  - Full-band LSTM: captures global spectral context across all mel bins
  - Sub-band LSTM: per-frequency-bin local refinement using full-band features

Efficient implementation processes all sub-bands in parallel via reshaping.

Input/Output: (B, 1, 80, T) mel spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Full-Band Model
# =============================================================================

class FullBandModel(nn.Module):
    """
    Full-band LSTM: captures global spectral context.
    Processes all frequency bins at each time frame.

    Input:  (B, 1, F, T) mel spectrogram
    Output: (B, T, 2*hidden) bidirectional features
    """

    def __init__(self, n_freqs=80, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_freqs)
        self.lstm = nn.LSTM(
            input_size=n_freqs,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x):
        B, C, F, T = x.shape
        x_t = x.squeeze(1).permute(0, 2, 1)  # (B, T, F)
        x_t = self.norm(x_t)
        out, _ = self.lstm(x_t)               # (B, T, 2*hidden)
        return out


# =============================================================================
# Efficient FullSubNet Enhancer
# =============================================================================

class EfficientFullSubNetEnhancer(nn.Module):
    """
    FullSubNet+-style enhancer with efficient parallel sub-band processing.

    Full-band LSTM captures global context, then all sub-bands are processed
    in parallel by reshaping (B, F, T, D) -> (B*F, T, D) through a shared
    sub-band LSTM.

    Input:  (B, 1, 80, T) noisy mel spectrogram
    Output: (B, 1, 80, T) enhanced mel spectrogram

    Args:
        n_freqs: number of mel frequency bins (default 80)
        fb_hidden: full-band LSTM hidden size (default 64)
        sb_hidden: sub-band LSTM hidden size (default 32)
        fb_layers: full-band LSTM layers (default 2)
        sb_layers: sub-band LSTM layers (default 2)
        n_neighbor: sub-band neighborhood size (default 3)
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, n_freqs=80, fb_hidden=64, sb_hidden=32,
                 fb_layers=2, sb_layers=2, n_neighbor=3, dropout=0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_neighbor = n_neighbor

        # Full-band model
        self.fullband = FullBandModel(n_freqs, fb_hidden, fb_layers, dropout)
        fb_out_dim = fb_hidden * 2  # bidirectional

        # Sub-band model (shared across all frequency bins)
        sb_input_dim = fb_out_dim + (2 * n_neighbor + 1)
        self.sb_lstm = nn.LSTM(
            input_size=sb_input_dim,
            hidden_size=sb_hidden,
            num_layers=sb_layers,
            batch_first=True,
            dropout=dropout if sb_layers > 1 else 0.0,
        )
        self.sb_proj = nn.Linear(sb_hidden, 1)

    def forward(self, x):
        B, C, n_freqs, T = x.shape
        noisy = x.squeeze(1)  # (B, n_freqs, T)

        # Full-band pass
        fb_feat = self.fullband(x)  # (B, T, 2*fb_hidden)

        # Prepare sub-band neighborhoods for ALL frequencies at once
        # Pad frequency axis with reflection
        padded = F.pad(noisy, (0, 0, self.n_neighbor, self.n_neighbor),
                       mode='reflect')  # (B, n_freqs+2*n, T)

        # Extract local neighborhoods using unfold
        # unfold(dim, size, step): (B, n_freqs, T, 2*n+1)
        neighborhoods = padded.unfold(1, 2 * self.n_neighbor + 1, 1)
        # neighborhoods: (B, n_freqs, T, 2*n+1)

        # Expand full-band features to all frequencies
        # fb_feat: (B, T, fb_dim) -> (B, n_freqs, T, fb_dim)
        fb_expanded = fb_feat.unsqueeze(1).expand(B, n_freqs, T, -1)

        # Concatenate: (B, n_freqs, T, fb_dim + 2*n+1)
        combined = torch.cat([fb_expanded, neighborhoods], dim=-1)

        # Reshape for parallel LSTM: (B*n_freqs, T, input_dim)
        combined = combined.reshape(B * n_freqs, T, -1)

        # Sub-band LSTM
        out, _ = self.sb_lstm(combined)    # (B*n_freqs, T, sb_hidden)
        gains = torch.sigmoid(self.sb_proj(out))  # (B*n_freqs, T, 1)
        gains = gains.squeeze(-1).reshape(B, n_freqs, T)  # (B, n_freqs, T)

        # Apply multiplicative mask + global residual
        enhanced = (gains * noisy).unsqueeze(1)  # (B, 1, F, T)
        return enhanced + x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Multi-Scale STFT Loss
# =============================================================================

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def _loss_at_scale(self, pred, target, scale):
        if scale > 1:
            pred   = F.avg_pool2d(pred,   kernel_size=scale)
            target = F.avg_pool2d(target, kernel_size=scale)
        sc = torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-8)
        l1 = F.l1_loss(pred, target)
        return sc + l1

    def forward(self, pred, target):
        loss = sum(self._loss_at_scale(pred, target, s) for s in self.scales)
        return loss / len(self.scales)


class GeneratorLoss(nn.Module):
    def __init__(self, l1_weight=1.0, mstft_weight=1.0):
        super().__init__()
        self.l1_w = l1_weight
        self.mstft_w = mstft_weight
        self.mstft = MultiScaleSTFTLoss()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        mstft = self.mstft(pred, target)
        loss = self.l1_w * l1 + self.mstft_w * mstft
        return loss, l1.item(), mstft.item()


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    model = EfficientFullSubNetEnhancer(
        n_freqs=80, fb_hidden=64, sb_hidden=32,
        fb_layers=2, sb_layers=2, n_neighbor=3)
    print(f"EfficientFullSubNetEnhancer params: {model.count_params():,}")

    x = torch.randn(2, 1, 80, 98)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")

    loss_fn = GeneratorLoss()
    clean = torch.randn_like(x)
    loss, l1, mstft = loss_fn(y, clean)
    print(f"Loss: {loss:.4f}  L1: {l1:.4f}  MSTFT: {mstft:.4f}")
    print("Test passed.")
