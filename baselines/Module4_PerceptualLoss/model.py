"""
model.py
--------
Module 4: Perception-Aware UNet for powerline-to-audio reconstruction.

Contains:
  - PowerlineUNet: Same spectrogram U-Net as model_bandpass.py
  - CTCPerceptualEncoder: Frozen pretrained CTC encoder as perceptual feature extractor
  - MultiScaleSTFTLoss: Multi-resolution spectral loss
  - PerceptionAwareLoss: L1 + MSTFT + CTC perceptual cosine-similarity loss

The key idea: instead of only minimising pixel-level spectral error (L1/MSTFT),
we also require that the UNet output produces similar internal representations
in a frozen CTC encoder as the clean spectrogram does.  This encourages
reconstructions that are discriminative for speech recognition.

Input/Output shape: (batch, 1, N_MELS=80, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks (same as UNet/model_bandpass.py)
# =============================================================================

class ConvBlock(nn.Module):
    """Conv2d -> BN -> LeakyReLU  (x2)"""
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block for bottleneck."""
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    """ConvBlock + MaxPool along time only."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d((1, 2))   # pool along time axis only

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """Bilinear upsample + ConvBlock + skip concatenation."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=(1, 2), mode='bilinear',
                                align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [0, diff])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# PowerlineUNet
# =============================================================================

class PowerlineUNet(nn.Module):
    """
    Spectrogram U-Net for audio reconstruction from bandpass-filtered
    powerline signal.

    Encoder:    4 DownBlocks (1 -> 64 -> 128 -> 256 -> 512)
    Bottleneck: 2 ResBlocks
    Decoder:    4 UpBlocks with skip connections
    Global residual: output = conv(decoded) + input

    Input : (B, 1, 80, T)  - noisy powerline log-mel spectrogram
    Output: (B, 1, 80, T)  - reconstructed clean log-mel spectrogram
    """

    def __init__(self, base_ch=64):
        super().__init__()

        self.enc1 = DownBlock(1,        base_ch)
        self.enc2 = DownBlock(base_ch,  base_ch*2)
        self.enc3 = DownBlock(base_ch*2, base_ch*4)
        self.enc4 = DownBlock(base_ch*4, base_ch*8)

        self.bottleneck = nn.Sequential(
            ResBlock(base_ch*8),
            ResBlock(base_ch*8),
        )

        self.dec4 = UpBlock(base_ch*8,  base_ch*8,  base_ch*4)
        self.dec3 = UpBlock(base_ch*4,  base_ch*4,  base_ch*2)
        self.dec2 = UpBlock(base_ch*2,  base_ch*2,  base_ch)
        self.dec1 = UpBlock(base_ch,    base_ch,    base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.bottleneck(x4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1) + x   # global residual

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# CTCPerceptualEncoder - frozen CTC encoder as perceptual feature extractor
# =============================================================================

class CTCPerceptualEncoder(nn.Module):
    """
    Same architecture as CTCEncoder (Conv stem + BiGRU + linear head),
    but used as a FROZEN perceptual feature extractor.

    Extracts intermediate features at two scales:
      - After conv stem:  (B, T//4, 32*20) = (B, T//4, 640)
      - After BiGRU:      (B, T//4, hidden*2)

    All parameters are frozen (requires_grad=False).
    """

    VOCAB_SIZE = 29

    def __init__(self, n_mels=80, hidden=256, n_layers=2,
                 vocab_size=29, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        conv_out_freq = n_mels // 4
        gru_in = 32 * conv_out_freq

        self.gru = nn.GRU(
            gru_in, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden * 2, vocab_size)

        self.hidden = hidden

    def forward(self, x):
        """Standard CTC forward: (B, 1, M, T) -> (T', B, V)."""
        c = self.conv(x)                    # (B, 32, 20, T//4)
        B, C, M, T = c.shape
        c = c.permute(0, 3, 1, 2).reshape(B, T, C * M)
        out, _ = self.gru(c)
        out    = self.dropout(out)
        logits = self.linear(out)           # (B, T, V)
        return logits.permute(1, 0, 2)      # (T, B, V)

    def extract_features(self, x):
        """
        Extract perceptual features for loss computation.

        Returns a single feature tensor that combines conv and GRU features
        via concatenation, giving a rich multi-scale representation.

        Input:  (B, 1, M, T)
        Output: (B, T//4, 640 + hidden*2)
        """
        c = self.conv(x)                    # (B, 32, M//4, T//4)
        B, C, M, T = c.shape
        conv_feat = c.permute(0, 3, 1, 2).reshape(B, T, C * M)  # (B, T//4, 640)

        gru_feat, _ = self.gru(conv_feat)   # (B, T//4, hidden*2)

        # Concatenate conv stem and GRU features
        features = torch.cat([conv_feat, gru_feat], dim=-1)  # (B, T//4, 640+hidden*2)
        return features


# =============================================================================
# Multi-Scale STFT Loss (same as model_bandpass.py)
# =============================================================================

class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-resolution spectral loss on the mel spectrogram.
    Spectral convergence + L1 at multiple pooling scales.
    """

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
        loss = 0.0
        for s in self.scales:
            loss = loss + self._loss_at_scale(pred, target, s)
        return loss / len(self.scales)


# =============================================================================
# PerceptionAwareLoss
# =============================================================================

class PerceptionAwareLoss(nn.Module):
    """
    Combined loss: L1 + Multi-Scale STFT + CTC Perceptual.

    The perceptual term uses a FROZEN pretrained CTC encoder to extract
    intermediate features from both the predicted and clean spectrograms,
    then penalises their cosine distance.  This encourages the UNet to
    produce spectrograms that are discriminative for speech recognition,
    not just spectrally accurate.

    Total = l1_w * L1 + mstft_w * MSTFT + percep_w * (1 - cos_sim)

    Args:
        ctc_ckpt_path: Path to pretrained CTC encoder checkpoint (.pt)
        l1_weight:     Weight for L1 loss (default 1.0)
        mstft_weight:  Weight for multi-scale STFT loss (default 1.0)
        percep_weight: Weight for perceptual loss (default 0.1)
        n_mels:        Number of mel bins (default 80)
        hidden:        GRU hidden size in CTC encoder (default 256)
        n_layers:      Number of GRU layers (default 2)
    """

    def __init__(self, ctc_ckpt_path, l1_weight=1.0, mstft_weight=1.0,
                 percep_weight=0.1, n_mels=80, hidden=256, n_layers=2):
        super().__init__()
        self.l1_w     = l1_weight
        self.mstft_w  = mstft_weight
        self.percep_w = percep_weight

        # Multi-scale STFT loss
        self.mstft = MultiScaleSTFTLoss(scales=(1, 2, 4))

        # Load and freeze CTC encoder
        self.ctc_encoder = CTCPerceptualEncoder(
            n_mels=n_mels, hidden=hidden, n_layers=n_layers,
            dropout=0.0,  # no dropout at inference
        )
        self._load_ctc_checkpoint(ctc_ckpt_path)
        self._freeze_ctc()

    def _load_ctc_checkpoint(self, ckpt_path):
        """Load pretrained CTC encoder weights."""
        print(f"  Loading CTC perceptual encoder from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Try to load, allowing missing/unexpected keys (e.g., if checkpoint
        # has extra keys from training)
        missing, unexpected = self.ctc_encoder.load_state_dict(
            state_dict, strict=False)
        if missing:
            print(f"  WARNING: missing keys in CTC encoder: {missing}")
        if unexpected:
            print(f"  INFO: unexpected keys (ignored): {unexpected}")
        print(f"  CTC encoder loaded successfully.")

    def _freeze_ctc(self):
        """Freeze ALL parameters of the CTC encoder."""
        for param in self.ctc_encoder.parameters():
            param.requires_grad = False
        self.ctc_encoder.eval()
        print(f"  CTC encoder frozen "
              f"({sum(p.numel() for p in self.ctc_encoder.parameters()):,} params)")

    def forward(self, pred, target):
        """
        Compute combined loss.

        Args:
            pred:   (B, 1, M, T) - UNet predicted spectrogram
            target: (B, 1, M, T) - clean ground-truth spectrogram

        Returns:
            loss:      scalar tensor (total loss for backward)
            l1_val:    float (L1 component, detached)
            mstft_val: float (MSTFT component, detached)
            percep_val: float (perceptual component, detached)
        """
        # L1 loss
        l1 = F.l1_loss(pred, target)

        # Multi-scale STFT loss
        mstft = self.mstft(pred, target)

        # Perceptual loss: match CTC encoder features
        # Clean features: no_grad, eval mode fine
        # Pred features: need gradients; use train() so cuDNN RNN allows backward
        with torch.no_grad():
            self.ctc_encoder.eval()
            clean_features = self.ctc_encoder.extract_features(target)

        self.ctc_encoder.train()
        pred_features = self.ctc_encoder.extract_features(pred)

        # Cosine similarity loss: 1 - mean(cos_sim)
        # Higher cos_sim = more similar features = lower loss
        perceptual = 1.0 - F.cosine_similarity(
            pred_features, clean_features, dim=-1).mean()

        # Total loss
        loss = self.l1_w * l1 + self.mstft_w * mstft + self.percep_w * perceptual

        return loss, l1.item(), mstft.item(), perceptual.item()


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    # Test UNet
    gen = PowerlineUNet(base_ch=64)
    print(f"PowerlineUNet params: {gen.count_params():,}")

    noisy = torch.randn(4, 1, 80, 98)
    clean = torch.randn(4, 1, 80, 98)
    pred = gen(noisy)
    print(f"Input  shape: {noisy.shape}")
    print(f"Output shape: {pred.shape}")

    # Test CTC perceptual encoder
    ctc = CTCPerceptualEncoder(n_mels=80, hidden=256, n_layers=2, dropout=0.0)
    ctc_params = sum(p.numel() for p in ctc.parameters())
    print(f"\nCTCPerceptualEncoder params: {ctc_params:,}")

    features = ctc.extract_features(noisy)
    print(f"Feature shape: {features.shape}")

    # Test MSTFT loss
    mstft = MultiScaleSTFTLoss()
    mstft_val = mstft(pred, clean)
    print(f"\nMSTFT loss: {mstft_val:.4f}")

    # Test combined (without real checkpoint - just architecture check)
    print("\nArchitecture test passed.")
    print("To test PerceptionAwareLoss, provide a CTC checkpoint path.")
