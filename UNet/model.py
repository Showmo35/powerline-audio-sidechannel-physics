"""
model.py
--------
Spectrogram U-Net for powerline-to-audio reconstruction with Whisper
perceptual loss + multi-scale STFT loss.

Architecture:
  Generator (PowerlineUNet):
    Encoder: 4 conv-blocks that downsample along the time axis
    Bottleneck: 2 residual blocks
    Decoder: 4 up-conv blocks with skip connections from encoder
    Output: same shape as input (1, N_MELS, T)

Input/Output shape: (batch, 1, N_MELS=80, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ──────────────────────────────────────────────────────────

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
    """ConvBlock + MaxPool."""
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


# ── U-Net Generator ─────────────────────────────────────────────────────────

class PowerlineUNet(nn.Module):
    """
    Spectrogram U-Net for audio reconstruction from powerline EM.

    Input : (B, 1, 80, T)  – noisy powerline log-mel spectrogram (Whisper format)
    Output: (B, 1, 80, T)  – reconstructed clean log-mel spectrogram (Whisper format)
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
        return self.out_conv(d1) + x   # global residual: predict correction, not full signal

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Whisper Perceptual Loss ──────────────────────────────────────────────────

class WhisperPerceptualLoss(nn.Module):
    """
    Perceptual loss using frozen Whisper encoder features.

    Passes both predicted and target spectrograms through Whisper's encoder
    and compares intermediate features at multiple layers. This directly
    optimizes for what Whisper needs to decode speech.

    The Whisper encoder expects (B, 80, 3000). Our spectrograms are (B, 1, 80, T)
    where T=98 (~1 sec). We pad to 3000 frames before feeding to the encoder.
    """

    def __init__(self, whisper_model_name='base', feature_layers=(1, 3, 5),
                 device='cuda'):
        super().__init__()
        import whisper as _whisper
        model = _whisper.load_model(whisper_model_name, device=device)
        self.encoder = model.encoder
        self.feature_layers = feature_layers

        # Freeze all Whisper parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def _extract_features(self, mel):
        """
        Extract intermediate features from Whisper encoder.

        mel: (B, 80, 3000) — Whisper-format mel spectrogram
        Returns list of feature tensors at specified layers.
        """
        features = []
        with torch.no_grad():
            # Conv stem
            x = F.gelu(self.encoder.conv1(mel))
            x = F.gelu(self.encoder.conv2(x))
            x = x.permute(0, 2, 1)                              # (B, T, C)
            x = x + self.encoder.positional_embedding[:x.shape[1]]

            # Transformer blocks
            for i, block in enumerate(self.encoder.blocks):
                x = block(x)
                if i in self.feature_layers:
                    features.append(x)

        return features

    def forward(self, pred, target):
        """
        pred:   (B, 1, 80, T)  — UNet output
        target: (B, 1, 80, T)  — clean ground truth
        Returns scalar perceptual loss.
        """
        B, _, M, T = pred.shape

        # Remove channel dim and pad to 3000 frames
        pred_mel   = pred.squeeze(1)                              # (B, 80, T)
        target_mel = target.squeeze(1)

        if T < 3000:
            pred_mel   = F.pad(pred_mel,   (0, 3000 - T))
            target_mel = F.pad(target_mel, (0, 3000 - T))
        else:
            pred_mel   = pred_mel[:, :, :3000]
            target_mel = target_mel[:, :, :3000]

        # Extract features (no grad through Whisper)
        # But we need grad to flow through pred_mel back to UNet
        # So we run target with no_grad and pred with grad enabled
        # through the frozen encoder

        # For target: completely no grad
        with torch.no_grad():
            target_features = self._extract_features(target_mel)

        # For pred: we need gradients to flow back through the input
        # Re-run encoder with grad enabled for the input
        pred_features = self._extract_features_with_grad(pred_mel)

        # L1 loss between features at each layer
        loss = torch.tensor(0.0, device=pred.device)
        for pf, tf in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pf, tf)

        return loss / len(self.feature_layers)

    def _extract_features_with_grad(self, mel):
        """Extract features allowing gradients to flow through mel input."""
        features = []
        # Conv stem — frozen weights but grad flows through input
        x = F.gelu(self.encoder.conv1(mel))
        x = F.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)
        x = x + self.encoder.positional_embedding[:x.shape[1]]

        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.feature_layers:
                features.append(x)

        return features


# ── Multi-scale STFT Loss ───────────────────────────────────────────────────

class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-resolution spectral loss on the mel spectrogram.
    Applies spectral convergence + L1 at multiple scales.
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


# ── Combined Generator Loss ─────────────────────────────────────────────────

class GeneratorLoss(nn.Module):
    """
    Combined loss for the generator:
      - L1 reconstruction loss
      - Multi-scale STFT loss
      - Whisper perceptual loss (frozen encoder features)

    Total = l1_w * L1 + mstft_w * MSTFT + percep_w * WhisperPercep
    """

    def __init__(self, l1_weight=1.0, mstft_weight=1.0, percep_weight=0.1,
                 whisper_loss=None):
        super().__init__()
        self.l1_w     = l1_weight
        self.mstft_w  = mstft_weight
        self.percep_w = percep_weight
        self.mstft    = MultiScaleSTFTLoss(scales=(1, 2, 4))
        self.whisper_loss = whisper_loss  # WhisperPerceptualLoss instance

    def forward(self, pred, target):
        l1    = F.l1_loss(pred, target)
        mstft = self.mstft(pred, target)

        loss = self.l1_w * l1 + self.mstft_w * mstft

        percep = torch.tensor(0.0, device=pred.device)
        if self.whisper_loss is not None:
            percep = self.whisper_loss(pred, target)
            loss = loss + self.percep_w * percep

        return loss, l1.item(), mstft.item(), percep.item()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = PowerlineUNet(base_ch=64)
    print(f"Generator params: {gen.count_params():,}")

    noisy = torch.randn(4, 1, 80, 98)
    clean = torch.randn(4, 1, 80, 98)
    pred = gen(noisy)
    print(f"Input  shape: {noisy.shape}")
    print(f"Output shape: {pred.shape}")

    mstft = MultiScaleSTFTLoss()
    mstft_val = mstft(pred, clean)
    print(f"MSTFT loss: {mstft_val:.4f}")
