"""
Wavelet-based spectral attention module for PyTorch.

This file provides:
- WaveletConv: fixed Morlet wavelet filters implemented as Conv1d (real + imag), producing time-frequency magnitude maps.
- SpectralWaveletAttentionCNN: Large ResNet-DenseNet hybrid CNN with multi-scale wavelet attention at every encoder/decoder level.

Architecture:
- Uses all 50 wavelet scales distributed across 5+ encoder and 5+ decoder stages
- Each stage has residual blocks + dense skip connections
- Wavelet attention applied at every level (encoder and decoder)
- Much larger and more expressive than previous versions

Author: GitHub Copilot (assistant)
"""

import math
from typing import Sequence, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_morlet_kernels(scales: Sequence[float], sr: int, kernel_len: int) -> (np.ndarray, np.ndarray):
    """
    Build Morlet-like wavelet kernels (real and imag) for the requested scales.

    This implementation avoids SciPy's `morlet2` dependency by constructing a
    complex Morlet (plane wave modulated by a Gaussian) for each scale. The
    function returns two arrays (real, imag) shaped (n_scales, kernel_len),
    dtype float32.
    """
    ks = int(kernel_len)
    # time axis in samples centered at zero
    t = (np.arange(ks) - (ks // 2)).astype(np.float32)
    kernels_real = []
    kernels_imag = []

    # central angular frequency for the mother Morlet; common choice ~5
    w0 = 5.0

    for s in scales:
        s_f = float(s)
        if s_f <= 0:
            s_f = 1.0

        # Create a Morlet-like wavelet in sample units. We interpret `s` as
        # a scale (spread) in samples. The oscillation is given by w0 * t / s.
        # Multiply the complex exponential by a Gaussian envelope with std = s.
        wave = np.exp(1j * (w0 * t / s_f)) * np.exp(-0.5 * (t ** 2) / (s_f ** 2))

        # Remove mean (admissibility) and normalize to unit L2 norm
        wave = wave - np.mean(wave)
        norm = np.sqrt(np.sum(np.abs(wave) ** 2)) + 1e-12
        wave = wave / norm

        kernels_real.append(np.real(wave))
        kernels_imag.append(np.imag(wave))

    kernels_real = np.stack(kernels_real).astype(np.float32)
    kernels_imag = np.stack(kernels_imag).astype(np.float32)
    return kernels_real, kernels_imag


class WaveletConv(nn.Module):
    """Apply a bank of Morlet wavelets via Conv1d and return magnitudes.

    Input: (batch, 1, T)
    Output: (batch, n_scales, T) (magnitude of complex wavelet coefficients)
    """

    def __init__(self, scales: Sequence[float], sr: int, kernel_len: int = 257, padding: str = 'same'):
        super().__init__()
        self.scales = list(scales)
        self.sr = sr
        self.kernel_len = kernel_len
        # Build kernels
        real_k, imag_k = make_morlet_kernels(self.scales, sr, kernel_len)
        # Convert to torch tensors shaped (out_channels, in_channels=1, kernel_len)
        self.register_buffer('kernels_real', torch.from_numpy(real_k)[:, None, :])
        self.register_buffer('kernels_imag', torch.from_numpy(imag_k)[:, None, :])
        # Kernels fixed (no grad)
        # Determine padding
        if padding == 'same':
            self.pad = kernel_len // 2
        else:
            self.pad = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        real = F.conv1d(F.pad(x, (self.pad, self.pad)), self.kernels_real, bias=None)
        imag = F.conv1d(F.pad(x, (self.pad, self.pad)), self.kernels_imag, bias=None)
        mag = torch.sqrt(real * real + imag * imag + 1e-12)
        return mag


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class DenseBlock(nn.Module):
    """Dense block with concatenation of all layer outputs."""
    def __init__(self, in_ch, growth_rate=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(in_ch + i * growth_rate),
                nn.ReLU(),
                nn.Conv1d(in_ch + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            ))
        self.num_layers = num_layers
        self.growth_rate = growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class ConvBlock(nn.Module):
    """Basic convolution block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SpectralWaveletAttentionCNN(nn.Module):
    """
    Large ResNet-DenseNet hybrid with multi-scale wavelet attention.
    
    Architecture:
    - 50 wavelet scales computed once and reused at all levels
    - 6-stage encoder with residual + dense blocks at each stage
    - 6-stage decoder with dense skip connections
    - Wavelet attention applied at every encoder and decoder stage (12 attention modules total)
    - Base channels: 64, progressively increasing to 1024 at bottleneck
    
    Total depth: ~50+ layers with residual connections and dense blocks
    
    Example:
      model = SpectralWaveletAttentionCNN(sample_rate=200000, base_channels=64)
      y = model(x)  # x shape (B, 1, T)
    """

    def __init__(self, sample_rate: int = 200000, base_channels: int = 64, num_scales: int = 50):
        super().__init__()
        self.sr = sample_rate
        self.base = base_channels
        self.num_scales = num_scales

        # Generate 50 scales spanning from coarse (low freq) to fine (high freq)
        # Logarithmic spacing for better frequency coverage
        self.scales = list(np.logspace(0.5, 3.5, num=num_scales))
        
        # Single wavelet conv that computes all 50 scales
        self.wavelet_conv = WaveletConv(scales=self.scales, sr=self.sr, kernel_len=257)

        # Channel progression: 64 -> 128 -> 256 -> 512 -> 768 -> 1024
        ch = [self.base, self.base*2, self.base*4, self.base*8, self.base*12, self.base*16]
        
        # Dense block growth rate
        growth = 32
        
        # === ENCODER ===
        # Stage 1: Input processing
        self.enc1_conv = ConvBlock(1, ch[0])
        self.enc1_res = ResidualBlock(ch[0])
        self.enc1_dense = DenseBlock(ch[0], growth_rate=growth, num_layers=3)
        enc1_out_ch = ch[0] + 3 * growth
        self.enc1_transition = ConvBlock(enc1_out_ch, ch[0])
        self.attn1_enc = self._make_attention_module(self.num_scales, ch[0])
        
        # Stage 2
        self.pool1 = nn.AvgPool1d(2)
        self.enc2_conv = ConvBlock(ch[0], ch[1])
        self.enc2_res = ResidualBlock(ch[1])
        self.enc2_dense = DenseBlock(ch[1], growth_rate=growth, num_layers=4)
        enc2_out_ch = ch[1] + 4 * growth
        self.enc2_transition = ConvBlock(enc2_out_ch, ch[1])
        self.attn2_enc = self._make_attention_module(self.num_scales, ch[1])
        
        # Stage 3
        self.pool2 = nn.AvgPool1d(2)
        self.enc3_conv = ConvBlock(ch[1], ch[2])
        self.enc3_res1 = ResidualBlock(ch[2])
        self.enc3_res2 = ResidualBlock(ch[2])
        self.enc3_dense = DenseBlock(ch[2], growth_rate=growth, num_layers=4)
        enc3_out_ch = ch[2] + 4 * growth
        self.enc3_transition = ConvBlock(enc3_out_ch, ch[2])
        self.attn3_enc = self._make_attention_module(self.num_scales, ch[2])
        
        # Stage 4
        self.pool3 = nn.AvgPool1d(2)
        self.enc4_conv = ConvBlock(ch[2], ch[3])
        self.enc4_res1 = ResidualBlock(ch[3])
        self.enc4_res2 = ResidualBlock(ch[3])
        self.enc4_dense = DenseBlock(ch[3], growth_rate=growth, num_layers=5)
        enc4_out_ch = ch[3] + 5 * growth
        self.enc4_transition = ConvBlock(enc4_out_ch, ch[3])
        self.attn4_enc = self._make_attention_module(self.num_scales, ch[3])
        
        # Stage 5
        self.pool4 = nn.AvgPool1d(2)
        self.enc5_conv = ConvBlock(ch[3], ch[4])
        self.enc5_res1 = ResidualBlock(ch[4])
        self.enc5_res2 = ResidualBlock(ch[4])
        self.enc5_res3 = ResidualBlock(ch[4])
        self.enc5_dense = DenseBlock(ch[4], growth_rate=growth, num_layers=5)
        enc5_out_ch = ch[4] + 5 * growth
        self.enc5_transition = ConvBlock(enc5_out_ch, ch[4])
        self.attn5_enc = self._make_attention_module(self.num_scales, ch[4])
        
        # Stage 6 (Bottleneck)
        self.pool5 = nn.AvgPool1d(2)
        self.bottleneck_conv = ConvBlock(ch[4], ch[5])
        self.bottleneck_res1 = ResidualBlock(ch[5])
        self.bottleneck_res2 = ResidualBlock(ch[5])
        self.bottleneck_res3 = ResidualBlock(ch[5])
        self.bottleneck_dense = DenseBlock(ch[5], growth_rate=growth, num_layers=6)
        bn_out_ch = ch[5] + 6 * growth
        self.bottleneck_transition = ConvBlock(bn_out_ch, ch[5])
        self.attn_bottleneck = self._make_attention_module(self.num_scales, ch[5])
        
        # === DECODER ===
        # Stage 6 (Decoder start)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec6_conv = ConvBlock(ch[5] + ch[4], ch[4])  # +skip from enc5
        self.dec6_res1 = ResidualBlock(ch[4])
        self.dec6_res2 = ResidualBlock(ch[4])
        self.dec6_dense = DenseBlock(ch[4], growth_rate=growth, num_layers=4)
        dec6_out_ch = ch[4] + 4 * growth
        self.dec6_transition = ConvBlock(dec6_out_ch, ch[4])
        self.attn6_dec = self._make_attention_module(self.num_scales, ch[4])
        
        # Stage 5
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec5_conv = ConvBlock(ch[4] + ch[3], ch[3])  # +skip from enc4
        self.dec5_res1 = ResidualBlock(ch[3])
        self.dec5_res2 = ResidualBlock(ch[3])
        self.dec5_dense = DenseBlock(ch[3], growth_rate=growth, num_layers=4)
        dec5_out_ch = ch[3] + 4 * growth
        self.dec5_transition = ConvBlock(dec5_out_ch, ch[3])
        self.attn5_dec = self._make_attention_module(self.num_scales, ch[3])
        
        # Stage 4
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4_conv = ConvBlock(ch[3] + ch[2], ch[2])  # +skip from enc3
        self.dec4_res1 = ResidualBlock(ch[2])
        self.dec4_res2 = ResidualBlock(ch[2])
        self.dec4_dense = DenseBlock(ch[2], growth_rate=growth, num_layers=3)
        dec4_out_ch = ch[2] + 3 * growth
        self.dec4_transition = ConvBlock(dec4_out_ch, ch[2])
        self.attn4_dec = self._make_attention_module(self.num_scales, ch[2])
        
        # Stage 3
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3_conv = ConvBlock(ch[2] + ch[1], ch[1])  # +skip from enc2
        self.dec3_res = ResidualBlock(ch[1])
        self.dec3_dense = DenseBlock(ch[1], growth_rate=growth, num_layers=3)
        dec3_out_ch = ch[1] + 3 * growth
        self.dec3_transition = ConvBlock(dec3_out_ch, ch[1])
        self.attn3_dec = self._make_attention_module(self.num_scales, ch[1])
        
        # Stage 2
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2_conv = ConvBlock(ch[1] + ch[0], ch[0])  # +skip from enc1
        self.dec2_res = ResidualBlock(ch[0])
        self.dec2_dense = DenseBlock(ch[0], growth_rate=growth, num_layers=3)
        dec2_out_ch = ch[0] + 3 * growth
        self.dec2_transition = ConvBlock(dec2_out_ch, ch[0])
        self.attn2_dec = self._make_attention_module(self.num_scales, ch[0])
        
        # Final output
        self.final_conv1 = ConvBlock(ch[0], ch[0]//2)
        self.final_conv2 = nn.Conv1d(ch[0]//2, 1, kernel_size=1)

    def _make_attention_module(self, n_scales: int, n_channels: int):
        """
        Create an attention module that projects wavelet scales to per-time attention weights.
        Uses a deeper MLP for better expressiveness.
        """
        return nn.Sequential(
            nn.Conv1d(n_scales, n_scales, kernel_size=1),
            nn.BatchNorm1d(n_scales),
            nn.ReLU(),
            nn.Conv1d(n_scales, n_scales // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_scales // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _apply_attention(self, features: torch.Tensor, wavelet_map: torch.Tensor, 
                         attn_module: nn.Module) -> torch.Tensor:
        """
        Apply wavelet-based attention to features.
        - features: (B, C, T)
        - wavelet_map: (B, n_scales, T_original) 
        - attn_module: MLP that maps scales -> attention weights
        
        Returns: (B, C, T) attention-gated features
        """
        B, C, T = features.shape
        
        # Resample wavelet map to match feature temporal length
        if wavelet_map.shape[-1] != T:
            wave = F.interpolate(wavelet_map, size=T, mode='linear', align_corners=False)
        else:
            wave = wavelet_map
            
        # Normalize wavelet magnitudes per scale
        wave = wave / (wave.mean(dim=-1, keepdim=True) + 1e-12)
        
        # Project scales to attention weights: (B, n_scales, T) -> (B, 1, T)
        attn_weights = attn_module(wave)
        
        # Apply attention: multiply features by per-time weights
        return features * attn_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: (B, 1, T)
        Output: (B, 1, T)
        """
        # Compute all 50 wavelet scales once at full resolution
        wavelet_map = self.wavelet_conv(x)  # (B, 50, T)
        
        # === ENCODER ===
        # Stage 1
        e1 = self.enc1_conv(x)
        e1 = self.enc1_res(e1)
        e1 = self.enc1_dense(e1)
        e1 = self.enc1_transition(e1)
        e1 = self._apply_attention(e1, wavelet_map, self.attn1_enc)
        
        # Stage 2
        e2 = self.pool1(e1)
        e2 = self.enc2_conv(e2)
        e2 = self.enc2_res(e2)
        e2 = self.enc2_dense(e2)
        e2 = self.enc2_transition(e2)
        e2 = self._apply_attention(e2, wavelet_map, self.attn2_enc)
        
        # Stage 3
        e3 = self.pool2(e2)
        e3 = self.enc3_conv(e3)
        e3 = self.enc3_res1(e3)
        e3 = self.enc3_res2(e3)
        e3 = self.enc3_dense(e3)
        e3 = self.enc3_transition(e3)
        e3 = self._apply_attention(e3, wavelet_map, self.attn3_enc)
        
        # Stage 4
        e4 = self.pool3(e3)
        e4 = self.enc4_conv(e4)
        e4 = self.enc4_res1(e4)
        e4 = self.enc4_res2(e4)
        e4 = self.enc4_dense(e4)
        e4 = self.enc4_transition(e4)
        e4 = self._apply_attention(e4, wavelet_map, self.attn4_enc)
        
        # Stage 5
        e5 = self.pool4(e4)
        e5 = self.enc5_conv(e5)
        e5 = self.enc5_res1(e5)
        e5 = self.enc5_res2(e5)
        e5 = self.enc5_res3(e5)
        e5 = self.enc5_dense(e5)
        e5 = self.enc5_transition(e5)
        e5 = self._apply_attention(e5, wavelet_map, self.attn5_enc)
        
        # Stage 6 (Bottleneck)
        e6 = self.pool5(e5)
        e6 = self.bottleneck_conv(e6)
        e6 = self.bottleneck_res1(e6)
        e6 = self.bottleneck_res2(e6)
        e6 = self.bottleneck_res3(e6)
        e6 = self.bottleneck_dense(e6)
        e6 = self.bottleneck_transition(e6)
        e6 = self._apply_attention(e6, wavelet_map, self.attn_bottleneck)
        
        # === DECODER ===
        # Stage 6
        d6 = self.up5(e6)
        d6 = torch.cat([d6, e5], dim=1)
        d6 = self.dec6_conv(d6)
        d6 = self.dec6_res1(d6)
        d6 = self.dec6_res2(d6)
        d6 = self.dec6_dense(d6)
        d6 = self.dec6_transition(d6)
        d6 = self._apply_attention(d6, wavelet_map, self.attn6_dec)
        
        # Stage 5
        d5 = self.up4(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5_conv(d5)
        d5 = self.dec5_res1(d5)
        d5 = self.dec5_res2(d5)
        d5 = self.dec5_dense(d5)
        d5 = self.dec5_transition(d5)
        d5 = self._apply_attention(d5, wavelet_map, self.attn5_dec)
        
        # Stage 4
        d4 = self.up3(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4_conv(d4)
        d4 = self.dec4_res1(d4)
        d4 = self.dec4_res2(d4)
        d4 = self.dec4_dense(d4)
        d4 = self.dec4_transition(d4)
        d4 = self._apply_attention(d4, wavelet_map, self.attn4_dec)
        
        # Stage 3
        d3 = self.up2(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3_conv(d3)
        d3 = self.dec3_res(d3)
        d3 = self.dec3_dense(d3)
        d3 = self.dec3_transition(d3)
        d3 = self._apply_attention(d3, wavelet_map, self.attn3_dec)
        
        # Stage 2
        d2 = self.up1(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_conv(d2)
        d2 = self.dec2_res(d2)
        d2 = self.dec2_dense(d2)
        d2 = self.dec2_transition(d2)
        d2 = self._apply_attention(d2, wavelet_map, self.attn2_dec)
        
        # Final output
        out = self.final_conv1(d2)
        out = self.final_conv2(out)
        out = out.squeeze(1)  # (B, 1, T) -> (B, T)
        
        return out


if __name__ == '__main__':
    # Smoke test for the large wavelet attention model
    print("Testing large ResNet-DenseNet wavelet attention model...")
    model = SpectralWaveletAttentionCNN(sample_rate=200000, base_channels=64, num_scales=50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB (float32)")
    
    # Test forward pass
    x = torch.randn(2, 1, 200000)  # 2 samples, 1 second at 200kHz
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print("\n✓ Forward pass successful!")
    print(f"✓ Model has {len([m for m in model.modules() if 'attn' in str(type(m)).lower()])} attention modules")

