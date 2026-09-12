"""
Multi-stage powerline-to-audio reconstruction pipeline.

This file implements a modular 3-stage system:
  Stage A: Powerline → USB (with latent features)
  Stage B: USB + Latent → Audio
  Stage C: End-to-end joint training

Architecture Overview:
- Stage A uses the SpectralWaveletAttentionCNN as encoder-predictor
- Stage A exposes both USB predictions and bottleneck latent features
- Stage B fuses USB signal + PL latents to reconstruct audio waveform
- Stage C enables end-to-end training with multi-task losses

Author: GitHub Copilot (assistant)
Date: November 19, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from wavelet_attention import (
    SpectralWaveletAttentionCNN,
    WaveletConv,
    ResidualBlock,
    DenseBlock,
    ConvBlock
)


# ==============================================================================
# STAGE A: Powerline → USB (Enhanced with Latent Feature Extraction)
# ==============================================================================

class PowerlineToUSBEncoder(nn.Module):
    """
    Stage A: Powerline → USB prediction with latent feature extraction.
    
    This wraps the SpectralWaveletAttentionCNN and exposes:
    1. USB waveform prediction (primary output)
    2. Bottleneck latent features (for Stage B fusion)
    3. Multi-scale encoder features (optional for skip connections)
    
    Architecture:
    - Input: (B, 1, T_pl) powerline waveform at powerline_sample_rate
    - Outputs:
      - usb_pred: (B, T_pl) predicted USB waveform at same sample rate
      - latent: (B, C_latent, T_latent) bottleneck features
      - encoder_features: list of encoder outputs for skip connections
    """
    
    def __init__(
        self,
        powerline_sample_rate: int,
        base_channels: int = 64,
        num_scales: int = 50,
        latent_projection_dim: int = 256
    ):
        super().__init__()
        
        self.sr = powerline_sample_rate
        self.base = base_channels
        self.latent_dim = latent_projection_dim
        
        # Core wavelet-attention CNN (from wavelet_attention.py)
        self.wavelet_cnn = SpectralWaveletAttentionCNN(
            sample_rate=powerline_sample_rate,
            base_channels=base_channels,
            num_scales=num_scales
        )
        
        # Latent projection head: maps bottleneck to compact latent space
        # Bottleneck has base*16 channels after the transition layer
        bottleneck_channels = base_channels * 16
        self.latent_projection = nn.Sequential(
            nn.Conv1d(bottleneck_channels, latent_projection_dim * 2, kernel_size=1),
            nn.BatchNorm1d(latent_projection_dim * 2),
            nn.ReLU(),
            nn.Conv1d(latent_projection_dim * 2, latent_projection_dim, kernel_size=1),
            nn.BatchNorm1d(latent_projection_dim)
        )
        
    def forward(
        self, 
        x_pl: torch.Tensor,
        return_encoder_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage A.
        
        Args:
            x_pl: (B, 1, T_pl) powerline input
            return_encoder_features: if True, return intermediate encoder outputs
            
        Returns:
            Dictionary containing:
                - 'usb_pred': (B, T_pl) predicted USB waveform
                - 'latent': (B, latent_dim, T_latent) bottleneck latent features
                - 'encoder_features': (optional) list of encoder stage outputs
        """
        # We need to manually extract intermediate features from the wavelet CNN
        # Let's do a custom forward pass through the wavelet CNN's encoder
        
        model = self.wavelet_cnn
        
        # Compute wavelet map (same as in original forward)
        wavelet_map = model.wavelet_conv(x_pl)
        
        # === ENCODER (manual forward to capture intermediates) ===
        e1 = model.enc1_conv(x_pl)
        e1 = model.enc1_res(e1)
        e1 = model.enc1_dense(e1)
        e1 = model.enc1_transition(e1)
        e1 = model._apply_attention(e1, wavelet_map, model.attn1_enc)
        
        e2 = model.pool1(e1)
        e2 = model.enc2_conv(e2)
        e2 = model.enc2_res(e2)
        e2 = model.enc2_dense(e2)
        e2 = model.enc2_transition(e2)
        e2 = model._apply_attention(e2, wavelet_map, model.attn2_enc)
        
        e3 = model.pool2(e2)
        e3 = model.enc3_conv(e3)
        e3 = model.enc3_res1(e3)
        e3 = model.enc3_res2(e3)
        e3 = model.enc3_dense(e3)
        e3 = model.enc3_transition(e3)
        e3 = model._apply_attention(e3, wavelet_map, model.attn3_enc)
        
        e4 = model.pool3(e3)
        e4 = model.enc4_conv(e4)
        e4 = model.enc4_res1(e4)
        e4 = model.enc4_res2(e4)
        e4 = model.enc4_dense(e4)
        e4 = model.enc4_transition(e4)
        e4 = model._apply_attention(e4, wavelet_map, model.attn4_enc)
        
        e5 = model.pool4(e4)
        e5 = model.enc5_conv(e5)
        e5 = model.enc5_res1(e5)
        e5 = model.enc5_res2(e5)
        e5 = model.enc5_res3(e5)
        e5 = model.enc5_dense(e5)
        e5 = model.enc5_transition(e5)
        e5 = model._apply_attention(e5, wavelet_map, model.attn5_enc)
        
        # Bottleneck (this is our latent representation source)
        e6 = model.pool5(e5)
        e6 = model.bottleneck_conv(e6)
        e6 = model.bottleneck_res1(e6)
        e6 = model.bottleneck_res2(e6)
        e6 = model.bottleneck_res3(e6)
        e6 = model.bottleneck_dense(e6)
        e6 = model.bottleneck_transition(e6)
        e6 = model._apply_attention(e6, wavelet_map, model.attn_bottleneck)
        
        # === DECODER (generate USB prediction) ===
        d6 = model.up5(e6)
        d6 = torch.cat([d6, e5], dim=1)
        d6 = model.dec6_conv(d6)
        d6 = model.dec6_res1(d6)
        d6 = model.dec6_res2(d6)
        d6 = model.dec6_dense(d6)
        d6 = model.dec6_transition(d6)
        d6 = model._apply_attention(d6, wavelet_map, model.attn6_dec)
        
        d5 = model.up4(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = model.dec5_conv(d5)
        d5 = model.dec5_res1(d5)
        d5 = model.dec5_res2(d5)
        d5 = model.dec5_dense(d5)
        d5 = model.dec5_transition(d5)
        d5 = model._apply_attention(d5, wavelet_map, model.attn5_dec)
        
        d4 = model.up3(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = model.dec4_conv(d4)
        d4 = model.dec4_res1(d4)
        d4 = model.dec4_res2(d4)
        d4 = model.dec4_dense(d4)
        d4 = model.dec4_transition(d4)
        d4 = model._apply_attention(d4, wavelet_map, model.attn4_dec)
        
        d3 = model.up2(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = model.dec3_conv(d3)
        d3 = model.dec3_res(d3)
        d3 = model.dec3_dense(d3)
        d3 = model.dec3_transition(d3)
        d3 = model._apply_attention(d3, wavelet_map, model.attn3_dec)
        
        d2 = model.up1(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = model.dec2_conv(d2)
        d2 = model.dec2_res(d2)
        d2 = model.dec2_dense(d2)
        d2 = model.dec2_transition(d2)
        d2 = model._apply_attention(d2, wavelet_map, model.attn2_dec)
        
        # Final USB prediction
        out = model.final_conv1(d2)
        out = model.final_conv2(out)
        usb_pred = out.squeeze(1)  # (B, 1, T) -> (B, T)
        
        # Project bottleneck e6 to latent space
        latent = self.latent_projection(e6)  # (B, latent_dim, T_latent)
        
        # Prepare output dictionary
        outputs = {
            'usb_pred': usb_pred,
            'latent': latent
        }
        
        if return_encoder_features:
            outputs['encoder_features'] = [e1, e2, e3, e4, e5, e6]
            
        return outputs


# ==============================================================================
# STAGE B: USB + Latent → Audio Reconstruction
# ==============================================================================

class USBToAudioDecoder(nn.Module):
    """
    Stage B: USB signal + powerline latents → Audio waveform.
    
    This is a dual-branch fusion decoder:
    1. USB branch: processes predicted/true USB signal
    2. Latent branch: processes powerline latent features from Stage A
    3. Fusion: combines both branches with attention
    4. Audio decoder: generates audio waveform
    
    Architecture:
    - Input:
      - usb: (B, 1, T_usb) USB waveform at usb_sample_rate
      - latent: (B, C_latent, T_latent) powerline latent features
    - Output:
      - audio: (B, T_audio) audio waveform at audio_sample_rate
    """
    
    def __init__(
        self,
        usb_sample_rate: int,
        audio_sample_rate: int,
        latent_dim: int = 256,
        base_channels: int = 64,
        use_dilated_convs: bool = True
    ):
        super().__init__()
        
        self.usb_sr = usb_sample_rate
        self.audio_sr = audio_sample_rate
        self.rate_ratio = usb_sample_rate // audio_sample_rate  # e.g., 200k/48k ≈ 4.17
        self.base = base_channels
        
        # === USB Encoder Branch ===
        # Processes USB signal to extract features
        self.usb_conv1 = ConvBlock(1, base_channels)
        self.usb_res1 = ResidualBlock(base_channels)
        self.usb_pool1 = nn.AvgPool1d(2)
        
        self.usb_conv2 = ConvBlock(base_channels, base_channels * 2)
        self.usb_res2 = ResidualBlock(base_channels * 2)
        self.usb_pool2 = nn.AvgPool1d(2)
        
        self.usb_conv3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.usb_res3 = ResidualBlock(base_channels * 4)
        
        # === Latent Processing Branch ===
        # Projects and processes powerline latents
        self.latent_conv1 = ConvBlock(latent_dim, base_channels * 2)
        self.latent_res1 = ResidualBlock(base_channels * 2)
        self.latent_conv2 = ConvBlock(base_channels * 2, base_channels * 4)
        self.latent_res2 = ResidualBlock(base_channels * 4)
        
        # === Cross-modal Fusion ===
        # Combine USB and latent features with attention
        fusion_channels = base_channels * 4
        self.fusion_attn = nn.Sequential(
            nn.Conv1d(fusion_channels * 2, fusion_channels, kernel_size=1),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(),
            nn.Conv1d(fusion_channels, fusion_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = ConvBlock(fusion_channels * 2, fusion_channels)
        
        # === Audio Decoder (U-Net style with TCN) ===
        # Note: Need to downsample from USB sample rate to audio sample rate
        # USB encoding reduces by 4x (two pool layers), so fused features are at T_usb/4
        # Need final output at audio_sr, which is ~9x lower than usb_sr
        # Strategy: Upsample 4x to get back to T_usb, then downsample by rate_ratio
        
        # Option 1: Use dilated convolutions for large receptive field
        if use_dilated_convs:
            self.decoder = nn.Sequential(
                # Dilated TCN blocks (at T_usb/4 resolution)
                self._make_tcn_block(fusion_channels, base_channels * 4, dilation=1),
                self._make_tcn_block(base_channels * 4, base_channels * 4, dilation=2),
                self._make_tcn_block(base_channels * 4, base_channels * 4, dilation=4),
                self._make_tcn_block(base_channels * 4, base_channels * 4, dilation=8),
                self._make_tcn_block(base_channels * 4, base_channels * 4, dilation=16),
                
                # Upsampling path to restore to T_usb resolution
                ConvBlock(base_channels * 4, base_channels * 2),
                ResidualBlock(base_channels * 2),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Now at T_usb/2
                
                ConvBlock(base_channels * 2, base_channels),
                ResidualBlock(base_channels),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Now at T_usb
                
                # Final projection to audio (still at T_usb resolution)
                ConvBlock(base_channels, base_channels // 2),
                nn.Conv1d(base_channels // 2, 1, kernel_size=1)
                # Note: Downsampling to audio_sr happens in forward() after decoder
            )
        else:
            # Option 2: Standard U-Net decoder
            self.decoder = nn.Sequential(
                ConvBlock(fusion_channels, base_channels * 4),
                ResidualBlock(base_channels * 4),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Now at T_usb/2
                
                ConvBlock(base_channels * 4, base_channels * 2),
                ResidualBlock(base_channels * 2),
                nn.Upsample(scale_factor=2, mode='nearest'),  # Now at T_usb
                
                ConvBlock(base_channels * 2, base_channels),
                ResidualBlock(base_channels),
                
                ConvBlock(base_channels, base_channels // 2),
                nn.Conv1d(base_channels // 2, 1, kernel_size=1)
                # Note: Downsampling to audio_sr happens in forward() after decoder
            )
        
    def _make_tcn_block(self, in_ch: int, out_ch: int, dilation: int) -> nn.Module:
        """Create a dilated temporal convolutional block."""
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(
        self,
        usb: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Stage B.
        
        Args:
            usb: (B, 1, T_usb) USB waveform (can be ground truth or predicted)
            latent: (B, latent_dim, T_latent) powerline latent features
            
        Returns:
            audio: (B, T_audio) reconstructed audio waveform
        """
        # === USB Branch ===
        u1 = self.usb_conv1(usb)
        u1 = self.usb_res1(u1)
        u1 = self.usb_pool1(u1)
        
        u2 = self.usb_conv2(u1)
        u2 = self.usb_res2(u2)
        u2 = self.usb_pool2(u2)
        
        u3 = self.usb_conv3(u2)
        u3 = self.usb_res3(u3)  # (B, base*4, T_usb/4)
        
        # === Latent Branch ===
        l1 = self.latent_conv1(latent)
        l1 = self.latent_res1(l1)
        l2 = self.latent_conv2(l1)
        l2 = self.latent_res2(l2)  # (B, base*4, T_latent)
        
        # === Temporal Alignment ===
        # Resample latent features to match USB temporal dimension
        T_target = u3.shape[-1]
        if l2.shape[-1] != T_target:
            l2_aligned = F.interpolate(l2, size=T_target, mode='linear', align_corners=False)
        else:
            l2_aligned = l2
        
        # === Fusion with Attention ===
        # Concatenate USB and latent features
        fused_raw = torch.cat([u3, l2_aligned], dim=1)  # (B, base*8, T)
        
        # Compute attention weights
        attn_weights = self.fusion_attn(fused_raw)  # (B, base*4, T)
        
        # Apply attention to latent branch
        l2_gated = l2_aligned * attn_weights
        
        # Combine gated latent with USB features
        fused = torch.cat([u3, l2_gated], dim=1)  # (B, base*8, T)
        fused = self.fusion_conv(fused)  # (B, base*4, T)
        
        # === Audio Decoder ===
        audio = self.decoder(fused)  # (B, 1, T_usb) - still at USB sample rate
        
        # === Downsample to audio sample rate ===
        # Use interpolate to resample from usb_sr to audio_sr
        # Target length: T_audio = T_usb * (audio_sr / usb_sr)
        T_usb_full = usb.shape[-1]
        T_audio_target = int(T_usb_full * self.audio_sr / self.usb_sr)
        
        # Resample audio to target sample rate
        audio_resampled = F.interpolate(
            audio, 
            size=T_audio_target, 
            mode='linear', 
            align_corners=False
        )  # (B, 1, T_audio)
        
        audio_resampled = audio_resampled.squeeze(1)  # (B, T_audio)
        
        return audio_resampled


# ==============================================================================
# STAGE C: End-to-End Pipeline with Multi-Task Training
# ==============================================================================

class PowerlineToAudioPipeline(nn.Module):
    """
    Complete end-to-end pipeline: Powerline → USB → Audio.
    
    This combines Stage A and Stage B into a single trainable model
    with support for multi-task losses and end-to-end optimization.
    
    Training modes:
    1. Stage A only: train powerline → USB
    2. Stage B only: train USB → audio (with frozen Stage A)
    3. End-to-end: jointly optimize both stages with combined loss
    
    Architecture:
    - Input: (B, 1, T_pl) powerline waveform at powerline_sr
    - Outputs:
      - usb_pred: (B, T_pl) predicted USB at usb_sr
      - audio_pred: (B, T_audio) predicted audio at audio_sr
      - latent: (B, latent_dim, T_latent) intermediate latent features
    """
    
    def __init__(
        self,
        powerline_sr: int,
        usb_sr: int,
        audio_sr: int,
        base_channels_stageA: int = 64,
        base_channels_stageB: int = 64,
        latent_dim: int = 256,
        num_wavelet_scales: int = 50,
        use_dilated_convs: bool = True
    ):
        super().__init__()
        
        self.powerline_sr = powerline_sr
        self.usb_sr = usb_sr
        self.audio_sr = audio_sr
        
        # === Stage A: Powerline → USB ===
        self.stage_a = PowerlineToUSBEncoder(
            powerline_sample_rate=powerline_sr,
            base_channels=base_channels_stageA,
            num_scales=num_wavelet_scales,
            latent_projection_dim=latent_dim
        )
        
        # === Stage B: USB + Latent → Audio ===
        self.stage_b = USBToAudioDecoder(
            usb_sample_rate=usb_sr,
            audio_sample_rate=audio_sr,
            latent_dim=latent_dim,
            base_channels=base_channels_stageB,
            use_dilated_convs=use_dilated_convs
        )
        
    def forward(
        self,
        x_powerline: torch.Tensor,
        mode: str = 'end_to_end'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.
        
        Args:
            x_powerline: (B, 1, T_pl) powerline input waveform
            mode: training mode
                - 'stage_a': only compute USB prediction
                - 'stage_b': use ground truth USB (must be provided separately)
                - 'end_to_end': full pipeline (default)
                
        Returns:
            Dictionary with keys:
                - 'usb_pred': (B, T_pl) predicted USB waveform
                - 'audio_pred': (B, T_audio) predicted audio waveform
                - 'latent': (B, latent_dim, T_latent) latent features
        """
        # === Stage A: Powerline → USB + Latent ===
        stage_a_outputs = self.stage_a(x_powerline)
        usb_pred = stage_a_outputs['usb_pred']
        latent = stage_a_outputs['latent']
        
        if mode == 'stage_a':
            # Only return Stage A outputs
            return {
                'usb_pred': usb_pred,
                'latent': latent
            }
        
        # === Stage B: USB + Latent → Audio ===
        # Add channel dimension to USB for conv input
        usb_input = usb_pred.unsqueeze(1)  # (B, T) -> (B, 1, T)
        audio_pred = self.stage_b(usb_input, latent)
        
        return {
            'usb_pred': usb_pred,
            'audio_pred': audio_pred,
            'latent': latent
        }
    
    def forward_stage_b_only(
        self,
        usb_true: torch.Tensor,
        x_powerline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Stage B only, using ground truth USB.
        
        Useful for training Stage B in isolation with teacher forcing.
        
        Args:
            usb_true: (B, T_usb) or (B, 1, T_usb) ground truth USB signal
            x_powerline: (B, 1, T_pl) powerline input (to extract latent)
                        If None, will use zero latent features
                        
        Returns:
            audio_pred: (B, T_audio) predicted audio
        """
        if x_powerline is not None:
            # Extract latent from powerline
            with torch.no_grad():  # Don't update Stage A during Stage B training
                stage_a_outputs = self.stage_a(x_powerline)
                latent = stage_a_outputs['latent']
        else:
            # Use zero latent (baseline: pure USB → audio)
            B = usb_true.shape[0]
            T_latent = usb_true.shape[-1] // 32  # Approximate latent temporal dimension
            device = usb_true.device
            latent = torch.zeros(B, 256, T_latent, device=device)
        
        # Ensure USB has channel dimension
        if usb_true.ndim == 2:
            usb_true = usb_true.unsqueeze(1)
            
        audio_pred = self.stage_b(usb_true, latent)
        return audio_pred
    
    def freeze_stage_a(self):
        """Freeze Stage A parameters (for Stage B-only training)."""
        for param in self.stage_a.parameters():
            param.requires_grad = False
            
    def unfreeze_stage_a(self):
        """Unfreeze Stage A parameters (for end-to-end training)."""
        for param in self.stage_a.parameters():
            param.requires_grad = True
            
    def freeze_stage_b(self):
        """Freeze Stage B parameters (for Stage A-only training)."""
        for param in self.stage_b.parameters():
            param.requires_grad = False
            
    def unfreeze_stage_b(self):
        """Unfreeze Stage B parameters."""
        for param in self.stage_b.parameters():
            param.requires_grad = True


# ==============================================================================
# Multi-Task Loss Functions
# ==============================================================================

class MultiStageLoss(nn.Module):
    """
    Combined loss for end-to-end training.
    
    Components:
    1. USB reconstruction loss (L1 waveform + optional frequency domain)
    2. Audio reconstruction loss (L1 + multi-resolution STFT)
    3. Optional perceptual loss
    
    Loss weighting:
        L_total = λ_usb * L_usb + λ_audio * L_audio
    """
    
    def __init__(
        self,
        lambda_usb: float = 1.0,
        lambda_audio: float = 10.0,
        use_stft_loss: bool = True,
        stft_scales: list = None
    ):
        super().__init__()
        
        self.lambda_usb = lambda_usb
        self.lambda_audio = lambda_audio
        self.use_stft_loss = use_stft_loss
        
        # Multi-resolution STFT parameters
        if stft_scales is None:
            # Default: 3 scales for audio
            self.stft_scales = [
                {'n_fft': 2048, 'hop_length': 512, 'win_length': 2048},
                {'n_fft': 1024, 'hop_length': 256, 'win_length': 1024},
                {'n_fft': 512, 'hop_length': 128, 'win_length': 512}
            ]
        else:
            self.stft_scales = stft_scales
    
    def stft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Multi-resolution STFT loss (magnitude + log-magnitude).
        Common in audio waveform generation (e.g., Parallel WaveGAN, HiFi-GAN).
        """
        total_loss = 0.0
        
        for scale in self.stft_scales:
            # Compute STFT
            pred_stft = torch.stft(
                pred,
                n_fft=scale['n_fft'],
                hop_length=scale['hop_length'],
                win_length=scale['win_length'],
                return_complex=True,
                window=torch.hann_window(scale['win_length'], device=pred.device)
            )
            target_stft = torch.stft(
                target,
                n_fft=scale['n_fft'],
                hop_length=scale['hop_length'],
                win_length=scale['win_length'],
                return_complex=True,
                window=torch.hann_window(scale['win_length'], device=target.device)
            )
            
            # Magnitude
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            # L1 magnitude loss
            mag_loss = F.l1_loss(pred_mag, target_mag)
            
            # Log magnitude loss (emphasizes lower magnitudes)
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-5),
                torch.log(target_mag + 1e-5)
            )
            
            total_loss += mag_loss + log_mag_loss
        
        return total_loss / len(self.stft_scales)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: dict with 'usb_pred' and 'audio_pred'
            targets: dict with 'usb_true' and 'audio_true'
            
        Returns:
            dict with 'total_loss', 'usb_loss', 'audio_loss', etc.
        """
        losses = {}
        
        # === USB Loss ===
        if 'usb_pred' in predictions and 'usb_true' in targets:
            usb_l1 = F.l1_loss(predictions['usb_pred'], targets['usb_true'])
            losses['usb_loss'] = usb_l1
        else:
            losses['usb_loss'] = torch.tensor(0.0, device=predictions['audio_pred'].device)
        
        # === Audio Loss ===
        if 'audio_pred' in predictions and 'audio_true' in targets:
            # Waveform L1 loss
            audio_l1 = F.l1_loss(predictions['audio_pred'], targets['audio_true'])
            losses['audio_l1'] = audio_l1
            
            # Multi-resolution STFT loss
            if self.use_stft_loss:
                audio_stft = self.stft_loss(predictions['audio_pred'], targets['audio_true'])
                losses['audio_stft'] = audio_stft
                losses['audio_loss'] = audio_l1 + audio_stft
            else:
                losses['audio_loss'] = audio_l1
        else:
            losses['audio_loss'] = torch.tensor(0.0, device=predictions['usb_pred'].device)
        
        # === Total Loss ===
        losses['total_loss'] = (
            self.lambda_usb * losses['usb_loss'] +
            self.lambda_audio * losses['audio_loss']
        )
        
        return losses


# ==============================================================================
# Testing & Utilities
# ==============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Testing Multi-Stage Powerline-to-Audio Pipeline")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # === Test Stage A ===
    print("\n" + "="*80)
    print("STAGE A: Powerline → USB + Latent")
    print("="*80)
    
    # Define sample rates
    powerline_sr = 200000
    usb_sr = 200000  # USB runs at same rate as powerline
    audio_sr = 44100  # Standard audio sample rate
    
    stage_a = PowerlineToUSBEncoder(
        powerline_sample_rate=powerline_sr,
        base_channels=32,  # Smaller for testing
        num_scales=25,
        latent_projection_dim=128
    ).to(device)
    
    # Count parameters
    stage_a_params = sum(p.numel() for p in stage_a.parameters() if p.requires_grad)
    print(f"Stage A parameters: {stage_a_params:,}")
    
    # Test forward
    x_pl = torch.randn(2, 1, 100000, device=device)  # 0.5 sec at 200kHz
    with torch.no_grad():
        outputs_a = stage_a(x_pl, return_encoder_features=True)
    
    print(f"\nInput shape: {x_pl.shape}")
    print(f"USB prediction shape: {outputs_a['usb_pred'].shape}")
    print(f"Latent shape: {outputs_a['latent'].shape}")
    print(f"Number of encoder features: {len(outputs_a['encoder_features'])}")
    print("✓ Stage A forward pass successful!")
    
    # === Test Stage B ===
    print("\n" + "="*80)
    print("STAGE B: USB + Latent → Audio")
    print("="*80)
    
    stage_b = USBToAudioDecoder(
        usb_sample_rate=usb_sr,
        audio_sample_rate=audio_sr,
        latent_dim=128,
        base_channels=32,
        use_dilated_convs=True
    ).to(device)
    
    stage_b_params = sum(p.numel() for p in stage_b.parameters() if p.requires_grad)
    print(f"Stage B parameters: {stage_b_params:,}")
    
    # Test forward
    usb_input = torch.randn(2, 1, 100000, device=device)
    latent_input = outputs_a['latent']
    
    with torch.no_grad():
        audio_pred = stage_b(usb_input, latent_input)
    
    print(f"USB input shape: {usb_input.shape}")
    print(f"Latent input shape: {latent_input.shape}")
    print(f"Audio output shape: {audio_pred.shape}")
    print(f"Expected audio length: ~{100000 / usb_sr * audio_sr:.0f} samples")
    print("✓ Stage B forward pass successful!")
    
    # === Test End-to-End Pipeline ===
    print("\n" + "="*80)
    print("STAGE C: End-to-End Pipeline")
    print("="*80)
    
    pipeline = PowerlineToAudioPipeline(
        powerline_sr=powerline_sr,
        usb_sr=usb_sr,
        audio_sr=audio_sr,
        base_channels_stageA=32,
        base_channels_stageB=32,
        latent_dim=128,
        num_wavelet_scales=25,
        use_dilated_convs=True
    ).to(device)
    
    total_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"Total pipeline parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (float32)")
    
    # Test end-to-end forward
    with torch.no_grad():
        outputs = pipeline(x_pl, mode='end_to_end')
    
    print(f"\nPowerline input: {x_pl.shape}")
    print(f"USB prediction: {outputs['usb_pred'].shape}")
    print(f"Audio prediction: {outputs['audio_pred'].shape}")
    print(f"Latent features: {outputs['latent'].shape}")
    print("✓ End-to-end forward pass successful!")
    
    # === Test Loss Function ===
    print("\n" + "="*80)
    print("Multi-Task Loss Function")
    print("="*80)
    
    loss_fn = MultiStageLoss(
        lambda_usb=1.0,
        lambda_audio=10.0,
        use_stft_loss=True
    )
    
    # Create dummy targets
    targets = {
        'usb_true': torch.randn_like(outputs['usb_pred']),
        'audio_true': torch.randn_like(outputs['audio_pred'])
    }
    
    with torch.no_grad():
        losses = loss_fn(outputs, targets)
    
    print("\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    print("✓ Loss computation successful!")
    
    # === Test Training Modes ===
    print("\n" + "="*80)
    print("Training Mode Tests")
    print("="*80)
    
    # Freeze/unfreeze tests
    print("\nFreezing Stage A...")
    pipeline.freeze_stage_a()
    frozen_params = sum(1 for p in pipeline.stage_a.parameters() if not p.requires_grad)
    total_stage_a_params = sum(1 for p in pipeline.stage_a.parameters())
    print(f"  Frozen: {frozen_params}/{total_stage_a_params} parameters")
    
    print("Unfreezing Stage A...")
    pipeline.unfreeze_stage_a()
    unfrozen_params = sum(1 for p in pipeline.stage_a.parameters() if p.requires_grad)
    print(f"  Trainable: {unfrozen_params}/{total_stage_a_params} parameters")
    
    # Test Stage B only mode
    print("\nTesting Stage B-only forward (with ground truth USB)...")
    with torch.no_grad():
        audio_from_true_usb = pipeline.forward_stage_b_only(
            usb_true=targets['usb_true'],
            x_powerline=x_pl
        )
    print(f"  Audio output shape: {audio_from_true_usb.shape}")
    print("✓ Stage B-only mode successful!")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    
    print("\n" + "="*80)
    print("Architecture Summary")
    print("="*80)
    print(f"""
Stage A (Powerline → USB + Latent):
  - Input: Powerline waveform at {powerline_sr} Hz
  - Wavelet scales: 25 (50 in production)
  - Encoder: 6-stage ResNet-DenseNet with wavelet attention
  - Output 1: USB prediction at {usb_sr} Hz
  - Output 2: Latent features ({128} channels)
  - Parameters: {stage_a_params:,}

Stage B (USB + Latent → Audio):
  - Input 1: USB signal at {usb_sr} Hz
  - Input 2: Latent features from Stage A
  - Architecture: Dual-branch fusion decoder
  - USB branch: 3-stage encoder
  - Latent branch: 2-stage encoder
  - Fusion: Cross-modal attention
  - Decoder: Dilated TCN + U-Net upsampling
  - Output: Audio waveform at {audio_sr} Hz
  - Parameters: {stage_b_params:,}

Total Pipeline:
  - End-to-end: Powerline → Audio
  - Sample rates: {powerline_sr} Hz (PL) → {usb_sr} Hz (USB) → {audio_sr} Hz (Audio)
  - Total parameters: {total_params:,}
  - Training modes: Stage A only, Stage B only, End-to-end
  - Loss: Multi-task (USB L1 + Audio L1 + Multi-resolution STFT)
    """)
