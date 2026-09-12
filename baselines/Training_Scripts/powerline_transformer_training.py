#!/usr/bin/env python
# coding: utf-8

# # Powerline to USB Signal - Transformer Model
# 
# Train a Transformer-based model to extract USB signals from powerline spectrograms.
# 
# **Architecture:**
# - **Input**: 2D spectrogram from bandpass-filtered powerline signal (17.5-22.5 kHz)
# - **Output**: 1D USB signal (time-domain audio)
# - **Model**: Vision Transformer (ViT) encoder + Transformer decoder
# 
# **Pipeline:**
# 1. Load all chapters
# 2. Downsample from 200 kHz → 100 kHz (reduces memory by 50%)
# 3. Create 5-second chunks
# 4. Apply bandpass filtering (17.5-22.5 kHz)
# 5. Generate spectrograms
# 6. Train Transformer model (encoder-decoder architecture)
# 7. Evaluate on validation set
# 
# **Advantages of Transformer:**
# - Better at capturing long-range dependencies
# - Self-attention mechanisms learn important frequency-time relationships
# - State-of-the-art for sequence tasks

# ## 1. Configuration and Imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import spectrogram, butter, filtfilt, resample_poly
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import math

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# In[ ]:


# === CONFIGURATION ===
CONFIG = {
    # Data parameters
    'DATA_DIR': '/fs/scratch/<allocation>/May29_Alice',
    'OUTPUT_DIR': '/fs/scratch/<allocation>/model_runs/powerline_transformer',
    'ORIGINAL_SAMPLE_RATE': 200_000,  # 200 kHz (original)
    'SAMPLE_RATE': 100_000,  # 100 kHz (downsampled for training)
    'CHUNK_DURATION': 5,  # seconds
    
    # Signal processing
    'CARRIER_FREQ': 20000,  # 20 kHz
    'BANDWIDTH': 5000,  # 5 kHz
    'NPERSEG': 2048,  # FFT window size for spectrogram
    'NOVERLAP': 1024,  # 50% overlap
    
    # Transformer model parameters
    'PATCH_SIZE': (8, 8),  # Patch size for Vision Transformer
    'EMBED_DIM': 512,  # Embedding dimension
    'NUM_HEADS': 8,  # Number of attention heads
    'NUM_ENCODER_LAYERS': 6,  # Number of encoder layers
    'NUM_DECODER_LAYERS': 6,  # Number of decoder layers
    'FF_DIM': 2048,  # Feedforward dimension
    'DROPOUT': 0.1,  # Dropout rate
    
    # Training parameters
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'BATCH_SIZE': 4,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'TRAIN_SPLIT': 0.8,  # 80% train, 20% validation
    
    # Training options
    'USE_GPU': True,
    'SAVE_EVERY': 5,  # Save checkpoint every N epochs
    'EARLY_STOPPING_PATIENCE': 15,
    'GRAD_CLIP': 1.0,  # Gradient clipping for stability
}

# Override device if USE_GPU is False
if not CONFIG['USE_GPU']:
    CONFIG['DEVICE'] = 'cpu'

# Create output directory
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

print(f"\n💡 Transformer Model:")
print(f"  Embedding dim: {CONFIG['EMBED_DIM']}")
print(f"  Attention heads: {CONFIG['NUM_HEADS']}")
print(f"  Encoder layers: {CONFIG['NUM_ENCODER_LAYERS']}")
print(f"  Decoder layers: {CONFIG['NUM_DECODER_LAYERS']}")

# Save configuration
with open(os.path.join(CONFIG['OUTPUT_DIR'], 'config.json'), 'w') as f:
    # Convert tuple to list for JSON serialization
    config_save = CONFIG.copy()
    config_save['PATCH_SIZE'] = list(CONFIG['PATCH_SIZE'])
    json.dump(config_save, f, indent=2)


# ## 2. Data Preparation
# 
# Load all chapters, downsample to 100 kHz, and create chunks with bandpass filtering.

# In[ ]:


def bandpass_filter(signal, low_freq, high_freq, sample_rate, order=4):
    """
    Apply bandpass filter to signal.
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def create_spectrogram(signal, sample_rate, nperseg=2048, noverlap=1024):
    """
    Create spectrogram from signal.
    Returns: frequencies, times, spectrogram (in dB)
    """
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db

def load_and_process_chapter(powerline_path, usb_path, config):
    """
    Load a chapter and create chunks with spectrograms.
    """
    # Load data
    powerline = np.fromfile(powerline_path, dtype=np.float32)
    usb = np.fromfile(usb_path, dtype=np.float32)
    
    # Downsample from 200 kHz to 100 kHz (2:1 ratio)
    print(f"  Downsampling from {config['ORIGINAL_SAMPLE_RATE']/1000:.0f} kHz to {config['SAMPLE_RATE']/1000:.0f} kHz...")
    downsample_factor = config['ORIGINAL_SAMPLE_RATE'] // config['SAMPLE_RATE']
    powerline = resample_poly(powerline, 1, downsample_factor)
    usb = resample_poly(usb, 1, downsample_factor)
    print(f"  After downsampling: {len(powerline):,} samples")
    
    chunk_samples = config['CHUNK_DURATION'] * config['SAMPLE_RATE']
    num_chunks = len(powerline) // chunk_samples
    
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples
        
        # Extract chunk
        powerline_chunk = powerline[start_idx:end_idx]
        usb_chunk = usb[start_idx:end_idx]
        
        # Apply bandpass filter
        low_freq = config['CARRIER_FREQ'] - config['BANDWIDTH'] / 2
        high_freq = config['CARRIER_FREQ'] + config['BANDWIDTH'] / 2
        filtered_chunk = bandpass_filter(powerline_chunk, low_freq, high_freq, config['SAMPLE_RATE'])
        
        # Create spectrogram
        f, t, spec = create_spectrogram(filtered_chunk, config['SAMPLE_RATE'], 
                                       config['NPERSEG'], config['NOVERLAP'])
        
        # Focus on the relevant frequency band
        freq_mask = (f >= low_freq) & (f <= high_freq)
        spec_focused = spec[freq_mask, :]
        
        # Normalize USB signal
        usb_normalized = usb_chunk / (np.max(np.abs(usb_chunk)) + 1e-8)
        
        chunks.append({
            'spectrogram': spec_focused.astype(np.float32),
            'usb_signal': usb_normalized.astype(np.float32),
            'frequencies': f[freq_mask],
            'times': t
        })
    
    return chunks

print("Data processing functions defined.")


# In[ ]:


# === LOAD ALL CHAPTERS ===
print("Loading and processing all chapters...\n")

powerline_files = sorted(glob.glob(os.path.join(CONFIG['DATA_DIR'], 'Chap_*_real.bin')))
print(f"Found {len(powerline_files)} chapter files\n")

all_chunks = []

for powerline_path in tqdm(powerline_files, desc="Processing chapters"):
    # Get corresponding USB file
    usb_path = powerline_path.replace('_real.bin', '_img.bin')
    
    if not os.path.exists(usb_path):
        print(f"⚠️  USB file not found: {usb_path}")
        continue
    
    try:
        chunks = load_and_process_chapter(powerline_path, usb_path, CONFIG)
        all_chunks.extend(chunks)
        
        chapter_name = os.path.basename(powerline_path)
        print(f"  ✓ {chapter_name}: {len(chunks)} chunks")
    except Exception as e:
        print(f"  ✗ Error processing {powerline_path}: {e}")

print(f"\n✓ Total chunks created: {len(all_chunks)}")

if len(all_chunks) > 0:
    print(f"\nSample chunk info:")
    print(f"  Spectrogram shape: {all_chunks[0]['spectrogram'].shape}")
    print(f"  USB signal shape: {all_chunks[0]['usb_signal'].shape}")
    print(f"  Frequency range: {all_chunks[0]['frequencies'].min():.1f} - {all_chunks[0]['frequencies'].max():.1f} Hz")


# ## 3. Dataset and DataLoader

# In[ ]:


class PowerlineDataset(Dataset):
    """
    Dataset for powerline spectrogram to USB signal mapping.
    """
    def __init__(self, chunks):
        self.chunks = chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Spectrogram: (1, freq_bins, time_bins) - add channel dimension
        spec = torch.FloatTensor(chunk['spectrogram']).unsqueeze(0)
        
        # USB signal: (samples,)
        usb = torch.FloatTensor(chunk['usb_signal'])
        
        return spec, usb

# Split data into train and validation
num_train = int(len(all_chunks) * CONFIG['TRAIN_SPLIT'])
train_chunks = all_chunks[:num_train]
val_chunks = all_chunks[num_train:]

print(f"Dataset split:")
print(f"  Training: {len(train_chunks)} chunks")
print(f"  Validation: {len(val_chunks)} chunks")

# Create datasets and dataloaders
train_dataset = PowerlineDataset(train_chunks)
val_dataset = PowerlineDataset(val_chunks)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)

print(f"\n✓ DataLoaders created (batch size: {CONFIG['BATCH_SIZE']})")


# ## 4. Transformer Model Architecture
# 
# Vision Transformer-inspired encoder + Transformer decoder for sequence generation.

# In[ ]:


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SpectrogramToSignalTransformer(nn.Module):
    """
    Transformer-based model for converting spectrograms to signals.
    
    Uses Vision Transformer-style patch embedding + Transformer encoder-decoder.
    """
    def __init__(self, freq_bins, time_bins, output_samples, config):
        super(SpectrogramToSignalTransformer, self).__init__()
        
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.output_samples = output_samples
        self.embed_dim = config['EMBED_DIM']
        self.patch_size = config['PATCH_SIZE']
        
        # Calculate number of patches
        self.num_patches_h = freq_bins // self.patch_size[0]
        self.num_patches_w = time_bins // self.patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding (like ViT)
        patch_dim = self.patch_size[0] * self.patch_size[1]
        self.patch_embed = nn.Linear(patch_dim, self.embed_dim)
        
        # Positional encoding for patches
        self.pos_encoding = PositionalEncoding(self.embed_dim, max_len=self.num_patches + 1)
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config['NUM_HEADS'],
            dim_feedforward=config['FF_DIM'],
            dropout=config['DROPOUT'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['NUM_ENCODER_LAYERS']
        )
        
        # Decoder sequence length (downsample output for efficiency)
        self.decoder_seq_len = 1024  # Target sequence length
        
        # Learnable decoder queries
        self.decoder_queries = nn.Parameter(torch.randn(1, self.decoder_seq_len, self.embed_dim))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=config['NUM_HEADS'],
            dim_feedforward=config['FF_DIM'],
            dropout=config['DROPOUT'],
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['NUM_DECODER_LAYERS']
        )
        
        # Positional encoding for decoder
        self.decoder_pos_encoding = PositionalEncoding(self.embed_dim, max_len=self.decoder_seq_len)
        
        # Output projection: map each decoder token to multiple samples
        samples_per_token = output_samples // self.decoder_seq_len
        self.output_projection = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(config['DROPOUT']),
            nn.Linear(512, samples_per_token),
            nn.Tanh()
        )
        
        self.samples_per_token = samples_per_token
    
    def patchify(self, x):
        """
        Convert spectrogram to patches.
        x: (batch, 1, freq_bins, time_bins)
        Returns: (batch, num_patches, patch_dim)
        """
        batch_size = x.size(0)
        x = x.squeeze(1)  # (batch, freq_bins, time_bins)
        
        # Reshape to patches
        patches = x.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
        # patches: (batch, num_patches_h, num_patches_w, patch_h, patch_w)
        
        patches = patches.contiguous().view(batch_size, -1, self.patch_size[0] * self.patch_size[1])
        # patches: (batch, num_patches, patch_dim)
        
        return patches
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convert to patches
        patches = self.patchify(x)  # (batch, num_patches, patch_dim)
        
        # Embed patches
        x = self.patch_embed(patches)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        memory = self.transformer_encoder(x)  # (batch, num_patches+1, embed_dim)
        
        # Decoder queries with positional encoding
        decoder_input = self.decoder_queries.expand(batch_size, -1, -1)
        decoder_input = self.decoder_pos_encoding(decoder_input)
        
        # Transformer decoder
        decoder_output = self.transformer_decoder(decoder_input, memory)
        # decoder_output: (batch, decoder_seq_len, embed_dim)
        
        # Project to output samples
        output = self.output_projection(decoder_output)
        # output: (batch, decoder_seq_len, samples_per_token)
        
        # Reshape to 1D signal
        output = output.view(batch_size, -1)  # (batch, decoder_seq_len * samples_per_token)
        
        # Interpolate to exact output length if needed
        if output.size(1) != self.output_samples:
            output = torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=self.output_samples,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return output

print("Transformer model architecture defined.")


# In[ ]:


# Get dimensions from sample data
sample_spec = train_chunks[0]['spectrogram']
freq_bins, time_bins = sample_spec.shape
output_samples = len(train_chunks[0]['usb_signal'])

print(f"Model dimensions:")
print(f"  Input: (1, {freq_bins}, {time_bins})")
print(f"  Output: ({output_samples},)")

# Adjust dimensions to be divisible by patch size
patch_h, patch_w = CONFIG['PATCH_SIZE']
freq_bins_adjusted = (freq_bins // patch_h) * patch_h
time_bins_adjusted = (time_bins // patch_w) * patch_w

if freq_bins != freq_bins_adjusted or time_bins != time_bins_adjusted:
    print(f"\n⚠️  Adjusting dimensions to be divisible by patch size:")
    print(f"  Frequency: {freq_bins} → {freq_bins_adjusted}")
    print(f"  Time: {time_bins} → {time_bins_adjusted}")
    freq_bins = freq_bins_adjusted
    time_bins = time_bins_adjusted

num_patches = (freq_bins // patch_h) * (time_bins // patch_w)
print(f"\nPatch configuration:")
print(f"  Patch size: {patch_h} × {patch_w}")
print(f"  Number of patches: {num_patches}")

# Create model
model = SpectrogramToSignalTransformer(freq_bins, time_bins, output_samples, CONFIG)
model = model.to(CONFIG['DEVICE'])

print(f"\n✓ Model created and moved to {CONFIG['DEVICE']}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (float32)")

# Estimate memory usage
print(f"\nEstimated GPU memory:")
print(f"  Model parameters: {total_params * 4 / 1024**2:.1f} MB")
print(f"  Gradients: {total_params * 4 / 1024**2:.1f} MB")
print(f"  Optimizer state (Adam): {total_params * 8 / 1024**2:.1f} MB")
print(f"  Total (model + optimizer): {total_params * 16 / 1024**3:.2f} GB")


# ## 5. Training Setup

# In[ ]:


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"Loss function: MSE")
print(f"Optimizer: AdamW (lr={CONFIG['LEARNING_RATE']}, weight_decay=0.01)")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Gradient clipping: {CONFIG['GRAD_CLIP']}")

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'learning_rate': []
}

# Best model tracking
best_val_loss = float('inf')
patience_counter = 0

print(f"\n✓ Training setup complete")


# ## 6. Training Loop

# In[ ]:


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    
    for spectrograms, usb_signals in tqdm(dataloader, desc="Training", leave=False):
        spectrograms = spectrograms.to(device)
        usb_signals = usb_signals.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, usb_signals)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for spectrograms, usb_signals in tqdm(dataloader, desc="Validation", leave=False):
            spectrograms = spectrograms.to(device)
            usb_signals = usb_signals.to(device)
            
            outputs = model(spectrograms)
            loss = criterion(outputs, usb_signals)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

print("Training functions defined. Ready to train!")


# In[ ]:


# === MAIN TRAINING LOOP ===
print(f"Starting training on {CONFIG['DEVICE'].upper()}...\n")
print("=" * 70)

start_time = datetime.now()

for epoch in range(CONFIG['NUM_EPOCHS']):
    epoch_start = datetime.now()
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                            CONFIG['DEVICE'], grad_clip=CONFIG['GRAD_CLIP'])
    
    # Validate
    val_loss = validate(model, val_loader, criterion, CONFIG['DEVICE'])
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    # Print progress
    epoch_time = (datetime.now() - epoch_start).total_seconds()
    print(f"Epoch [{epoch+1}/{CONFIG['NUM_EPOCHS']}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss:   {val_loss:.6f}")
    print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pt'))
        print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
    
    # Save periodic checkpoint
    if (epoch + 1) % CONFIG['SAVE_EVERY'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(CONFIG['OUTPUT_DIR'], f'checkpoint_epoch_{epoch+1}.pt'))
        print(f"  ✓ Checkpoint saved")
    
    # Early stopping
    if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
        print(f"\n⚠️  Early stopping triggered (patience: {CONFIG['EARLY_STOPPING_PATIENCE']})")
        break
    
    print()

total_time = (datetime.now() - start_time).total_seconds()
print("=" * 70)
print(f"\n✓ Training complete!")
print(f"  Total time: {total_time/60:.1f} minutes")
print(f"  Best val loss: {best_val_loss:.6f}")

# Save training history
np.save(os.path.join(CONFIG['OUTPUT_DIR'], 'training_history.npy'), history)
print(f"\n✓ Training history saved to {CONFIG['OUTPUT_DIR']}")


# ## 7. Training Visualization

# In[ ]:


# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning rate
axes[1].plot(history['learning_rate'], linewidth=2, color='green')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Learning Rate', fontsize=12)
axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Training curves saved")


# ## 8. Model Evaluation
# 
# Load the best model and evaluate on validation set.

# In[ ]:


# Load best model
checkpoint = torch.load(os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Best model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")

# Evaluate on a few validation samples
num_samples_to_plot = min(3, len(val_dataset))

fig, axes = plt.subplots(num_samples_to_plot, 3, figsize=(18, 4 * num_samples_to_plot))
if num_samples_to_plot == 1:
    axes = axes.reshape(1, -1)

with torch.no_grad():
    for i in range(num_samples_to_plot):
        spec, usb_true = val_dataset[i]
        spec_input = spec.unsqueeze(0).to(CONFIG['DEVICE'])
        
        # Predict
        usb_pred = model(spec_input).cpu().squeeze().numpy()
        usb_true = usb_true.numpy()
        
        # Time axis
        time_axis = np.arange(len(usb_true)) / CONFIG['SAMPLE_RATE']
        
        # Plot spectrogram input
        im = axes[i, 0].imshow(spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Sample {i+1} - Input Spectrogram', fontweight='bold')
        axes[i, 0].set_ylabel('Frequency Bin')
        axes[i, 0].set_xlabel('Time Bin')
        plt.colorbar(im, ax=axes[i, 0])
        
        # Plot predicted signal
        axes[i, 1].plot(time_axis, usb_pred, color='red', linewidth=0.5, alpha=0.8, label='Predicted')
        axes[i, 1].set_title(f'Sample {i+1} - Predicted USB Signal', fontweight='bold')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
        
        # Plot true signal
        axes[i, 2].plot(time_axis, usb_true, color='blue', linewidth=0.5, alpha=0.8, label='Ground Truth')
        axes[i, 2].set_title(f'Sample {i+1} - True USB Signal', fontweight='bold')
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].legend()
        
        # Calculate correlation
        corr = np.corrcoef(usb_pred, usb_true)[0, 1]
        print(f"Sample {i+1} - Correlation: {corr:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Evaluation complete")


# ## 9. Save Final Model

# In[ ]:


# Save final model with full metadata
final_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'final_model.pt')

# Convert config for saving
config_save = CONFIG.copy()
config_save['PATCH_SIZE'] = list(CONFIG['PATCH_SIZE'])

torch.save({
    'model_state_dict': model.state_dict(),
    'config': config_save,
    'history': history,
    'best_val_loss': best_val_loss,
    'freq_bins': freq_bins,
    'time_bins': time_bins,
    'output_samples': output_samples,
}, final_model_path)

print(f"✓ Final model saved to: {final_model_path}")
print(f"\n📊 Model Summary:")
print(f"  Architecture: Transformer (Encoder-Decoder)")
print(f"  Input shape: (1, {freq_bins}, {time_bins})")
print(f"  Output shape: ({output_samples},)")
print(f"  Parameters: {total_params:,}")
print(f"  Embedding dim: {CONFIG['EMBED_DIM']}")
print(f"  Attention heads: {CONFIG['NUM_HEADS']}")
print(f"  Encoder layers: {CONFIG['NUM_ENCODER_LAYERS']}")
print(f"  Decoder layers: {CONFIG['NUM_DECODER_LAYERS']}")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"  Device used: {CONFIG['DEVICE'].upper()}")


# ## 10. Inference Function
# 
# Function to use the trained model for inference on new data.

# In[ ]:


def extract_usb_from_powerline(powerline_signal, model, config, device='cpu'):
    """
    Extract USB signal from powerline signal using trained Transformer model.
    
    Args:
        powerline_signal: Raw powerline signal (numpy array)
        model: Trained PyTorch model
        config: Configuration dictionary
        device: 'cuda' or 'cpu'
    
    Returns:
        Extracted USB signal (numpy array)
    """
    # Apply bandpass filter
    low_freq = config['CARRIER_FREQ'] - config['BANDWIDTH'] / 2
    high_freq = config['CARRIER_FREQ'] + config['BANDWIDTH'] / 2
    filtered = bandpass_filter(powerline_signal, low_freq, high_freq, config['SAMPLE_RATE'])
    
    # Create spectrogram
    f, t, spec = create_spectrogram(filtered, config['SAMPLE_RATE'], 
                                    config['NPERSEG'], config['NOVERLAP'])
    
    # Focus on relevant frequency band
    freq_mask = (f >= low_freq) & (f <= high_freq)
    spec_focused = spec[freq_mask, :]
    
    # Convert to tensor
    spec_tensor = torch.FloatTensor(spec_focused).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        usb_signal = model(spec_tensor).cpu().squeeze().numpy()
    
    return usb_signal

print("✓ Inference function defined")
print("\nUsage example:")
print("  usb = extract_usb_from_powerline(powerline_signal, model, CONFIG, device='cuda')")

