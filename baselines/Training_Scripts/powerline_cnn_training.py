#!/usr/bin/env python
# coding: utf-8

# # Powerline to USB Signal CNN Training
# 
# Train a CNN model to extract USB signals from powerline spectrograms.
# 
# **Architecture:**
# - **Input**: 2D spectrogram from bandpass-filtered powerline signal (17.5-22.5 kHz)
# - **Output**: 1D USB signal (time-domain audio)
# - **Model**: Encoder-Decoder with transposed convolutions (memory-efficient)
# 
# **Data Sources:**
# - Supports multiple data folders (Alice in Wonderland chapters, Podcasts, etc.)
# - Automatically detects `*_real.bin` (powerline) and `*_img.bin` (USB) pairs
# - Configure which folders to use in the configuration section
# 
# **Pipeline:**
# 1. Load data from selected folders
# 2. Downsample from 200 kHz → 100 kHz (reduces memory by 50%)
# 3. Create 5-second chunks
# 4. Apply bandpass filtering (17.5-22.5 kHz)
# 5. Generate spectrograms
# 6. Train CNN model (encoder-decoder architecture)
# 7. Evaluate on validation set

# ## 1. Configuration and Imports
# 
# **Data Folder Selection:**
# - Set `USE_FOLDERS = []` to use ALL available folders
# - Set `USE_FOLDERS = ['May29_Alice']` to use only Alice in Wonderland
# - Set `USE_FOLDERS = ['July10_Podcasts']` to use only Podcasts
# - Set `USE_FOLDERS = ['May29_Alice', 'July10_Podcasts']` to use both

# In[1]:


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

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# In[ ]:


# === CONFIGURATION ===
CONFIG = {
    # Data parameters
    'DATA_FOLDERS': [
        '/fs/scratch/<allocation>/May29_Alice',
        '/fs/scratch/<allocation>/July10_Podcasts',
    ],
    'USE_FOLDERS': [],  # Empty = use all folders, or specify: ['May29_Alice', 'July10_Podcasts']
    'OUTPUT_DIR': '/fs/scratch/<allocation>/model_runs/powerline_cnn',
    'ORIGINAL_SAMPLE_RATE': 200_000,  # 200 kHz (original)
    'SAMPLE_RATE': 200_000,  # 200 kHz (downsampled for training)
    'CHUNK_DURATION': 5,  # seconds
    
    # Signal processing
    'CARRIER_FREQ': 20000,  # 20 kHz
    'BANDWIDTH': 10000,  # 5 kHz
    'NPERSEG': 2048,  # FFT window size for spectrogram
    'NOVERLAP': 1024,  # 50% overlap
    
    # Model parameters
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',  # 'cuda' or 'cpu'
    'BATCH_SIZE': 10,
    'NUM_EPOCHS': 20,
    'LEARNING_RATE': 1e-4,
    'TRAIN_SPLIT': 0.8,  # 80% train, 20% validation
    
    # Training options
    'USE_GPU': True,  # Set to False to force CPU training
    'SAVE_EVERY': 5,  # Save checkpoint every N epochs
    'EARLY_STOPPING_PATIENCE': 10,
}

# Override device if USE_GPU is False
if not CONFIG['USE_GPU']:
    CONFIG['DEVICE'] = 'cpu'

# Filter folders based on USE_FOLDERS setting
if CONFIG['USE_FOLDERS']:
    # Filter to only use specified folders
    filtered_folders = []
    for folder in CONFIG['DATA_FOLDERS']:
        folder_name = os.path.basename(folder)
        if folder_name in CONFIG['USE_FOLDERS']:
            filtered_folders.append(folder)
    CONFIG['DATA_FOLDERS'] = filtered_folders
    print(f"📁 Using selected folders: {CONFIG['USE_FOLDERS']}")
else:
    print(f"📁 Using all available folders")

# Create output directory
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

print("\nConfiguration:")
for key, value in CONFIG.items():
    if key == 'DATA_FOLDERS':
        print(f"  {key}:")
        for folder in value:
            print(f"    - {folder}")
    else:
        print(f"  {key}: {value}")


print(f"  Output size per chunk: {CONFIG['CHUNK_DURATION'] * CONFIG['SAMPLE_RATE']:,} samples (was {CONFIG['CHUNK_DURATION'] * CONFIG['ORIGINAL_SAMPLE_RATE']:,})")

# Save configuration
with open(os.path.join(CONFIG['OUTPUT_DIR'], 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=2)


# ## Data Folder Preview
# 
# Preview available data files in each folder.

# In[6]:


# Preview data files in each folder
print("=" * 70)
print("DATA FOLDER PREVIEW")
print("=" * 70)

for folder in CONFIG['DATA_FOLDERS']:
    folder_name = os.path.basename(folder)
    powerline_files = sorted(glob.glob(os.path.join(folder, '*_real.bin')))
    
    print(f"\n📁 {folder_name}: {len(powerline_files)} files")
    
    # Show first 5 files
    for i, f in enumerate(powerline_files[:5]):
        basename = os.path.basename(f).replace('_real.bin', '')
        usb_file = f.replace('_real.bin', '_img.bin')
        has_usb = '✓' if os.path.exists(usb_file) else '✗'
        print(f"  {i+1}. {has_usb} {basename}")
    
    if len(powerline_files) > 5:
        print(f"  ... and {len(powerline_files) - 5} more files")

print("\n" + "=" * 70)


# ## 2. Data Preparation
# 
# Load all chapters, downsample to 100 kHz, and create chunks with bandpass filtering.

# In[7]:


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
    Load a chapter and create 60-second chunks with spectrograms.
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


# === LOAD DATA FROM ALL FOLDERS ===
print("Loading and processing data from all folders...\n")

all_chunks = []
file_count_by_folder = {}

for data_folder in CONFIG['DATA_FOLDERS']:
    folder_name = os.path.basename(data_folder)
    print(f"📁 Processing folder: {folder_name}")
    
    # Find all *_real.bin files (powerline signals)
    powerline_files = sorted(glob.glob(os.path.join(data_folder, '*_real.bin')))
    print(f"   Found {len(powerline_files)} files\n")
    
    file_count_by_folder[folder_name] = 0
    
    for powerline_path in tqdm(powerline_files, desc=f"  {folder_name}", leave=False):
        # Get corresponding USB file (*_img.bin)
        usb_path = powerline_path.replace('_real.bin', '_img.bin')
        
        if not os.path.exists(usb_path):
            print(f"  ⚠️  USB file not found: {os.path.basename(usb_path)}")
            continue
        
        try:
            chunks = load_and_process_chapter(powerline_path, usb_path, CONFIG)
            all_chunks.extend(chunks)
            file_count_by_folder[folder_name] += len(chunks)
            
            file_name = os.path.basename(powerline_path)
            print(f"    ✓ {file_name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"    ✗ Error processing {os.path.basename(powerline_path)}: {e}")
    
    print()

print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
for folder, count in file_count_by_folder.items():
    print(f"  {folder}: {count} chunks")
print(f"  TOTAL: {len(all_chunks)} chunks")
print("=" * 70)

if len(all_chunks) > 0:
    print(f"\nSample chunk info:")
    print(f"  Spectrogram shape: {all_chunks[0]['spectrogram'].shape}")
    print(f"  USB signal shape: {all_chunks[0]['usb_signal'].shape}")
    print(f"  Frequency range: {all_chunks[0]['frequencies'].min():.1f} - {all_chunks[0]['frequencies'].max():.1f} Hz")
    
    # Calculate total audio duration
    total_duration_sec = len(all_chunks) * CONFIG['CHUNK_DURATION']
    print(f"\nTotal audio duration: {total_duration_sec / 60:.1f} minutes ({total_duration_sec / 3600:.2f} hours)")


# ## 3. Dataset and DataLoader

# In[5]:


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


# ## 4. Model Architecture
# 
# CNN model that takes 2D spectrograms and outputs 1D time-domain signals.

# In[ ]:


class SpectrogramToSignalCNN(nn.Module):
    """
    Memory-efficient CNN model using encoder-decoder architecture.
    Uses transposed convolutions instead of massive FC layers.
    
    Input: (batch, 1, freq_bins, time_bins)
    Output: (batch, num_samples)
    """
    def __init__(self, freq_bins, time_bins, output_samples):
        super(SpectrogramToSignalCNN, self).__init__()
        
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.output_samples = output_samples
        
        # === ENCODER: Extract features from spectrogram ===
        self.encoder = nn.Sequential(
            # Conv1: (1, freq, time) -> (32, freq/2, time/2)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv2: (32, freq/2, time/2) -> (64, freq/4, time/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv3: (64, freq/4, time/4) -> (128, freq/8, time/8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # === BOTTLENECK: Collapse frequency dimension ===
        # Adaptive pooling to (128, 1, time/8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, time_bins // 8)),  # Collapse freq to 1
        )
        
        # === DECODER: Upsample to 1D signal using transposed convolutions ===
        # Now we have (256, 1, time/8) - treat as (256, time/8) for 1D upsampling
        
        # Calculate how much upsampling we need
        # time_bins // 8 -> output_samples
        # For 5s @ 100kHz = 500,000 samples
        # time_bins ≈ 488, so time_bins//8 ≈ 61
        # Need to upsample 61 -> 500,000 (factor of ~8200)
        
        # We'll use a combination of transposed conv1d and interpolation
        self.decoder = nn.Sequential(
            # First reshape to 1D: (256 channels, time/8 length)
            # TransposedConv1d: upsample by 8x
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Upsample by 4x
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Upsample by 4x
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Final conv to 1 channel
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder: extract features
        x = self.encoder(x)  # (batch, 128, freq/8, time/8)
        
        # Bottleneck: collapse frequency
        x = self.bottleneck(x)  # (batch, 256, 1, time/8)
        
        # Reshape for 1D decoder: (batch, 256, time/8)
        x = x.squeeze(2)  # Remove frequency dim
        
        # Decoder: upsample to output length
        x = self.decoder(x)  # (batch, 1, upsampled_length)
        
        # Remove channel dimension
        x = x.squeeze(1)  # (batch, upsampled_length)
        
        # Interpolate to exact output length if needed
        if x.size(1) != self.output_samples:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), 
                size=self.output_samples, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        return x

# Get dimensions from sample data
sample_spec = train_chunks[0]['spectrogram']
freq_bins, time_bins = sample_spec.shape
output_samples = len(train_chunks[0]['usb_signal'])

print(f"Model dimensions:")
print(f"  Input: (1, {freq_bins}, {time_bins})")
print(f"  Output: ({output_samples},)")
print(f"  Upsampling factor: {output_samples / (time_bins // 8):.1f}x")

# Create model
model = SpectrogramToSignalCNN(freq_bins, time_bins, output_samples)
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
optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"Loss function: MSE")
print(f"Optimizer: Adam (lr={CONFIG['LEARNING_RATE']})")
print(f"Scheduler: ReduceLROnPlateau")

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


def train_epoch(model, dataloader, criterion, optimizer, device):
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
    train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['DEVICE'])
    
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

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'history': history,
    'best_val_loss': best_val_loss,
    'freq_bins': freq_bins,
    'time_bins': time_bins,
    'output_samples': output_samples,
}, final_model_path)

print(f"✓ Final model saved to: {final_model_path}")
print(f"\n📊 Model Summary:")
print(f"  Input shape: (1, {freq_bins}, {time_bins})")
print(f"  Output shape: ({output_samples},)")
print(f"  Parameters: {total_params:,}")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"  Device used: {CONFIG['DEVICE'].upper()}")


# ## 10. Inference Function
# 
# Function to use the trained model for inference on new data.

# In[ ]:


def extract_usb_from_powerline(powerline_signal, model, config, device='cpu'):
    """
    Extract USB signal from powerline signal using trained model.
    
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

