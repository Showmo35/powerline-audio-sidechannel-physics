#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Consolidated imports (added cell)
import os
import glob
from pathlib import Path
import json
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# SciPy signal utilities
from scipy.signal import spectrogram, butter, filtfilt, resample_poly
from scipy import signal

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Notebook utilities
from IPython.display import Audio, display

warnings.filterwarnings('ignore')
print('Consolidated imports loaded')


# In[ ]:


# === CONFIGURATION ===
CONFIG = {
    # Data parameters
    'DATA_FOLDERS': [
        '/fs/scratch/<allocation>/May29_Alice',
        '/fs/scratch/<allocation>/July10_Podcasts',
    ],
    'USE_FOLDERS': ['May29_Alice'],  # Empty = use all folders, or specify: ['May29_Alice', 'July10_Podcasts']
    'OUTPUT_DIR': '/fs/scratch/<allocation>/model_runs/powerline_cnn',
    'ORIGINAL_SAMPLE_RATE': 200_000,  # 200 kHz (original)
    'SAMPLE_RATE': 200_000,  # 200 kHz (downsampled for training)
    'CHUNK_DURATION': 1,  # seconds
    
    # Signal processing
    'CARRIER_FREQ': 20000,  # 20 kHz
    'BANDWIDTH': 10000,  # 10 kHz (17.5-22.5 kHz)
    
    # Model parameters
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',  # 'cuda' or 'cpu'
    'BATCH_SIZE': 5,
    'NUM_EPOCHS': 10,
    'LEARNING_RATE': 1e-4,
    'TRAIN_SPLIT': 0.8,  # 80% train, 20% validation
    
    # Training options
    'USE_GPU': True,  # Set to False to force CPU training
    'SAVE_EVERY': 5,  # Save checkpoint every N epochs
    'EARLY_STOPPING_PATIENCE': 20,
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

def lowpass_filter(signal, cutoff_freq, sample_rate, order=4):
    """
    Apply lowpass filter to signal.
    """
    nyquist = sample_rate / 2
    cutoff = cutoff_freq / nyquist
    b, a = butter(order, cutoff, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

def load_and_process_chapter(powerline_path, usb_path, config):
    """
    Load a chapter and create fixed-length chunks WITHOUT pre-bandpassing the powerline input.
    Keep the raw (downsampled) powerline signal so the model's spectral branch
    can learn to focus on frequency regions of interest itself. USB is lowpass-filtered
    to the audio band (e.g., 0-17.5 kHz).
    """
    # Load raw data from binary files
    powerline = np.fromfile(powerline_path, dtype=np.float32)
    usb = np.fromfile(usb_path, dtype=np.float32)
    
    # Downsample from ORIGINAL_SAMPLE_RATE to SAMPLE_RATE if needed
    print(f"  Downsampling from {config['ORIGINAL_SAMPLE_RATE']/1000:.0f} kHz to {config['SAMPLE_RATE']/1000:.0f} kHz...")
    downsample_factor = max(1, config['ORIGINAL_SAMPLE_RATE'] // config['SAMPLE_RATE'])
    if downsample_factor > 1:
        powerline = resample_poly(powerline, 1, downsample_factor)
        usb = resample_poly(usb, 1, downsample_factor)
    print(f"  After downsampling: {len(powerline):,} samples")
    
    # DO NOT bandpass the powerline here. Keep raw downsampled powerline signal.
    raw_powerline = powerline

    # Apply lowpass filter to USB signal to retain audio band (default 20 kHz)
    usb_cutoff = config.get('USB_CUTOFF_HZ', 20000)
    print(f"  Applying lowpass filter to USB (0-{usb_cutoff/1000:.1f} kHz)...")
    filtered_usb = lowpass_filter(usb, usb_cutoff, config['SAMPLE_RATE'])
    
    # Now create fixed-duration chunks from the raw powerline and filtered USB
    chunk_samples = int(config['CHUNK_DURATION'] * config['SAMPLE_RATE'])
    num_chunks = len(raw_powerline) // chunk_samples
    print(f"  Creating {num_chunks} chunks of {config['CHUNK_DURATION']}s each...")
    
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples
        
        powerline_chunk = raw_powerline[start_idx:end_idx]
        usb_chunk = filtered_usb[start_idx:end_idx]
        
        # Normalize each chunk independently to stabilize training
        powerline_max = np.max(np.abs(powerline_chunk)) + 1e-8
        usb_max = np.max(np.abs(usb_chunk)) + 1e-8
        powerline_normalized = (powerline_chunk / powerline_max).astype(np.float32)
        usb_normalized = (usb_chunk / usb_max).astype(np.float32)
        
        chunks.append({
            'powerline_signal': powerline_normalized,
            'usb_signal': usb_normalized,
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
    print(f"  Powerline signal shape: {all_chunks[0]['powerline_signal'].shape}")
    print(f"  USB signal shape: {all_chunks[0]['usb_signal'].shape}")
    
    # Calculate total audio duration
    total_duration_sec = len(all_chunks) * CONFIG['CHUNK_DURATION']
    print(f"\nTotal audio duration: {total_duration_sec / 60:.1f} minutes ({total_duration_sec / 3600:.2f} hours)")


# ## 3. Dataset and DataLoader

# In[ ]:


class PowerlineDataset(Dataset):
    """
    Dataset for 1D powerline signal to USB signal mapping.
    """
    def __init__(self, chunks):
        self.chunks = chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Powerline signal: (1, samples) - add channel dimension for Conv1d
        powerline = torch.FloatTensor(chunk['powerline_signal']).unsqueeze(0)
        
        # USB signal: (samples,)
        usb = torch.FloatTensor(chunk['usb_signal'])
        
        return powerline, usb

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
# 1D CNN model that takes filtered powerline signals and outputs USB signals (signal-to-signal mapping).

# In[ ]:


import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D signals with skip connection.
    """
    def __init__(self, channels, kernel_size=15):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SpectralTimeAttentionCNN(nn.Module):
    """
    Dual-branch model: time-domain conv encoder/decoder + spectral branch that
    computes STFT magnitude and produces time-frame attention weights to gate
    time-domain features. This lets the model learn which frequency/time regions
    are informative without pre-bandpassing the input.
    """
    def __init__(self, input_samples, n_fft=2048, hop_length=512, time_channels=[64,128,256,512], attn_embed=256):
        super(SpectralTimeAttentionCNN, self).__init__()
        self.input_samples = input_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Time encoder
        enc_layers = []
        in_ch = 1
        for ch in time_channels:
            enc_layers.append(nn.Conv1d(in_ch, ch, kernel_size=15, stride=2, padding=7))
            enc_layers.append(nn.BatchNorm1d(ch))
            enc_layers.append(nn.ReLU())
            in_ch = ch
        self.time_encoder = nn.Sequential(*enc_layers)

        # Bottleneck residuals
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(time_channels[-1], kernel_size=7),
            ResidualBlock1D(time_channels[-1], kernel_size=7),
        )

        # Spectral projection: project F frequency bins -> embedding per time-frame
        self.spec_proj = nn.Sequential(
            nn.Linear((self.n_fft // 2 + 1), attn_embed),
            nn.ReLU(),
            nn.Linear(attn_embed, attn_embed),
            nn.ReLU(),
        )
        self.attn_head = nn.Sequential(
            nn.Linear(attn_embed, attn_embed//2),
            nn.ReLU(),
            nn.Linear(attn_embed//2, 1),
        )

        # Decoder (transpose convs)
        dec_layers = []
        rev_channels = list(reversed(time_channels))
        in_ch = time_channels[-1]
        for ch in rev_channels[:-1]:
            dec_layers.append(nn.ConvTranspose1d(in_ch, ch, kernel_size=16, stride=2, padding=7))
            dec_layers.append(nn.BatchNorm1d(ch))
            dec_layers.append(nn.ReLU())
            dec_layers.append(ResidualBlock1D(ch))
            in_ch = ch
        dec_layers.append(nn.ConvTranspose1d(in_ch, 32, kernel_size=16, stride=2, padding=7))
        dec_layers.append(nn.BatchNorm1d(32))
        dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Conv1d(32, 1, kernel_size=15, padding=7))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x: (B,1,L)
        B, C, L = x.shape
        t = self.time_encoder(x)
        t = self.bottleneck(t)
        L_down = t.size(2)

        # Spectral branch: compute STFT magnitude and produce time-frame attention
        x_flat = x.squeeze(1)
        spec = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, center=True, pad_mode='reflect')
        spec_mag = spec.abs()  # (B, F, T)
        spec_frames = spec_mag.permute(0, 2, 1)  # (B, T, F)
        spec_frames = torch.log1p(spec_frames)
        Bf, Tf, Ff = spec_frames.shape
        spec_frames_flat = spec_frames.reshape(Bf * Tf, Ff)
        proj = self.spec_proj(spec_frames_flat).reshape(Bf, Tf, -1)
        attn_logits = self.attn_head(proj).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_upsampled = F.interpolate(attn_weights.unsqueeze(1), size=L_down, mode='linear', align_corners=False).squeeze(1)

        gated = t * (1.0 + attn_upsampled.unsqueeze(1))
        out = self.decoder(gated)
        out = out.squeeze(1)
        if out.size(1) != self.input_samples:
            out = F.interpolate(out.unsqueeze(1), size=self.input_samples, mode='linear', align_corners=False).squeeze(1)
        return out

# Get dimensions from sample data
sample_powerline = train_chunks[0]['powerline_signal']
input_samples = len(sample_powerline)
output_samples = len(train_chunks[0]['usb_signal'])

print(f"Model dimensions:")
print(f"  Input: (1, {input_samples})")
print(f"  Output: ({output_samples},)")

# Create the new Spectral-Time Attention model
model = SpectralTimeAttentionCNN(input_samples, n_fft=2048, hop_length=512)
model = model.to(CONFIG['DEVICE'])

print(f"\n✓ Spectral-Time Attention model created")
print(f"✓ Model moved to {CONFIG['DEVICE']}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB (float32)")


# ## 5. Training Setup

# In[ ]:


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-5)

# Use OneCycleLR for better learning rate scheduling
steps_per_epoch = len(train_loader)
total_steps = CONFIG['NUM_EPOCHS'] * steps_per_epoch
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=CONFIG['LEARNING_RATE'],
    total_steps=total_steps,
    pct_start=0.3,  # Warm up for 30% of training
    anneal_strategy='cos',
    div_factor=25.0,  # Initial LR = max_lr / 25
    final_div_factor=10000.0  # Final LR = max_lr / 10000
)

print(f"Loss function: MSE")
print(f"Optimizer: AdamW (lr={CONFIG['LEARNING_RATE']}, weight_decay=1e-5)")
print(f"Scheduler: OneCycleLR (cosine annealing)")
print(f"  - Warmup: 30% of training")
print(f"  - Max LR: {CONFIG['LEARNING_RATE']}")
print(f"  - Initial LR: {CONFIG['LEARNING_RATE']/25:.6f}")
print(f"  - Final LR: {CONFIG['LEARNING_RATE']/10000:.6f}")

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


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Train for one epoch with OneCycleLR scheduler.
    """
    model.train()
    total_loss = 0.0
    
    for powerline_signals, usb_signals in tqdm(dataloader, desc="Training", leave=False):
        powerline_signals = powerline_signals.to(device)
        usb_signals = usb_signals.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(powerline_signals)
        loss = criterion(outputs, usb_signals)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step scheduler after each batch for OneCycleLR
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for powerline_signals, usb_signals in tqdm(dataloader, desc="Validation", leave=False):
            powerline_signals = powerline_signals.to(device)
            usb_signals = usb_signals.to(device)
            
            outputs = model(powerline_signals)
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
    
    # Train (scheduler is stepped inside train_epoch for OneCycleLR)
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, CONFIG['DEVICE'])
    
    # Validate
    val_loss = validate(model, val_loader, criterion, CONFIG['DEVICE'])
    
    # Record history (get current LR from scheduler)
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(current_lr)
    
    # Print progress
    epoch_time = (datetime.now() - epoch_start).total_seconds()
    print(f"Epoch [{epoch+1}/{CONFIG['NUM_EPOCHS']}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss:   {val_loss:.6f}")
    print(f"  LR:         {current_lr:.6f}")
    
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
checkpoint = torch.load(os.path.join(CONFIG['OUTPUT_DIR'], 'checkpoint_epoch_10.pt'))
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
        powerline_input, usb_true = val_dataset[i]
        powerline_input_batch = powerline_input.unsqueeze(0).to(CONFIG['DEVICE'])
        
        # Predict
        usb_pred = model(powerline_input_batch).cpu().squeeze().numpy()
        usb_true = usb_true.numpy()
        powerline_np = powerline_input.squeeze().numpy()
        
        # Time axis
        time_axis = np.arange(len(usb_true)) / CONFIG['SAMPLE_RATE']
        
        # Plot powerline input signal
        axes[i, 0].plot(time_axis, powerline_np, color='green', linewidth=0.5, alpha=0.8)
        axes[i, 0].set_title(f'Sample {i+1} - Input Powerline Signal (Filtered)', fontweight='bold')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].grid(True, alpha=0.3)
        
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
    'input_samples': input_samples,
    'output_samples': output_samples,
}, final_model_path)

print(f"✓ Final model saved to: {final_model_path}")
print(f"\n📊 Model Summary:")
print(f"  Input shape: (1, {input_samples})")
print(f"  Output shape: ({output_samples},)")
print(f"  Parameters: {total_params:,}")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"  Device used: {CONFIG['DEVICE'].upper()}")

