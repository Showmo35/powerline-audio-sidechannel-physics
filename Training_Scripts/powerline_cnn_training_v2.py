#!/usr/bin/env python
# coding: utf-8

# # Powerline to USB Signal CNN Training
# 
# Train a 1D CNN model to extract USB signals from bandpass-filtered powerline signals.
# 
# **Architecture:**
# - **Input**: 1D filtered powerline signal (17.5-22.5 kHz bandpass)
# - **Output**: 1D USB signal (time-domain audio)
# - **Model**: 1D Convolutional Encoder-Decoder (signal-to-signal)
# 
# **Data Sources:**
# - Supports multiple data folders (Alice in Wonderland chapters, Podcasts, etc.)
# - Automatically detects `*_real.bin` (powerline) and `*_img.bin` (USB) pairs
# - Configure which folders to use in the configuration section
# 
# **Pipeline:**
# 1. Load data from selected folders
# 2. Downsample from 200 kHz → target sample rate (optional)
# 3. Apply bandpass filtering (17.5-22.5 kHz) to entire chapters
# 4. Create 5-second chunks from filtered signals
# 5. Train 1D CNN model (encoder-decoder architecture)
# 6. Evaluate on validation set

# ## 1. Configuration and Imports
# 
# **Data Folder Selection:**
# - Set `USE_FOLDERS = []` to use ALL available folders
# - Set `USE_FOLDERS = ['May29_Alice']` to use only Alice in Wonderland
# - Set `USE_FOLDERS = ['July10_Podcasts']` to use only Podcasts
# - Set `USE_FOLDERS = ['May29_Alice', 'July10_Podcasts']` to use both

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
    'USE_FOLDERS': ['May29_Alice'],  # Empty = use all folders, or specify: ['May29_Alice', 'July10_Podcasts']
    'OUTPUT_DIR': '/fs/scratch/<allocation>/model_runs/powerline_cnn',
    'ORIGINAL_SAMPLE_RATE': 200_000,  # 200 kHz (original)
    'SAMPLE_RATE': 200_000,  # 200 kHz (downsampled for training)
    'CHUNK_DURATION': 5,  # seconds
    
    # Signal processing
    'CARRIER_FREQ': 20000,  # 20 kHz
    'BANDWIDTH': 10000,  # 10 kHz (17.5-22.5 kHz)
    
    # Model parameters
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',  # 'cuda' or 'cpu'
    'BATCH_SIZE': 10,
    'NUM_EPOCHS': 25,
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

# In[ ]:


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

def load_and_process_chapter(powerline_path, usb_path, config):
    """
    Load a chapter, apply bandpass filter, then create chunks.
    No spectrograms - use filtered powerline signal directly.
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
    
    # Apply bandpass filter to entire chapter first
    low_freq = config['CARRIER_FREQ'] - config['BANDWIDTH'] / 2
    high_freq = config['CARRIER_FREQ'] + config['BANDWIDTH'] / 2
    print(f"  Applying bandpass filter ({low_freq/1000:.1f}-{high_freq/1000:.1f} kHz)...")
    filtered_powerline = bandpass_filter(powerline, low_freq, high_freq, config['SAMPLE_RATE'])
    
    # Now create chunks from the filtered signal
    chunk_samples = config['CHUNK_DURATION'] * config['SAMPLE_RATE']
    num_chunks = len(filtered_powerline) // chunk_samples
    print(f"  Creating {num_chunks} chunks of {config['CHUNK_DURATION']}s each...")
    
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples
        
        # Extract chunk from filtered powerline
        powerline_chunk = filtered_powerline[start_idx:end_idx]
        usb_chunk = usb[start_idx:end_idx]
        
        # Normalize both signals
        powerline_normalized = powerline_chunk / (np.max(np.abs(powerline_chunk)) + 1e-8)
        usb_normalized = usb_chunk / (np.max(np.abs(usb_chunk)) + 1e-8)
        
        chunks.append({
            'powerline_signal': powerline_normalized.astype(np.float32),
            'usb_signal': usb_normalized.astype(np.float32),
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


class SignalToSignalCNN(nn.Module):
    """
    1D CNN encoder-decoder for signal-to-signal mapping.
    
    Input: (batch, 1, num_samples) - filtered powerline signal
    Output: (batch, num_samples) - USB signal
    """
    def __init__(self, input_samples):
        super(SignalToSignalCNN, self).__init__()
        
        self.input_samples = input_samples
        
        # === ENCODER: Extract features using Conv1d ===
        # Downsample progressively: input -> input/2 -> input/4 -> input/8
        self.encoder = nn.Sequential(
            # Conv1: (1, L) -> (32, L/2)
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Conv2: (32, L/2) -> (64, L/4)
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Conv3: (64, L/4) -> (128, L/8)
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Conv4: (128, L/8) -> (256, L/16)
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # === BOTTLENECK ===
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # === DECODER: Upsample back to original length ===
        self.decoder = nn.Sequential(
            # Upsample by 2x: (256, L/16) -> (128, L/8)
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Upsample by 2x: (128, L/8) -> (64, L/4)
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Upsample by 2x: (64, L/4) -> (32, L/2)
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Upsample by 2x: (32, L/2) -> (1, L)
            nn.ConvTranspose1d(32, 1, kernel_size=16, stride=2, padding=7),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, x):
        # x: (batch, 1, num_samples)
        
        # Encoder: downsample and extract features
        x = self.encoder(x)  # (batch, 256, num_samples/16)
        
        # Bottleneck: process features
        x = self.bottleneck(x)  # (batch, 256, num_samples/16)
        
        # Decoder: upsample back to original length
        x = self.decoder(x)  # (batch, 1, ~num_samples)
        
        # Remove channel dimension
        x = x.squeeze(1)  # (batch, num_samples)
        
        # Adjust length if needed (due to rounding in conv operations)
        if x.size(1) != self.input_samples:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), 
                size=self.input_samples, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        return x

# Get dimensions from sample data
sample_powerline = train_chunks[0]['powerline_signal']
input_samples = len(sample_powerline)
output_samples = len(train_chunks[0]['usb_signal'])

print(f"Model dimensions:")
print(f"  Input: (1, {input_samples})")
print(f"  Output: ({output_samples},)")

# Create model
model = SignalToSignalCNN(input_samples)
model = model.to(CONFIG['DEVICE'])

print(f"\n✓ Model created and moved to {CONFIG['DEVICE']}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())


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
    
    for powerline_signals, usb_signals in tqdm(dataloader, desc="Training", leave=False):
        powerline_signals = powerline_signals.to(device)
        usb_signals = usb_signals.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(powerline_signals)
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
    
    # Normalize
    filtered_normalized = filtered / (np.max(np.abs(filtered)) + 1e-8)
    
    # Convert to tensor: (1, 1, num_samples)
    signal_tensor = torch.FloatTensor(filtered_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        usb_signal = model(signal_tensor).cpu().squeeze().numpy()
    
    return usb_signal

print("✓ Inference function defined")
print("\nUsage example:")
print("  usb = extract_usb_from_powerline(powerline_signal, model, CONFIG, device='cuda')")


# ## 11. Load Trained Model and Spectrogram Analysis
# 
# Load the best model checkpoint and compare predicted vs true USB signals using spectrograms (0-20 kHz).

# In[ ]:


# Load the best trained model
best_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pt')

print(f"Loading trained model from: {best_model_path}")

# Check if model file exists
if not os.path.exists(best_model_path):
    print(f"❌ Model file not found: {best_model_path}")
    print("Please train the model first or check the path.")
else:
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location=CONFIG['DEVICE'])
    
    # Create model architecture (same as training)
    sample_powerline = val_chunks[0]['powerline_signal']
    input_samples = len(sample_powerline)
    
    # Initialize model
    loaded_model = SignalToSignalCNN(input_samples)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model = loaded_model.to(CONFIG['DEVICE'])
    loaded_model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.6f}")
    print(f"  Training Loss: {checkpoint['train_loss']:.6f}")


# In[ ]:


# Generate predictions for validation samples
num_samples = min(5, len(val_dataset))
predictions = []

print(f"\nGenerating predictions for {num_samples} validation samples...")

with torch.no_grad():
    for i in range(num_samples):
        powerline_input, usb_true = val_dataset[i]
        powerline_batch = powerline_input.unsqueeze(0).to(CONFIG['DEVICE'])
        
        # Get prediction
        usb_pred = loaded_model(powerline_batch).cpu().squeeze().numpy()
        usb_true_np = usb_true.numpy()
        powerline_np = powerline_input.squeeze().numpy()
        
        predictions.append({
            'powerline': powerline_np,
            'predicted': usb_pred,
            'true': usb_true_np,
            'sample_rate': CONFIG['SAMPLE_RATE']
        })
        
        print(f"  Sample {i+1}: Generated prediction ({len(usb_pred):,} samples)")

print(f"\n✓ Generated {len(predictions)} predictions")


# In[ ]:


# Function to create spectrogram (0-20 kHz)
def create_spectrogram_0_20khz(signal, sample_rate, nperseg=2048, noverlap=1024):
    """
    Create spectrogram and focus on 0-20 kHz range.
    """
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    # Focus on 0-20 kHz
    freq_mask = (f >= 0) & (f <= 20000*3)
    f_focused = f[freq_mask]
    Sxx_focused = Sxx_db[freq_mask, :]
    
    return f_focused, t, Sxx_focused

print("✓ Spectrogram function defined (0-20 kHz range)")


# In[ ]:


# Compare spectrograms: Predicted vs True USB signals
num_samples_to_plot = min(3, len(predictions))

fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(16, 5 * num_samples_to_plot))
if num_samples_to_plot == 1:
    axes = axes.reshape(1, -1)

for i in range(num_samples_to_plot):
    pred_signal = predictions[i]['predicted']
    true_signal = predictions[i]['true']
    sample_rate = predictions[i]['sample_rate']
    
    # Create spectrograms (0-20 kHz)
    f_pred, t_pred, spec_pred = create_spectrogram_0_20khz(pred_signal, sample_rate)
    f_true, t_true, spec_true = create_spectrogram_0_20khz(true_signal, sample_rate)
    
    # Plot predicted spectrogram
    im1 = axes[i, 0].imshow(spec_pred, aspect='auto', origin='lower', 
                            extent=[t_pred[0], t_pred[-1], f_pred[0]/1000, f_pred[-1]/1000],
                            cmap='viridis', vmin=-80, vmax=0)
    axes[i, 0].set_title(f'Sample {i+1} - Predicted USB Spectrogram', fontweight='bold', fontsize=14)
    axes[i, 0].set_ylabel('Frequency (kHz)', fontsize=12)
    axes[i, 0].set_xlabel('Time (s)', fontsize=12)
    axes[i, 0].set_ylim(0, 20)
    plt.colorbar(im1, ax=axes[i, 0], label='Power (dB)')
    
    # Plot true spectrogram
    im2 = axes[i, 1].imshow(spec_true, aspect='auto', origin='lower',
                            extent=[t_true[0], t_true[-1], f_true[0]/1000, f_true[-1]/1000],
                            cmap='viridis', vmin=-80, vmax=0)
    axes[i, 1].set_title(f'Sample {i+1} - True USB Spectrogram', fontweight='bold', fontsize=14)
    axes[i, 1].set_ylabel('Frequency (kHz)', fontsize=12)
    axes[i, 1].set_xlabel('Time (s)', fontsize=12)
    axes[i, 1].set_ylim(0, 20)
    plt.colorbar(im2, ax=axes[i, 1], label='Power (dB)')
    
    # Calculate similarity metrics
    corr = np.corrcoef(pred_signal, true_signal)[0, 1]
    mse = np.mean((pred_signal - true_signal) ** 2)
    
    print(f"Sample {i+1} Metrics:")
    print(f"  Correlation: {corr:.4f}")
    print(f"  MSE: {mse:.6f}")

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'spectrogram_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Spectrogram comparison saved to {CONFIG['OUTPUT_DIR']}/spectrogram_comparison.png")


# In[ ]:


# Detailed comparison: Time-domain + Spectrogram side-by-side
sample_idx = 0  # Choose which sample to analyze in detail

pred_signal = predictions[sample_idx]['predicted']
true_signal = predictions[sample_idx]['true']
sample_rate = predictions[sample_idx]['sample_rate']

# Create spectrograms
f_pred, t_pred, spec_pred = create_spectrogram_0_20khz(pred_signal, sample_rate, nperseg=1024, noverlap=512)
f_true, t_true, spec_true = create_spectrogram_0_20khz(true_signal, sample_rate, nperseg=1024, noverlap=512)

# Create comprehensive comparison plot
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Time domain - Predicted
ax1 = fig.add_subplot(gs[0, 0])
time_axis = np.arange(len(pred_signal)) / sample_rate
ax1.plot(time_axis, pred_signal, color='red', linewidth=0.5, alpha=0.8)
ax1.set_title(f'Sample {sample_idx+1} - Predicted USB (Time Domain)', fontweight='bold', fontsize=14)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, time_axis[-1])

# Time domain - True
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_axis, true_signal, color='blue', linewidth=0.5, alpha=0.8)
ax2.set_title(f'Sample {sample_idx+1} - True USB (Time Domain)', fontweight='bold', fontsize=14)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, time_axis[-1])

# Spectrogram - Predicted
ax3 = fig.add_subplot(gs[1, 0])
im1 = ax3.imshow(spec_pred, aspect='auto', origin='lower',
                 extent=[t_pred[0], t_pred[-1], f_pred[0]/1000, f_pred[-1]/1000],
                 cmap='viridis', vmin=-80, vmax=0)
ax3.set_title(f'Predicted USB Spectrogram (0-20 kHz)', fontweight='bold', fontsize=14)
ax3.set_ylabel('Frequency (kHz)', fontsize=12)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylim(0, 20)
plt.colorbar(im1, ax=ax3, label='Power (dB)')

# Spectrogram - True
ax4 = fig.add_subplot(gs[1, 1])
im2 = ax4.imshow(spec_true, aspect='auto', origin='lower',
                 extent=[t_true[0], t_true[-1], f_true[0]/1000, f_true[-1]/1000],
                 cmap='viridis', vmin=-80, vmax=0)
ax4.set_title(f'True USB Spectrogram (0-20 kHz)', fontweight='bold', fontsize=14)
ax4.set_ylabel('Frequency (kHz)', fontsize=12)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylim(0, 20)
plt.colorbar(im2, ax=ax4, label='Power (dB)')

# Difference spectrogram
ax5 = fig.add_subplot(gs[2, :])
spec_diff = spec_pred - spec_true
im3 = ax5.imshow(spec_diff, aspect='auto', origin='lower',
                 extent=[t_pred[0], t_pred[-1], f_pred[0]/1000, f_pred[-1]/1000],
                 cmap='RdBu_r', vmin=-20, vmax=20)
ax5.set_title(f'Difference (Predicted - True) Spectrogram', fontweight='bold', fontsize=14)
ax5.set_ylabel('Frequency (kHz)', fontsize=12)
ax5.set_xlabel('Time (s)', fontsize=12)
ax5.set_ylim(0, 20)
plt.colorbar(im3, ax=ax5, label='Power Difference (dB)')

# Calculate and display metrics
corr = np.corrcoef(pred_signal, true_signal)[0, 1]
mse = np.mean((pred_signal - true_signal) ** 2)
mae = np.mean(np.abs(pred_signal - true_signal))

print(f"\n{'='*60}")
print(f"Sample {sample_idx+1} - Detailed Analysis")
print(f"{'='*60}")
print(f"Correlation Coefficient: {corr:.6f}")
print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"Signal Length: {len(pred_signal):,} samples ({len(pred_signal)/sample_rate:.2f} seconds)")
print(f"Sample Rate: {sample_rate:,} Hz")
print(f"{'='*60}")

plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'detailed_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Detailed comparison saved")


# In[ ]:


# Play all validation samples
print("🔊 Audio Playback for All Samples")
print("=" * 60)

for i in range(min(3, len(predictions))):
    pred_signal = predictions[i]['predicted']
    true_signal = predictions[i]['true']
    sample_rate = predictions[i]['sample_rate']
    
    corr = np.corrcoef(pred_signal, true_signal)[0, 1]
    
    print(f"\n📌 Sample {i+1} (Correlation: {corr:.4f})")
    print("-" * 60)
    
    print("  True USB:")
    display(Audio(true_signal, rate=sample_rate))
    
    print("  Predicted USB:")
    display(Audio(pred_signal, rate=sample_rate))
    
print("\n" + "=" * 60)
print("✓ Audio playback ready")

