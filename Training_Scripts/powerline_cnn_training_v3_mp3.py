#!/usr/bin/env python
# coding: utf-8

# In[5]:


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

# Audio processing
import librosa

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


# In[6]:


# === CONFIGURATION ===
CONFIG = {
    # Data parameters
    'POWERLINE_FOLDER': '/fs/scratch/<allocation>/May29_Alice',  # Powerline binary files
    'MP3_FOLDER': '/fs/scratch/<allocation>/Alice_In_Wonderland_mp3',  # MP3 audio files
    'TIMING_FILE': '/fs/scratch/<allocation>/alice_timing_analysis.txt',  # Timing offsets
    'OUTPUT_DIR': '/fs/scratch/<allocation>/model_runs/powerline_cnn_mp3',
    'POWERLINE_SAMPLE_RATE': 200_000,  # 200 kHz (powerline input)
    'MP3_SAMPLE_RATE': None,  # Will be set from MP3 files (typically 44100 or 48000 Hz)
    'CHUNK_DURATION': 0.5,  # seconds
    
    # Signal processing
    'CARRIER_FREQ': 20000,  # 20 kHz
    'BANDWIDTH': 10000,  # 10 kHz (17.5-22.5 kHz)
    
    # Model parameters
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',  # 'cuda' or 'cpu'
    'BATCH_SIZE': 10,
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

# Create output directory
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Note: MP3_SAMPLE_RATE will be determined from the first MP3 file loaded

# Save configuration (will update after determining sample rate)
config_path = os.path.join(CONFIG['OUTPUT_DIR'], 'config.json')


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

def lowpass_filter(signal, cutoff_freq, sample_rate, order=4):
    """
    Apply lowpass filter to signal.
    """
    nyquist = sample_rate / 2
    cutoff = cutoff_freq / nyquist
    b, a = butter(order, cutoff, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered

def parse_timing_file(timing_file):
    """
    Parse the timing analysis file to extract start offsets for each chapter.
    Returns a dictionary: {chapter_num: start_offset_seconds}
    """
    timing_info = {}
    
    with open(timing_file, 'r') as f:
        content = f.read()
    
    # Parse each chapter section
    import re
    pattern = r'Chapter (\d+):.*?Start Offset:\s+([\d.]+) seconds'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for chapter_num, start_offset in matches:
        timing_info[int(chapter_num)] = float(start_offset)
    
    return timing_info

def load_and_process_chapter(powerline_path, mp3_path, start_offset, config):
    """
    Load a chapter and create fixed-length chunks.
    
    Powerline signal is kept at original 200 kHz sample rate.
    MP3 audio is kept at its native sample rate (typically 44.1 or 48 kHz).
    The model will learn to map from high sample rate powerline to lower sample rate audio.
    
    Args:
        powerline_path: Path to powerline binary file (_real.bin)
        mp3_path: Path to MP3 audio file
        start_offset: Time offset in seconds where MP3 audio starts in powerline signal
        config: Configuration dictionary
    """
    # Load powerline data from binary file (keep at 200 kHz)
    powerline = np.fromfile(powerline_path, dtype=np.float32)
    
    # Load MP3 file using librosa (returns mono audio at native sample rate)
    mp3_audio, mp3_sr = librosa.load(mp3_path, sr=None, mono=True)
    
    # Set MP3 sample rate if not already set
    if config['MP3_SAMPLE_RATE'] is None:
        config['MP3_SAMPLE_RATE'] = mp3_sr
        print(f"  Setting MP3_SAMPLE_RATE to: {mp3_sr} Hz")
    
    powerline_sr = config['POWERLINE_SAMPLE_RATE']
    mp3_sr = config['MP3_SAMPLE_RATE']
    
    print(f"  Powerline: {len(powerline):,} samples at {powerline_sr} Hz")
    print(f"  MP3: {len(mp3_audio):,} samples at {mp3_sr} Hz")
    
    # Apply timing offset: MP3 starts at start_offset seconds in the powerline signal
    offset_samples_powerline = int(start_offset * powerline_sr)
    print(f"  Applying timing offset: {start_offset:.2f}s ({offset_samples_powerline} powerline samples)")
    
    # Align signals by time: trim powerline start by offset
    aligned_powerline = powerline[offset_samples_powerline:]
    aligned_mp3 = mp3_audio
    
    # Calculate aligned duration (use the shorter duration)
    powerline_duration = len(aligned_powerline) / powerline_sr
    mp3_duration = len(aligned_mp3) / mp3_sr
    aligned_duration = min(powerline_duration, mp3_duration)
    
    # Trim both to the same duration
    aligned_powerline = aligned_powerline[:int(aligned_duration * powerline_sr)]
    aligned_mp3 = aligned_mp3[:int(aligned_duration * mp3_sr)]
    
    print(f"  Aligned duration: {aligned_duration:.1f}s")
    print(f"  Aligned powerline: {len(aligned_powerline):,} samples")
    print(f"  Aligned MP3: {len(aligned_mp3):,} samples")
    
    # Create fixed-duration chunks
    chunk_duration = config['CHUNK_DURATION']
    chunk_samples_powerline = int(chunk_duration * powerline_sr)
    chunk_samples_mp3 = int(chunk_duration * mp3_sr)
    
    num_chunks = min(
        len(aligned_powerline) // chunk_samples_powerline,
        len(aligned_mp3) // chunk_samples_mp3
    )
    
    print(f"  Creating {num_chunks} chunks of {chunk_duration}s each")
    print(f"    Powerline: {chunk_samples_powerline} samples/chunk")
    print(f"    MP3: {chunk_samples_mp3} samples/chunk")
    
    chunks = []
    
    for i in range(num_chunks):
        # Extract powerline chunk
        start_idx_pl = i * chunk_samples_powerline
        end_idx_pl = start_idx_pl + chunk_samples_powerline
        powerline_chunk = aligned_powerline[start_idx_pl:end_idx_pl]
        
        # Extract MP3 chunk
        start_idx_mp3 = i * chunk_samples_mp3
        end_idx_mp3 = start_idx_mp3 + chunk_samples_mp3
        mp3_chunk = aligned_mp3[start_idx_mp3:end_idx_mp3]
        
        # Normalize each chunk independently to stabilize training
        powerline_max = np.max(np.abs(powerline_chunk)) + 1e-8
        mp3_max = np.max(np.abs(mp3_chunk)) + 1e-8
        powerline_normalized = (powerline_chunk / powerline_max).astype(np.float32)
        mp3_normalized = (mp3_chunk / mp3_max).astype(np.float32)
        
        chunks.append({
            'powerline_signal': powerline_normalized,
            'mp3_signal': mp3_normalized,
        })
    
    return chunks

print("Data processing functions defined.")


# In[8]:


# === LOAD DATA FROM POWERLINE AND MP3 FILES ===
print("Loading and processing data...\n")

# Parse timing information
print(f"📄 Loading timing information from {CONFIG['TIMING_FILE']}")
timing_info = parse_timing_file(CONFIG['TIMING_FILE'])
print(f"   Found timing info for {len(timing_info)} chapters")
for ch, offset in sorted(timing_info.items()):
    print(f"     Chapter {ch}: offset = {offset:.2f}s")
print()

all_chunks = []
file_count = 0

print(f"📁 Processing powerline folder: {CONFIG['POWERLINE_FOLDER']}")
print(f"🎵 Processing MP3 folder: {CONFIG['MP3_FOLDER']}\n")

# Find all *_real.bin files (powerline signals)
powerline_files = sorted(glob.glob(os.path.join(CONFIG['POWERLINE_FOLDER'], '*_real.bin')))
print(f"   Found {len(powerline_files)} powerline files\n")

for powerline_path in tqdm(powerline_files, desc="  Processing files", leave=False):
    # Extract chapter number from filename (e.g., "Chap_1_real.bin" -> 1)
    powerline_name = os.path.basename(powerline_path)
    import re
    match = re.search(r'Chap_(\d+)_real\.bin', powerline_name)
    
    if not match:
        print(f"  ⚠️  Could not extract chapter number from: {powerline_name}")
        continue
    
    chapter_num = int(match.group(1))
    
    # Find corresponding MP3 file
    mp3_pattern = f"Alice_In_Wonderland_ch_{chapter_num:02d}.mp3"
    mp3_path = os.path.join(CONFIG['MP3_FOLDER'], mp3_pattern)
    
    if not os.path.exists(mp3_path):
        print(f"  ⚠️  MP3 file not found: {mp3_pattern}")
        continue
    
    # Get timing offset
    if chapter_num not in timing_info:
        print(f"  ⚠️  No timing info for chapter {chapter_num}, skipping")
        continue
    
    start_offset = timing_info[chapter_num]
    
    try:
        chunks = load_and_process_chapter(powerline_path, mp3_path, start_offset, CONFIG)
        all_chunks.extend(chunks)
        file_count += len(chunks)
        
        print(f"    ✓ Chapter {chapter_num}: {len(chunks)} chunks (offset: {start_offset:.2f}s)")
    except Exception as e:
        print(f"    ✗ Error processing chapter {chapter_num}: {e}")
        import traceback
        traceback.print_exc()

print()
print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"  Total files processed: {file_count}")
print(f"  TOTAL CHUNKS: {len(all_chunks)} chunks")
print(f"  Powerline sample rate: {CONFIG['POWERLINE_SAMPLE_RATE']} Hz")
print(f"  MP3 sample rate: {CONFIG['MP3_SAMPLE_RATE']} Hz")
print("=" * 70)

if len(all_chunks) > 0:
    print(f"\nSample chunk info:")
    print(f"  Powerline signal shape: {all_chunks[0]['powerline_signal'].shape}")
    print(f"  MP3 signal shape: {all_chunks[0]['mp3_signal'].shape}")
    print(f"  Powerline samples per chunk: {len(all_chunks[0]['powerline_signal']):,}")
    print(f"  MP3 samples per chunk: {len(all_chunks[0]['mp3_signal']):,}")
    
    # Calculate total audio duration
    total_duration_sec = len(all_chunks) * CONFIG['CHUNK_DURATION']
    print(f"\nTotal audio duration: {total_duration_sec / 60:.1f} minutes ({total_duration_sec / 3600:.2f} hours)")
    
    # Now save the final config with determined sample rates
    with open(os.path.join(CONFIG['OUTPUT_DIR'], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\n✓ Configuration saved")


# ## 3. Dataset and DataLoader

# In[9]:


class PowerlineDataset(Dataset):
    """
    Dataset for powerline signal to MP3 audio mapping.
    Input: Powerline signal at 200 kHz
    Output: MP3 audio at native rate (44.1 or 48 kHz)
    """
    def __init__(self, chunks):
        self.chunks = chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Powerline signal: (1, samples) - add channel dimension for Conv1d
        powerline = torch.FloatTensor(chunk['powerline_signal']).unsqueeze(0)
        
        # MP3 signal: (samples,)
        mp3 = torch.FloatTensor(chunk['mp3_signal'])
        
        return powerline, mp3

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

# In[10]:


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

class PowerlineToMP3CNN(nn.Module):
    """
    Multi-rate CNN that maps from high sample rate powerline (200 kHz) to lower sample rate audio (44.1/48 kHz).
    
    The model performs downsampling through strided convolutions to match the sample rate reduction,
    while using spectral attention to focus on relevant frequency regions.
    """
    def __init__(self, input_samples, output_samples, n_fft=2048, hop_length=512, time_channels=[64,128,256,512], attn_embed=256):
        super(PowerlineToMP3CNN, self).__init__()
        self.input_samples = input_samples
        self.output_samples = output_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Calculate required downsampling ratio
        self.downsample_ratio = input_samples / output_samples
        
        # Time encoder with aggressive downsampling
        # Each stride=2 layer reduces by 2x, so we need log2(downsample_ratio) such layers
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

        # Decoder (transpose convs) - fewer upsampling layers since we want lower output rate
        dec_layers = []
        rev_channels = list(reversed(time_channels))
        in_ch = time_channels[-1]
        
        # Only upsample by 2x per layer, fewer layers than encoder
        num_upsample_layers = len(rev_channels) - 2  # Fewer upsamples = net downsampling
        
        for i, ch in enumerate(rev_channels[:-1]):
            if i < num_upsample_layers:
                dec_layers.append(nn.ConvTranspose1d(in_ch, ch, kernel_size=16, stride=2, padding=7))
            else:
                dec_layers.append(nn.Conv1d(in_ch, ch, kernel_size=15, padding=7))
            dec_layers.append(nn.BatchNorm1d(ch))
            dec_layers.append(nn.ReLU())
            dec_layers.append(ResidualBlock1D(ch))
            in_ch = ch
            
        dec_layers.append(nn.Conv1d(in_ch, 32, kernel_size=15, padding=7))
        dec_layers.append(nn.BatchNorm1d(32))
        dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Conv1d(32, 1, kernel_size=15, padding=7))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x: (B, 1, L_high) where L_high is high sample rate length
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
        
        # Final resize to exact output sample count
        if out.size(1) != self.output_samples:
            out = F.interpolate(out.unsqueeze(1), size=self.output_samples, mode='linear', align_corners=False).squeeze(1)
        
        return out

# Get dimensions from sample data
sample_powerline = train_chunks[0]['powerline_signal']
sample_mp3 = train_chunks[0]['mp3_signal']
input_samples = len(sample_powerline)
output_samples = len(sample_mp3)

print(f"Model dimensions:")
print(f"  Input (powerline): (1, {input_samples}) samples at 200 kHz")
print(f"  Output (MP3): ({output_samples},) samples at {CONFIG['MP3_SAMPLE_RATE']} Hz")
print(f"  Sample rate ratio: {input_samples / output_samples:.2f}x")

# Create the multi-rate model
model = PowerlineToMP3CNN(input_samples, output_samples, n_fft=2048, hop_length=512)
model = model.to(CONFIG['DEVICE'])

print(f"\n✓ PowerlineToMP3CNN model created")
print(f"✓ Model moved to {CONFIG['DEVICE']}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB (float32)")


# ## 5. Training Setup

# In[11]:


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

# In[12]:


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Train for one epoch with OneCycleLR scheduler.
    """
    model.train()
    total_loss = 0.0
    
    for powerline_signals, mp3_signals in tqdm(dataloader, desc="Training", leave=False):
        powerline_signals = powerline_signals.to(device)
        mp3_signals = mp3_signals.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(powerline_signals)
        loss = criterion(outputs, mp3_signals)
        
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
        for powerline_signals, mp3_signals in tqdm(dataloader, desc="Validation", leave=False):
            powerline_signals = powerline_signals.to(device)
            mp3_signals = mp3_signals.to(device)
            
            outputs = model(powerline_signals)
            loss = criterion(outputs, mp3_signals)
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
        powerline_input, mp3_true = val_dataset[i]
        powerline_input_batch = powerline_input.unsqueeze(0).to(CONFIG['DEVICE'])
        
        # Predict
        mp3_pred = model(powerline_input_batch).cpu().squeeze().numpy()
        mp3_true = mp3_true.numpy()
        powerline_np = powerline_input.squeeze().numpy()
        
        # Time axes (different sample rates)
        time_axis_powerline = np.arange(len(powerline_np)) / CONFIG['POWERLINE_SAMPLE_RATE']
        time_axis_mp3 = np.arange(len(mp3_true)) / CONFIG['MP3_SAMPLE_RATE']
        
        # Plot powerline input signal
        axes[i, 0].plot(time_axis_powerline, powerline_np, color='green', linewidth=0.5, alpha=0.8)
        axes[i, 0].set_title(f'Sample {i+1} - Input Powerline Signal ({CONFIG["POWERLINE_SAMPLE_RATE"]/1000:.0f} kHz)', fontweight='bold')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot predicted signal
        axes[i, 1].plot(time_axis_mp3, mp3_pred, color='red', linewidth=0.5, alpha=0.8, label='Predicted')
        axes[i, 1].set_title(f'Sample {i+1} - Predicted MP3 ({CONFIG["MP3_SAMPLE_RATE"]/1000:.1f} kHz)', fontweight='bold')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
        
        # Plot true signal
        axes[i, 2].plot(time_axis_mp3, mp3_true, color='blue', linewidth=0.5, alpha=0.8, label='Ground Truth')
        axes[i, 2].set_title(f'Sample {i+1} - True MP3 ({CONFIG["MP3_SAMPLE_RATE"]/1000:.1f} kHz)', fontweight='bold')
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].legend()
        
        # Calculate correlation
        corr = np.corrcoef(mp3_pred, mp3_true)[0, 1]
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

