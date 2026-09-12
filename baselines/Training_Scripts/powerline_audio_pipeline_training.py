#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Consolidated imports
import os
import glob
from pathlib import Path
import json
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Audio processing
import librosa
import soundfile as sf

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
print('✓ Consolidated imports loaded')


# In[ ]:


# === CONFIGURATION ===
CONFIG = {
    # Data parameters
    'DATA_FOLDER': '/fs/scratch/<allocation>/May29_Alice',
    'AUDIO_FOLDER': '/fs/scratch/<allocation>/Alice_In_Wonderland_mp3',
    'TIMING_FILE': '/fs/scratch/<allocation>/alice_timing_analysis.txt',
    'OUTPUT_BASE_DIR': '/fs/scratch/<allocation>/model_runs',
    'MODEL_NAME': 'powerline_audio_pipeline',
    
    # Sample rates
    'POWERLINE_SR': 200_000,  # Powerline/USB sample rate (200 kHz)
    'USB_SR': 200_000,        # USB at same rate as powerline
    'AUDIO_SR': 22050,        # Audio sample rate from MP3 files
    
    # Signal processing
    'CHUNK_DURATION': 0.5,  # seconds (longer chunks for audio reconstruction)
    'USE_CHAPTERS': [1, 2, 4, 5, 6, 7, 8, 10, 11],  # Available chapters
    
    # Model parameters (multi-stage pipeline)
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'BASE_CHANNELS_STAGE_A': 32,
    'BASE_CHANNELS_STAGE_B': 32,
    'LATENT_DIM': 256,
    'NUM_WAVELET_SCALES': 25,
    'USE_DILATED_CONVS': True,
    
    # Training options
    'BATCH_SIZE': 2,
    'NUM_EPOCHS': 15,
    'LEARNING_RATE': 1e-5,
    'TRAIN_SPLIT': 0.8,
    'USE_GPU': True,
    'SAVE_EVERY': 5,
    'EARLY_STOPPING_PATIENCE': 15,
    
    # Training stages
    'TRAIN_STAGE': 'end_to_end',  # 'stage_a', 'stage_b', or 'end_to_end'
}

# Override device if USE_GPU is False
if not CONFIG['USE_GPU']:
    CONFIG['DEVICE'] = 'cpu'

# Create unique timestamped output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
CONFIG['OUTPUT_DIR'] = os.path.join(
    CONFIG['OUTPUT_BASE_DIR'], 
    f"{CONFIG['MODEL_NAME']}_{CONFIG['TRAIN_STAGE']}_{timestamp}"
)
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

print("=" * 80)
print("MULTI-STAGE POWERLINE-TO-AUDIO PIPELINE")
print("=" * 80)
print(f"\n📁 Output directory: {CONFIG['OUTPUT_DIR']}")
print(f"\nSample Rates:")
print(f"  Powerline: {CONFIG['POWERLINE_SR']:,} Hz")
print(f"  USB:       {CONFIG['USB_SR']:,} Hz")
print(f"  Audio:     {CONFIG['AUDIO_SR']:,} Hz")
print(f"\nTraining Stage: {CONFIG['TRAIN_STAGE'].upper()}")
print(f"Device: {CONFIG['DEVICE'].upper()}")

# Save configuration
with open(os.path.join(CONFIG['OUTPUT_DIR'], 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=2)

print(f"\n✓ Configuration saved")
print("=" * 80)


# ## 2. Load Timing Information and Audio Files
# 
# Parse timing offsets from alice_timing_analysis.txt to properly align MP3 audio with powerline/USB signals.

# In[3]:


def parse_timing_file(timing_file_path):
    """
    Parse alice_timing_analysis.txt to extract timing offsets for each chapter.
    Returns a dictionary: {chapter_num: {'start_offset': float, 'stop_time': float, ...}}
    """
    timing_info = {}
    
    with open(timing_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the summary table
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('Chapter'):
            continue
        if 'SUMMARY' in line or 'DETAILED' in line:
            continue
            
        # Try to parse chapter line from summary table
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            chapter_num = int(parts[0])
            # Format: Chapter MP3(s) BIN(s) Start(s) Stop(s)
            try:
                start_offset = float(parts[3])
                stop_time = float(parts[4])
                timing_info[chapter_num] = {
                    'start_offset': start_offset,
                    'stop_time': stop_time
                }
            except (ValueError, IndexError):
                continue
    
    return timing_info

# Load timing information
timing_info = parse_timing_file(CONFIG['TIMING_FILE'])
print(f"✓ Loaded timing info for {len(timing_info)} chapters:")
for ch, info in sorted(timing_info.items()):
    print(f"  Chapter {ch:02d}: start={info['start_offset']:.2f}s, stop={info['stop_time']:.2f}s")


# In[5]:


def load_chapter_data(chapter_num, data_folder, audio_folder, timing_info, config):
    """
    Load aligned powerline, USB, and audio data for a single chapter.
    
    Returns chunks with:
    - powerline_signal: (chunk_samples,) at POWERLINE_SR
    - usb_signal: (chunk_samples,) at USB_SR
    - audio_signal: (audio_chunk_samples,) at AUDIO_SR
    """
    # Construct file paths
    # Binary files: Chap_1_real.bin format (no zero padding)
    powerline_path = os.path.join(data_folder, f'Chap_{chapter_num}_real.bin')
    usb_path = os.path.join(data_folder, f'Chap_{chapter_num}_img.bin')
    # Audio files: Alice_In_Wonderland_ch_01.mp3 format (with zero padding)
    audio_path = os.path.join(audio_folder, f'Alice_In_Wonderland_ch_{chapter_num:02d}.mp3')
    
    # Check files exist
    if not all([os.path.exists(p) for p in [powerline_path, usb_path, audio_path]]):
        missing = []
        if not os.path.exists(powerline_path):
            missing.append(f"powerline: {os.path.basename(powerline_path)}")
        if not os.path.exists(usb_path):
            missing.append(f"USB: {os.path.basename(usb_path)}")
        if not os.path.exists(audio_path):
            missing.append(f"audio: {os.path.basename(audio_path)}")
        print(f"  ⚠️  Missing files for chapter {chapter_num}: {', '.join(missing)}")
        return []
    
    # Get timing info
    if chapter_num not in timing_info:
        print(f"  ⚠️  No timing info for chapter {chapter_num}")
        return []
    
    start_offset = timing_info[chapter_num]['start_offset']
    stop_time = timing_info[chapter_num]['stop_time']
    
    # Load binary signals (powerline and USB)
    powerline_raw = np.fromfile(powerline_path, dtype=np.float32)
    usb_raw = np.fromfile(usb_path, dtype=np.float32)
    
    # Load audio (MP3)
    audio_raw, audio_sr_loaded = librosa.load(audio_path, sr=config['AUDIO_SR'])
    
    print(f"  Chapter {chapter_num:02d}:")
    print(f"    Powerline/USB samples: {len(powerline_raw):,} ({len(powerline_raw)/config['POWERLINE_SR']:.1f}s)")
    print(f"    Audio samples: {len(audio_raw):,} ({len(audio_raw)/config['AUDIO_SR']:.1f}s)")
    print(f"    Timing: start={start_offset:.2f}s, stop={stop_time:.2f}s")
    
    # Extract aligned segments
    # Powerline/USB: from start_offset to stop_time
    start_idx_pl = int(start_offset * config['POWERLINE_SR'])
    stop_idx_pl = int(stop_time * config['POWERLINE_SR'])
    
    powerline = powerline_raw[start_idx_pl:stop_idx_pl]
    usb = usb_raw[start_idx_pl:stop_idx_pl]
    
    # Audio: entire MP3 (already aligned, starts at t=0 corresponding to powerline start_offset)
    audio = audio_raw
    
    # Ensure lengths match (audio might be slightly different due to encoding)
    expected_audio_len = int((stop_time - start_offset) * config['AUDIO_SR'])
    audio = audio[:expected_audio_len]  # Truncate if longer
    
    # Normalize signals
    powerline = powerline / (np.max(np.abs(powerline)) + 1e-8)
    usb = usb / (np.max(np.abs(usb)) + 1e-8)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    print(f"    After alignment: PL/USB={len(powerline):,}, Audio={len(audio):,}")
    
    # Create fixed-duration chunks
    chunk_samples_pl = int(config['CHUNK_DURATION'] * config['POWERLINE_SR'])
    chunk_samples_audio = int(config['CHUNK_DURATION'] * config['AUDIO_SR'])
    
    num_chunks = min(len(powerline) // chunk_samples_pl, len(audio) // chunk_samples_audio)
    print(f"    Creating {num_chunks} chunks of {config['CHUNK_DURATION']}s each")
    
    chunks = []
    for i in range(num_chunks):
        # Powerline/USB chunks
        start_pl = i * chunk_samples_pl
        end_pl = start_pl + chunk_samples_pl
        
        # Audio chunks
        start_audio = i * chunk_samples_audio
        end_audio = start_audio + chunk_samples_audio
        
        chunks.append({
            'powerline_signal': powerline[start_pl:end_pl].astype(np.float32),
            'usb_signal': usb[start_pl:end_pl].astype(np.float32),
            'audio_signal': audio[start_audio:end_audio].astype(np.float32),
            'chapter': chapter_num
        })
    
    return chunks

# === LOAD ALL CHAPTERS ===
print("\n" + "=" * 80)
print("LOADING ALIGNED DATA")
print("=" * 80)

all_chunks = []

for chapter_num in CONFIG['USE_CHAPTERS']:
    try:
        chunks = load_chapter_data(
            chapter_num,
            CONFIG['DATA_FOLDER'],
            CONFIG['AUDIO_FOLDER'],
            timing_info,
            CONFIG
        )
        all_chunks.extend(chunks)
        print(f"    ✓ Loaded {len(chunks)} chunks\n")
    except Exception as e:
        print(f"    ✗ Error loading chapter {chapter_num}: {e}\n")

print("=" * 80)
print(f"TOTAL CHUNKS: {len(all_chunks)}")
print("=" * 80)

if len(all_chunks) > 0:
    print(f"\nSample chunk shapes:")
    print(f"  Powerline: {all_chunks[0]['powerline_signal'].shape} @ {CONFIG['POWERLINE_SR']} Hz")
    print(f"  USB:       {all_chunks[0]['usb_signal'].shape} @ {CONFIG['USB_SR']} Hz")
    print(f"  Audio:     {all_chunks[0]['audio_signal'].shape} @ {CONFIG['AUDIO_SR']} Hz")
    
    total_duration = len(all_chunks) * CONFIG['CHUNK_DURATION']
    print(f"\nTotal duration: {total_duration/60:.1f} minutes")


# ## 3. Dataset and DataLoader
# 
# Multi-modal dataset with powerline, USB, and audio signals.

# In[6]:


class MultiModalPowerlineDataset(Dataset):
    """
    Dataset for multi-stage powerline → USB → audio pipeline.
    Returns aligned powerline, USB, and audio chunks.
    """
    def __init__(self, chunks):
        self.chunks = chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Powerline: (1, T_pl) - add channel dimension
        powerline = torch.FloatTensor(chunk['powerline_signal']).unsqueeze(0)
        
        # USB: (T_usb)
        usb = torch.FloatTensor(chunk['usb_signal'])
        
        # Audio: (T_audio)
        audio = torch.FloatTensor(chunk['audio_signal'])
        
        return powerline, usb, audio

# Split into train and validation
num_train = int(len(all_chunks) * CONFIG['TRAIN_SPLIT'])
train_chunks = all_chunks[:num_train]
val_chunks = all_chunks[num_train:]

print(f"Dataset split:")
print(f"  Training:   {len(train_chunks)} chunks")
print(f"  Validation: {len(val_chunks)} chunks")

# Create datasets
train_dataset = MultiModalPowerlineDataset(train_chunks)
val_dataset = MultiModalPowerlineDataset(val_chunks)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['BATCH_SIZE'], 
    shuffle=True, 
    num_workers=0
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['BATCH_SIZE'], 
    shuffle=False, 
    num_workers=0
)

print(f"\n✓ DataLoaders created (batch size: {CONFIG['BATCH_SIZE']})")


# ## 4. Multi-Stage Pipeline Model
# 
# Load the 3-stage powerline-to-audio pipeline:
# - **Stage A**: Powerline → USB (with latent features)
# - **Stage B**: USB + Latent → Audio
# - **Stage C**: End-to-end training with multi-task loss

# In[7]:


# Import the multi-stage pipeline
from powerline_audio_pipeline import (
    PowerlineToAudioPipeline,
    MultiStageLoss,
    PowerlineToUSBEncoder,
    USBToAudioDecoder
)

print("=" * 80)
print("MULTI-STAGE POWERLINE-TO-AUDIO PIPELINE")
print("=" * 80)

# Create the full pipeline
model = PowerlineToAudioPipeline(
    powerline_sr=CONFIG['POWERLINE_SR'],
    usb_sr=CONFIG['USB_SR'],
    audio_sr=CONFIG['AUDIO_SR'],
    base_channels_stageA=CONFIG['BASE_CHANNELS_STAGE_A'],
    base_channels_stageB=CONFIG['BASE_CHANNELS_STAGE_B'],
    latent_dim=CONFIG['LATENT_DIM'],
    num_wavelet_scales=CONFIG['NUM_WAVELET_SCALES'],
    use_dilated_convs=CONFIG['USE_DILATED_CONVS']
)

# Move to device
model = model.to(CONFIG['DEVICE'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
stage_a_params = sum(p.numel() for p in model.stage_a.parameters())
stage_b_params = sum(p.numel() for p in model.stage_b.parameters())

print(f"\nModel Architecture:")
print(f"  Stage A (Powerline → USB):")
print(f"    - Wavelet scales: {CONFIG['NUM_WAVELET_SCALES']}")
print(f"    - Base channels: {CONFIG['BASE_CHANNELS_STAGE_A']}")
print(f"    - Latent dim: {CONFIG['LATENT_DIM']}")
print(f"    - Parameters: {stage_a_params:,}")
print(f"\n  Stage B (USB + Latent → Audio):")
print(f"    - Base channels: {CONFIG['BASE_CHANNELS_STAGE_B']}")
print(f"    - Dilated convs: {CONFIG['USE_DILATED_CONVS']}")
print(f"    - Parameters: {stage_b_params:,}")
print(f"\n  Total:")
print(f"    - Parameters: {total_params:,}")
print(f"    - Model size: {total_params * 4 / 1024**2:.1f} MB")
print(f"    - Device: {CONFIG['DEVICE'].upper()}")

# Configure training mode
print(f"\n  Training Mode: {CONFIG['TRAIN_STAGE'].upper()}")
if CONFIG['TRAIN_STAGE'] == 'stage_a':
    # Train only Stage A (powerline → USB)
    model.freeze_stage_b()
    print("    - Stage B FROZEN")
    print("    - Training: Powerline → USB only")
elif CONFIG['TRAIN_STAGE'] == 'stage_b':
    # Train only Stage B (USB → audio)
    model.freeze_stage_a()
    print("    - Stage A FROZEN")
    print("    - Training: USB → Audio only")
else:  # end_to_end
    # Train both stages
    model.unfreeze_stage_a()
    model.unfreeze_stage_b()
    print("    - Both stages TRAINABLE")
    print("    - Training: End-to-end Powerline → Audio")

trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    - Trainable parameters: {trainable_now:,}")

print("=" * 80)

# Test forward pass
print("\nTesting forward pass...")
sample_pl, sample_usb, sample_audio = train_dataset[0]
sample_pl_batch = sample_pl.unsqueeze(0).to(CONFIG['DEVICE'])

with torch.no_grad():
    if CONFIG['TRAIN_STAGE'] == 'stage_a':
        outputs = model(sample_pl_batch, mode='stage_a')
        print(f"  Input (powerline): {sample_pl_batch.shape}")
        print(f"  Output (USB pred): {outputs['usb_pred'].shape}")
        print(f"  Latent features:   {outputs['latent'].shape}")
    else:
        outputs = model(sample_pl_batch, mode='end_to_end')
        print(f"  Input (powerline): {sample_pl_batch.shape}")
        print(f"  Output (USB pred): {outputs['usb_pred'].shape}")
        print(f"  Output (audio pred): {outputs['audio_pred'].shape}")
        print(f"  Latent features:   {outputs['latent'].shape}")

print("✓ Forward pass successful!")


# ## 5. Loss Functions and Optimizer

# In[ ]:


# Create multi-stage loss function
if CONFIG['TRAIN_STAGE'] == 'stage_a':
    # Stage A: only USB loss
    lambda_usb = 1.0
    lambda_audio = 0.0
elif CONFIG['TRAIN_STAGE'] == 'stage_b':
    # Stage B: only audio loss
    lambda_usb = 0.0
    lambda_audio = 1.0
else:  # end_to_end
    # Both losses
    lambda_usb = 1.0
    lambda_audio = 10.0  # Audio loss weighted higher

criterion = MultiStageLoss(
    lambda_usb=lambda_usb,
    lambda_audio=lambda_audio,
    use_stft_loss=True,
    stft_scales=[
        {'n_fft': 2048, 'hop_length': 512, 'win_length': 2048},
        {'n_fft': 1024, 'hop_length': 256, 'win_length': 1024},
        {'n_fft': 512, 'hop_length': 128, 'win_length': 512}
    ]
)

# Optimizer
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['LEARNING_RATE'],
    weight_decay=1e-5
)

# Learning rate scheduler
steps_per_epoch = len(train_loader)
total_steps = CONFIG['NUM_EPOCHS'] * steps_per_epoch

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG['LEARNING_RATE'],
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

print(f"Loss Configuration:")
print(f"  λ_USB:   {lambda_usb}")
print(f"  λ_Audio: {lambda_audio}")
print(f"  STFT Loss: Enabled (3 scales)")
print(f"\nOptimizer: AdamW")
print(f"  Learning rate: {CONFIG['LEARNING_RATE']}")
print(f"  Weight decay: 1e-5")
print(f"\nScheduler: OneCycleLR")
print(f"  Total steps: {total_steps:,}")
print(f"  Warmup: 30%")

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_usb_loss': [],
    'train_audio_loss': [],
    'val_usb_loss': [],
    'val_audio_loss': [],
    'learning_rate': []
}

best_val_loss = float('inf')
patience_counter = 0

print(f"\n✓ Training setup complete")


# ## 6. Training Loop

# In[ ]:


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, train_stage):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_usb_loss = 0.0
    total_audio_loss = 0.0
    
    for powerline, usb_true, audio_true in tqdm(dataloader, desc="Training", leave=False):
        powerline = powerline.to(device)
        usb_true = usb_true.to(device)
        audio_true = audio_true.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if train_stage == 'stage_a':
            # Stage A only: powerline → USB
            outputs = model(powerline, mode='stage_a')
            targets = {'usb_true': usb_true}
        elif train_stage == 'stage_b':
            # Stage B only: USB → audio (with teacher forcing)
            audio_pred = model.forward_stage_b_only(usb_true, powerline)
            outputs = {'audio_pred': audio_pred}
            targets = {'audio_true': audio_true}
        else:  # end_to_end
            # Full pipeline: powerline → USB → audio
            outputs = model(powerline, mode='end_to_end')
            targets = {'usb_true': usb_true, 'audio_true': audio_true}
        
        # Compute loss
        losses = criterion(outputs, targets)
        loss = losses['total_loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_usb_loss += losses.get('usb_loss', torch.tensor(0.0)).item()
        total_audio_loss += losses.get('audio_loss', torch.tensor(0.0)).item()
    
    n = len(dataloader)
    return {
        'total': total_loss / n,
        'usb': total_usb_loss / n,
        'audio': total_audio_loss / n
    }

def validate(model, dataloader, criterion, device, train_stage):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_usb_loss = 0.0
    total_audio_loss = 0.0
    
    with torch.no_grad():
        for powerline, usb_true, audio_true in tqdm(dataloader, desc="Validation", leave=False):
            powerline = powerline.to(device)
            usb_true = usb_true.to(device)
            audio_true = audio_true.to(device)
            
            # Forward pass
            if train_stage == 'stage_a':
                outputs = model(powerline, mode='stage_a')
                targets = {'usb_true': usb_true}
            elif train_stage == 'stage_b':
                audio_pred = model.forward_stage_b_only(usb_true, powerline)
                outputs = {'audio_pred': audio_pred}
                targets = {'audio_true': audio_true}
            else:  # end_to_end
                outputs = model(powerline, mode='end_to_end')
                targets = {'usb_true': usb_true, 'audio_true': audio_true}
            
            # Compute loss
            losses = criterion(outputs, targets)
            total_loss += losses['total_loss'].item()
            total_usb_loss += losses.get('usb_loss', torch.tensor(0.0)).item()
            total_audio_loss += losses.get('audio_loss', torch.tensor(0.0)).item()
    
    n = len(dataloader)
    return {
        'total': total_loss / n,
        'usb': total_usb_loss / n,
        'audio': total_audio_loss / n
    }

print("✓ Training functions defined")


# In[ ]:


# === MAIN TRAINING LOOP ===
print("\n" + "=" * 80)
print(f"TRAINING: {CONFIG['TRAIN_STAGE'].upper()}")
print("=" * 80)
print(f"Device: {CONFIG['DEVICE'].upper()}")
print(f"Epochs: {CONFIG['NUM_EPOCHS']}")
print(f"Batch size: {CONFIG['BATCH_SIZE']}")
print("=" * 80)

start_time = datetime.now()

for epoch in range(CONFIG['NUM_EPOCHS']):
    epoch_start = datetime.now()
    
    # Train
    train_losses = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, 
        CONFIG['DEVICE'], CONFIG['TRAIN_STAGE']
    )
    
    # Validate
    val_losses = validate(
        model, val_loader, criterion, 
        CONFIG['DEVICE'], CONFIG['TRAIN_STAGE']
    )
    
    # Record history
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_losses['total'])
    history['val_loss'].append(val_losses['total'])
    history['train_usb_loss'].append(train_losses['usb'])
    history['train_audio_loss'].append(train_losses['audio'])
    history['val_usb_loss'].append(val_losses['usb'])
    history['val_audio_loss'].append(val_losses['audio'])
    history['learning_rate'].append(current_lr)
    
    # Print progress
    epoch_time = (datetime.now() - epoch_start).total_seconds()
    print(f"\nEpoch [{epoch+1}/{CONFIG['NUM_EPOCHS']}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_losses['total']:.6f} (USB: {train_losses['usb']:.6f}, Audio: {train_losses['audio']:.6f})")
    print(f"  Val Loss:   {val_losses['total']:.6f} (USB: {val_losses['usb']:.6f}, Audio: {val_losses['audio']:.6f})")
    print(f"  LR:         {current_lr:.6f}")
    
    # Save best model
    if val_losses['total'] < best_val_loss:
        best_val_loss = val_losses['total']
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
        }, os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pt'))
        print(f"  ✓ Best model saved (val_loss: {val_losses['total']:.6f})")
    else:
        patience_counter += 1
    
    # Save periodic checkpoint
    if (epoch + 1) % CONFIG['SAVE_EVERY'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
        }, os.path.join(CONFIG['OUTPUT_DIR'], f'checkpoint_epoch_{epoch+1}.pt'))
        print(f"  ✓ Checkpoint saved")
    
    # Early stopping
    if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
        print(f"\n⚠️  Early stopping triggered (patience: {CONFIG['EARLY_STOPPING_PATIENCE']})")
        break

total_time = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best val loss: {best_val_loss:.6f}")
print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
print("=" * 80)

# Save training history
np.save(os.path.join(CONFIG['OUTPUT_DIR'], 'training_history.npy'), history)
print(f"\n✓ Training history saved")


# ## 7. Training Visualization

# In[ ]:


# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Total loss
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Total Loss', fontsize=12)
axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# USB loss
axes[0, 1].plot(history['train_usb_loss'], label='Train USB', linewidth=2)
axes[0, 1].plot(history['val_usb_loss'], label='Val USB', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('USB Loss', fontsize=12)
axes[0, 1].set_title('USB Reconstruction Loss', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Audio loss
axes[1, 0].plot(history['train_audio_loss'], label='Train Audio', linewidth=2, color='green')
axes[1, 0].plot(history['val_audio_loss'], label='Val Audio', linewidth=2, color='orange')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Audio Loss', fontsize=12)
axes[1, 0].set_title('Audio Reconstruction Loss', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning rate
axes[1, 1].plot(history['learning_rate'], linewidth=2, color='red')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Training curves saved to {CONFIG['OUTPUT_DIR']}/training_curves.png")


# ## 8. Model Evaluation
# 
# Load best model and evaluate predictions.

# In[ ]:


# Load best model
checkpoint = torch.load(os.path.join(CONFIG['OUTPUT_DIR'], 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Best model loaded")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val loss: {checkpoint['val_loss']:.6f}")

# Evaluate on validation samples
num_samples = min(3, len(val_dataset))

if CONFIG['TRAIN_STAGE'] == 'stage_a':
    # Evaluate Stage A: powerline → USB
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            powerline, usb_true, _ = val_dataset[i]
            powerline_batch = powerline.unsqueeze(0).to(CONFIG['DEVICE'])
            
            outputs = model(powerline_batch, mode='stage_a')
            usb_pred = outputs['usb_pred'].cpu().squeeze().numpy()
            usb_true = usb_true.numpy()
            powerline_np = powerline.squeeze().numpy()
            
            time_axis = np.arange(len(usb_true)) / CONFIG['USB_SR']
            
            # Powerline input
            axes[i, 0].plot(time_axis, powerline_np, color='green', linewidth=0.5, alpha=0.8)
            axes[i, 0].set_title(f'Sample {i+1} - Powerline Input', fontweight='bold')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # USB prediction
            axes[i, 1].plot(time_axis, usb_pred, color='red', linewidth=0.5, alpha=0.8)
            axes[i, 1].set_title(f'Sample {i+1} - USB Predicted', fontweight='bold')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].grid(True, alpha=0.3)
            
            # USB ground truth
            axes[i, 2].plot(time_axis, usb_true, color='blue', linewidth=0.5, alpha=0.8)
            axes[i, 2].set_title(f'Sample {i+1} - USB Ground Truth', fontweight='bold')
            axes[i, 2].set_ylabel('Amplitude')
            axes[i, 2].set_xlabel('Time (s)')
            axes[i, 2].grid(True, alpha=0.3)
            
            corr = np.corrcoef(usb_pred, usb_true)[0, 1]
            print(f"Sample {i+1} - USB Correlation: {corr:.4f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions_stage_a.png'), dpi=150, bbox_inches='tight')
    plt.show()

else:
    # Evaluate end-to-end or Stage B: include audio
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            powerline, usb_true, audio_true = val_dataset[i]
            powerline_batch = powerline.unsqueeze(0).to(CONFIG['DEVICE'])
            
            if CONFIG['TRAIN_STAGE'] == 'stage_b':
                # Use true USB
                usb_batch = usb_true.unsqueeze(0).to(CONFIG['DEVICE'])
                audio_pred = model.forward_stage_b_only(usb_batch, powerline_batch)
                usb_pred = usb_true.numpy()  # No USB prediction in stage B
            else:
                # End-to-end
                outputs = model(powerline_batch, mode='end_to_end')
                usb_pred = outputs['usb_pred'].cpu().squeeze().numpy()
                audio_pred = outputs['audio_pred']
            
            audio_pred = audio_pred.cpu().squeeze().numpy()
            audio_true = audio_true.numpy()
            powerline_np = powerline.squeeze().numpy()
            
            time_axis_pl = np.arange(len(usb_pred)) / CONFIG['USB_SR']
            time_axis_audio = np.arange(len(audio_true)) / CONFIG['AUDIO_SR']
            
            # Powerline
            axes[i, 0].plot(time_axis_pl, powerline_np, color='green', linewidth=0.5, alpha=0.8)
            axes[i, 0].set_title(f'Sample {i+1} - Powerline Input', fontweight='bold')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # USB (predicted or true)
            axes[i, 1].plot(time_axis_pl, usb_pred, color='red', linewidth=0.5, alpha=0.8)
            label = 'USB Predicted' if CONFIG['TRAIN_STAGE'] == 'end_to_end' else 'USB Input (GT)'
            axes[i, 1].set_title(f'Sample {i+1} - {label}', fontweight='bold')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Audio predicted
            axes[i, 2].plot(time_axis_audio, audio_pred, color='orange', linewidth=0.5, alpha=0.8)
            axes[i, 2].set_title(f'Sample {i+1} - Audio Predicted', fontweight='bold')
            axes[i, 2].set_ylabel('Amplitude')
            axes[i, 2].set_xlabel('Time (s)')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Audio ground truth
            axes[i, 3].plot(time_axis_audio, audio_true, color='blue', linewidth=0.5, alpha=0.8)
            axes[i, 3].set_title(f'Sample {i+1} - Audio Ground Truth', fontweight='bold')
            axes[i, 3].set_ylabel('Amplitude')
            axes[i, 3].set_xlabel('Time (s)')
            axes[i, 3].grid(True, alpha=0.3)
            
            audio_corr = np.corrcoef(audio_pred, audio_true)[0, 1]
            print(f"Sample {i+1} - Audio Correlation: {audio_corr:.4f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'predictions_audio.png'), dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n✓ Predictions saved to {CONFIG['OUTPUT_DIR']}")


# ## 9. Audio Playback Test
# 
# Generate and listen to predicted audio.

# In[ ]:


if CONFIG['TRAIN_STAGE'] != 'stage_a':
    # Generate audio samples
    print("Generating audio samples...")
    
    num_audio_samples = min(3, len(val_dataset))
    audio_output_dir = os.path.join(CONFIG['OUTPUT_DIR'], 'audio_samples')
    os.makedirs(audio_output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_audio_samples):
            powerline, usb_true, audio_true = val_dataset[i]
            powerline_batch = powerline.unsqueeze(0).to(CONFIG['DEVICE'])
            
            if CONFIG['TRAIN_STAGE'] == 'stage_b':
                usb_batch = usb_true.unsqueeze(0).to(CONFIG['DEVICE'])
                audio_pred = model.forward_stage_b_only(usb_batch, powerline_batch)
            else:
                outputs = model(powerline_batch, mode='end_to_end')
                audio_pred = outputs['audio_pred']
            
            audio_pred_np = audio_pred.cpu().squeeze().numpy()
            audio_true_np = audio_true.numpy()
            
            # Save predicted audio
            pred_path = os.path.join(audio_output_dir, f'sample_{i+1}_predicted.wav')
            sf.write(pred_path, audio_pred_np, CONFIG['AUDIO_SR'])
            
            # Save ground truth
            true_path = os.path.join(audio_output_dir, f'sample_{i+1}_groundtruth.wav')
            sf.write(true_path, audio_true_np, CONFIG['AUDIO_SR'])
            
            print(f"  Sample {i+1} saved")
            
            # Display audio widget
            print(f"\n  Sample {i+1} - Predicted:")
            display(Audio(audio_pred_np, rate=CONFIG['AUDIO_SR']))
            
            print(f"  Sample {i+1} - Ground Truth:")
            display(Audio(audio_true_np, rate=CONFIG['AUDIO_SR']))
    
    print(f"\n✓ Audio samples saved to {audio_output_dir}")
else:
    print("Stage A training - no audio output")

# Save final model
final_model_path = os.path.join(CONFIG['OUTPUT_DIR'], 'final_model.pt')

torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'history': history,
    'best_val_loss': best_val_loss,
    'train_stage': CONFIG['TRAIN_STAGE'],
}, final_model_path)

print(f"\n✓ Final model saved to: {final_model_path}")
print(f"\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)
print(f"Training stage: {CONFIG['TRAIN_STAGE'].upper()}")
print(f"Sample rates:")
print(f"  Powerline: {CONFIG['POWERLINE_SR']:,} Hz")
print(f"  USB:       {CONFIG['USB_SR']:,} Hz")
print(f"  Audio:     {CONFIG['AUDIO_SR']:,} Hz")
print(f"Parameters: {total_params:,}")
print(f"Best val loss: {best_val_loss:.6f}")
print(f"Device: {CONFIG['DEVICE'].upper()}")
print(f"Output: {CONFIG['OUTPUT_DIR']}")
print("=" * 80)

