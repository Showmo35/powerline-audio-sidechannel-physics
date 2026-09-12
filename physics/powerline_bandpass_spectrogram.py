import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from IPython.display import Audio
import os
from scipy import signal
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt, sosfiltfilt, freqz, lfilter, spectrogram
try:
    import sounddevice as sd
except Exception as e:
    sd = None
    print(f"⚠ sounddevice import failed: {e}. Audio playback disabled.\n  To enable audio: install the PortAudio library (system) and reinstall/compile python-sounddevice, or use conda: `conda install -c conda-forge portaudio python-sounddevice`.")
import pandas as pd
from scipy.interpolate import interp1d, interp2d, griddata, RegularGridInterpolator
import sys

def moving_average(signal, window_size):
    """Applies a moving average filter to a 1D signal."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def alternate_merge_loop(arr1, arr2):
    merged_arr = []
    min_len = min(len(arr1), len(arr2))

    for i in range(min_len):
        merged_arr.append(arr1[i])
        merged_arr.append(arr2[i])

    # Append remaining elements from the longer array
    if len(arr1) > min_len:
        merged_arr.extend(arr1[min_len:])
    elif len(arr2) > min_len:
        merged_arr.extend(arr2[min_len:])

    return merged_arr

def fm_demodulate_quadrature(iq_signal, fs):
            # Compute the product of current sample and conjugate of previous sample
            product = iq_signal[1:] * np.conj(iq_signal[:-1])
            # The angle of this product gives the instantaneous phase difference
            demodulated_signal = np.angle(product) * (fs / (2 * np.pi))
            return demodulated_signal

def find_row_extremes(arr_2d):
    """
    Finds the maximum and minimum entry within each row of a 2D array.

    Args:
        arr_2d: A list of lists representing the 2D array.

    Returns:
        A tuple containing two lists:
        - max_row_entries: A list where each element is the maximum value of the corresponding row.
        - min_row_entries: A list where each element is the minimum value of the corresponding row.
    """
    max_row_entries = []
    min_row_entries = []

    for row in arr_2d:
        if row:  # Check if the row is not empty
            max_row_entries.append(max(row))
            min_row_entries.append(min(row))
        else:
            # Handle empty rows if necessary, e.g., append None or a specific value
            max_row_entries.append(None)
            min_row_entries.append(None)
            
    return max_row_entries, min_row_entries

def interpolate_spectrogram_manual(Sxx_orig, times_orig, freqs_orig, times_new, freqs_new):
    """
    Performs 2D linear interpolation on a spectrogram using sequential 1D numpy.interp.

    Args:
        Sxx_orig (np.ndarray): Original spectrogram data (shape: [n_freqs, n_times]).
        times_orig (np.ndarray): Original time coordinates (1D).
        freqs_orig (np.ndarray): Original frequency coordinates (1D).
        times_new (np.ndarray): New time coordinates for interpolation (1D).
        freqs_new (np.ndarray): New frequency coordinates for interpolation (1D).

    Returns:
        np.ndarray: Interpolated spectrogram data (shape: [n_freqs_new, n_times_new]).
    """
    # Ensure original coordinates are monotonically increasing as required by np.interp
    if not np.all(np.diff(times_orig) > 0):
        # Sort if necessary, or add error handling
        pass 
    if not np.all(np.diff(freqs_orig) > 0):
        # Sort if necessary, or add error handling
        pass

    # 1. Interpolate along the time axis (axis 1) for all frequencies
    # np.interp expects 1D inputs. The process is applied row-wise (for each frequency).
    # Since np.interp only works on 1D arrays, we might need a loop or array manipulation.
    # A more efficient way to apply 1D interpolation over an axis of a 2D array:

    # Transpose to put the axis to be interpolated first (optional, but helps conceptually)
    Sxx_T = Sxx_orig.T
    Sxx_interp_time_T = np.zeros((len(times_new), Sxx_T.shape[1]))
    for i in range(Sxx_T.shape[1]):
        Sxx_interp_time_T[:, i] = np.interp(times_new, times_orig, Sxx_T[:, i])
    Sxx_interp_time = Sxx_interp_time_T.T # Transpose back

    # 2. Interpolate along the frequency axis (axis 0) for all new times
    Sxx_interp_2d = np.zeros((len(freqs_new), len(times_new)))
    for j in range(Sxx_interp_time.shape[1]):
        Sxx_interp_2d[:, j] = np.interp(freqs_new, freqs_orig, Sxx_interp_time[:, j])

    return Sxx_interp_2d

# === FILE PATHS ===
#folder = r"/content/Jan18_2025_downsampled"
folder = r"May29_Alice"
fsamp = 200000;
usb_path = os.path.join(folder, "Chap_1_img.bin")      # USB (output)
powerline_path = os.path.join(folder, "Chap_1_real.bin")  # Powerline (input)

# === LOAD SIGNALS ===
usb_data_orig = np.fromfile(usb_path, dtype=np.float32)
powerline_data_orig = np.fromfile(powerline_path, dtype=np.float32)

prenotch_usb_data = usb_data_orig[:2000000];
prenotch_powerline_data = powerline_data_orig[:2000000];

#Cut off frequencies
lowcut = 20000; #For USB
highcut = 25000;

lowcut_powerline = 18000; #For Powerline
highcut_powerline = 24000;

# Play the first few secs of the USB audio for sanity check
#print("Playing sample original audio")
#sd.play(np.abs(usb_data_orig[:2000000]),fsamp)
#sd.wait()
#print("Done playing sample audio")

# === PLOT SAMPLE SIGNALS ===
plt.figure(figsize=(12, 5))
plt.plot(prenotch_powerline_data[:2000000], label='Powerline (Input)', color='orange')
plt.plot(prenotch_usb_data[:2000000], label='USB (Output)', alpha=0.7, color='blue')
#plt.plot(am_demod_powerline[:2000000], label="Powerline Envelope", color='red')
plt.title("Powerline Input vs USB Output - First 20000 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('SamplesOverTime.png')
plt.cla()
#plt.show()

#Notch 60Hz powerline signal
f0 = 60; #Freq to be notched
w0 = f0 / (fsamp / 2) # Normalized Notch Freq
Q = 100; # Quality of the notch (higher the better)
b, a = signal.iirnotch(w0, Q)
powerline_data = signal.filtfilt(b, a, prenotch_powerline_data)
usb_data = signal.filtfilt(b, a, prenotch_usb_data)


# Compute the Power Spectral Density using Welch's method

frequencies_usb, psd_usb = signal.welch(usb_data, fsamp, nperseg=32768)
frequencies_powerline, psd_powerline = signal.welch(powerline_data, fsamp, nperseg=32768)

# Plot the PSD
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies_usb, psd_usb, label="USB", color='blue')
plt.semilogy(frequencies_powerline, psd_powerline, label="Powerline", color='orange')
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (unit²/Hz)')
plt.grid(True)
plt.savefig('PowerSpectralDensity.png')
plt.cla()
#plt.show()

#Filter using spectrogram

# === Spectrogram (Freq vs Time) Analysis === #
frequencies, times, Sxx = spectrogram(prenotch_powerline_data, fsamp, nperseg=2048, noverlap=512);
#frequencies, times, Sxx = spectrogram(prenotch_powerline_data, fsamp, nperseg=32768, noverlap=512);

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Power/Frequency (dB/Hz)')
#plt.ylim([lowcut_powerline-1000, highcut_powerline+1000]) # Limit y-axis to Nyquist frequency
plt.savefig('FreqVsTimePlot_Original.png')
plt.cla()

min_freq = frequencies[0];
max_freq = frequencies[0];
min_id = frequencies[0];
max_id = frequencies[0];
count = 0;
for element in frequencies:
    if element <= 18500:
        min_freq = element;
        min_id = count;
    if element <= 23000:
        max_freq = element;
        max_id = count;
    count = count + 1;
print(min_freq, max_freq, min_id, max_id, count, len(times))

freq_bandpass = frequencies[min_id:max_id]
Sxx_bandpass = Sxx[min_id:max_id,:]
print("size of Sxx_bandpass", Sxx_bandpass.shape)

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, freq_bandpass, 10 * np.log10(Sxx_bandpass), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Power/Frequency (dB/Hz)')
#plt.ylim([lowcut_powerline-1000, highcut_powerline+1000]) # Limit y-axis to Nyquist frequency
plt.savefig('FreqVsTimePlot_Bandpass.png')
plt.cla()

#Nulling below a threshold
indices_below_threshold = np.argwhere(10*np.log10(Sxx_bandpass) < -128)
for row, col in indices_below_threshold:
    Sxx_bandpass[row][col] = 0

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, freq_bandpass, 10 * np.log10(Sxx_bandpass), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Power/Frequency (dB/Hz)')
#plt.ylim([lowcut_powerline-1000, highcut_powerline+1000]) # Limit y-axis to Nyquist frequency
plt.savefig('FreqVsTimePlot_Bandpass_Nulling.png')
plt.cla()

#print(min(times), max(times), min(frequencies), max(frequencies))

#Interpolate Sxx_bandpass

times_new = np.linspace(min(times), max(times), 1000) # e.g., 15 new time points
freqs_new = np.linspace(min(freq_bandpass), max(freq_bandpass), 10000) # e.g., 10 new frequency points
Sxx_inter = interpolate_spectrogram_manual(Sxx_bandpass, times, freq_bandpass, times_new, freqs_new)

#print("Size of grids and Sxx_inter", len(grid_x), len(grid_y), Sxx_inter.shape)


plt.figure(figsize=(10, 6))
plt.pcolormesh(times_new, freqs_new, 10 * np.log10(Sxx_inter), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Power/Frequency (dB/Hz)')
#plt.ylim([lowcut_powerline-1000, highcut_powerline+1000]) # Limit y-axis to Nyquist frequency
plt.savefig('FreqVsTimePlot_Bandpass_Inter.png')
plt.cla()

df = pd.DataFrame(Sxx_inter);

max_over_time = [];
min_over_time = [];
for col in df.columns:
    min_id = max(freqs_new);
    max_id = 0;
    iden = 0;
    for elem in df[col]:
        if elem != 0:
            if min_id > iden:
                min_id = iden;
            if max_id < iden:
                max_id = iden;
        iden = iden + 1;
    if (min_id == max(freqs_new)):
        continue
    max_over_time.append((max_id-200)/8000);
    min_over_time.append((min_id-200)/8000)
            
#Dilate the signal over time to match the original audio
original_x = np.linspace(0, 1, len(max_over_time));
#new_x = np.linspace(0, 1, 2000000);
new_x = np.linspace(0, 1, 1000000);
f = interp1d(original_x, max_over_time, kind='linear');
interpolate_max_over_time = f(new_x)

original_x1 = np.linspace(0, 1, len(min_over_time));
new_x1 = np.linspace(0, 1, 1000000);
f = interp1d(original_x1, min_over_time, kind='linear');
interpolate_min_over_time = f(new_x1)

new_interpolated_over_time = alternate_merge_loop(interpolate_max_over_time, interpolate_min_over_time);

# === PLOT INTERPOLATED MAX VALUES ===

plt.figure(figsize=(10, 6))
#plt.plot(interpolate_max_over_time[:2000000]/interpolate_max_over_time.max(), label='Interpolated', color='orange')
plt.plot(new_interpolated_over_time[:2000000], label='Interpolated', color='orange')#plt.plot(filtered_usb_signal[:2000000], label='USB (Output)', alpha=0.7, color='blue')
plt.title("Interpolated")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Interpolated.png')
plt.cla()

# === PLAYING INTERPOLATED SIGNALS ON SPEAKER (TAKES ONLY ABSOLUTE VALUES) ===
print("Playing Interpolated Audio Now of length", len(interpolate_max_over_time))

# Attempt playback if sounddevice (and PortAudio) is available; otherwise save WAV fallback
if sd is not None:
    try:
        sd.play(np.abs(new_interpolated_over_time) / (np.abs(new_interpolated_over_time).max() + 1e-12), fsamp)
        sd.wait()
    except Exception as e:
        print(f"⚠ Error during playback: {e}")
        sd = None

if sd is None:
    # Fallback: write a WAV file the user can play manually
    out_wav = 'interpolated_playback.wav'
    try:
        # Normalize to [-0.9, 0.9] and save as float32
        norm = np.abs(new_interpolated_over_time).max()
        if norm > 0:
            scaled = (new_interpolated_over_time / norm * 0.9).astype(np.float32)
        else:
            scaled = new_interpolated_over_time.astype(np.float32)
        write(out_wav, fsamp, scaled)
        print(f"Saved interpolated audio to {out_wav}. Use an external player (e.g. aplay, ffplay) to listen.")
    except Exception as e:
        print(f"Could not save WAV fallback: {e}")
