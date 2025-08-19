import mne
import numpy as np
import matplotlib.pyplot as plt

data_base_dir = "/data/datasets/20240201_UNEEG_ForMayo/B52K3P3G/V5a/B52K3P3G_01_0004245_20211008_01_EEGdata.edf"

# preload=True loads the data into memory
raw = mne.io.read_raw_edf(data_base_dir, preload=True, verbose=True)

# Display basic info
#print(raw.info)

# Plot (optional, requires a GUI)


# Access EEG data as numpy
data, times = raw.get_data(return_times=True)

print("Shape of EEG data:", data.shape)  # (n_channels, n_samples)


# Select one EEG channel (0 or 1 for your 2-channel data)
channel_idx = 0  # change to 1 if you want the second channel
channel_name = raw.info['ch_names'][channel_idx]

# Sampling frequency and window size
sfreq = raw.info['sfreq']
window_duration_sec = 5
samples_per_window = int(sfreq * window_duration_sec)

# Get EEG data and times
data, times = raw.get_data(return_times=True)
n_samples = data.shape[1]

# Number of samples you can start at to get a full window
max_start = n_samples - samples_per_window

# Randomly select 10 start points
np.random.seed(42)
random_starts = np.random.randint(0, max_start, size=10)

# Plot each window
for i, start in enumerate(random_starts):
    end = start + samples_per_window
    window_data = data[channel_idx, start:end]
    window_times = times[start:end]

    plt.figure(figsize=(10, 3))
    plt.plot(window_times, window_data, color='tab:blue')
    plt.title(f'{channel_name} - Random 5s Window {i+1} (from {window_times[0]:.2f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()