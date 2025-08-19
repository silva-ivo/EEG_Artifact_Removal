import numpy as np
import scipy.signal as signal


def butter_bandpass(lowcut, highcut, fs=256, order=4):
    """Design a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data, notch_freq=50, fs=256, order=2):
    """Apply a notch filter to remove powerline interference."""
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = signal.iirnotch(freq, Q=30)  # Q=30 gives a narrow notch
    return signal.filtfilt(b, a, data, axis=0)

def apply_bandpass_filter(data, lowcut=0.5, highcut=100, fs=256, order=4):
    """Apply a Butterworth bandpass filter to EEG data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return signal.filtfilt(b, a, data, axis=0)  # Zero-phase filtering

def remove_artifacts(data, threshold=600, flatline_duration=1, fs=256):
    """
    Remove EEG artifacts:
    - Saturated values (e.g., extreme peaks)
    - Flatlines (constant signal for a duration)
    - Abnormal peaks (values beyond a threshold)
    """
    data = np.copy(data)

    # Remove flatlines: If signal is constant for `flatline_duration` seconds, set to NaN
    flatline_samples = int(flatline_duration * fs)
    for i in range(data.shape[1]):  # Iterate over EEG channels
        if np.all(data[:flatline_samples, i] == data[0, i]):
            data[:, i] = np.nan  # Mark for removal

    # Remove extreme peaks (e.g., values above threshold)
    data[np.abs(data) > threshold] = np.nan

    # Interpolate NaN values
    for i in range(data.shape[1]):
        nan_indices = np.isnan(data[:, i])
        if np.any(nan_indices):
            data[:, i] = np.interp(
                np.arange(len(data[:, i])),
                np.where(~nan_indices)[0],
                data[~nan_indices, i]
            )

    return data

def preprocess_eeg(data, fs=256):
    """Apply bandpass filter, notch filter, and artifact removal to EEG data."""
    data = apply_bandpass_filter(data, fs=fs)  # Remove low/high frequency noise
    data = notch_filter(data, fs=fs)  # Remove 50Hz interference
    data = remove_artifacts(data, fs=fs)  # Remove unwanted artifacts
    return data