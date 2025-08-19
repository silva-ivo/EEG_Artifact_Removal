import os
import numpy as np
import utils
import glob
import preprocessing
import matplotlib.pyplot as plt




data_base_dir = "DATADIRECTORY"  # Replace with your actual data directory path


patient_folders = sorted(glob.glob(os.path.join(data_base_dir, "Data_pat_*")))

if not patient_folders:
    raise FileNotFoundError(" No patient folders found in Data directory.")

for patient_folder in patient_folders:
    patient_id = os.path.basename(patient_folder)  # Extract patient name (e.g., "Data_pat_402")
    patient_output_dir = os.path.join(data_base_dir, f"Filtered_{patient_id}")

    os.makedirs(patient_output_dir, exist_ok=True)
    print(f" Processing {patient_id}...")

    original_segment = utils.load_data("original_segment_*.npy", patient_folder)
    target_segment = utils.load_data("preprocessed_segment_*.npy", patient_folder)

    print(f" {patient_id}: Original Data Shape: {original_segment.shape}")  
    print(f" {patient_id}: Target Data Shape: {target_segment.shape}")

    check_path = os.path.join(patient_output_dir, "original_filtered_segment_001.npy")

    if os.path.exists(check_path):
        print(f"Preprocessed data found for {patient_id}. Skipping preprocessing...")

    else:
        segments_reshaped = original_segment.reshape(-1, 19)
        filtered_segments = preprocessing.preprocess_eeg(segments_reshaped, fs=256)
        original_filtered_segment = filtered_segments.reshape(original_segment.shape)

        for i in range(original_filtered_segment.shape[0]):  # Loop through all segments
            segment_path = os.path.join(patient_output_dir, f"original_filtered_segment_{i+1:03d}.npy")
            np.save(segment_path, original_filtered_segment[i])

            print(f" {i} segment saved in {patient_output_dir}/")

        # ##%% PLOT ORIGINAL VS FILTERED SIGNAL
        # Select a random segment and channel to compare
        random_segment = np.random.randint(0, original_segment.shape[0])  # Pick a random EEG segment
        random_channel = np.random.randint(0, original_segment.shape[2])  # Pick a random channel

        # Extract the time series for the selected segment and channel
        original_signal = original_segment[random_segment, :, random_channel, 0]  # Shape: (153600,)
        filtered_signal = original_filtered_segment[random_segment, :, random_channel, 0]  # Shape: (153600,)

        # Create a time axis 
        fs = 256  # Sampling frequency
        time = np.arange(len(original_signal)) / fs  # Convert sample index to time in seconds

        # # Plot original vs. filtered EEG signal
        # plt.figure(figsize=(12, 6))

        # plt.plot(time, original_signal, label="Original Signal", linestyle="-", linewidth=1, alpha=0.6, color="red")
        # plt.plot(time, filtered_signal, label="Filtered Signal", linestyle="-", linewidth=1.5, alpha=0.8, color="blue")

        # plt.title(f"EEG Signal Before and After Filtering (Segment {random_segment}, Channel {random_channel})")
        # plt.xlabel("Time (seconds)")
        # plt.ylabel("Amplitude (µV)")
        # plt.legend()
        # plt.xlim(0, min(5, time[-1]))  # Show only the first 5 seconds for better visualization
        
        # plt_direct="/data/home/silva/Documents/Pipline_2/Results/Filtering"
        # os.makedirs(plt_direct, exist_ok=True)
        # plt.savefig(f"{plt_direct}/original_vs_filtered_signal_{patient_id}.png", dpi=300, bbox_inches="tight")
        # plt.close()
        
    


















        

