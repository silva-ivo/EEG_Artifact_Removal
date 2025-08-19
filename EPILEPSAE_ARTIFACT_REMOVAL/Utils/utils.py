import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import os
import numpy as np

import matplotlib.pyplot as plt


def load_data(file_pattern,base_dir):
    file_list = sorted(glob.glob(os.path.join(base_dir, file_pattern)))
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    data = [np.load(file, allow_pickle=True) for file in file_list]
    return np.array(data)  # Ensure output is a NumPy array

def split_segments(inputs, targets, window_size):
   
    num_segments, total_timepoints, num_channels, extra_dim = inputs.shape
    num_splits = total_timepoints // window_size  # Number of new windows per segment

    # Reshape inputs and targets into small windows
    split_inputs = inputs.reshape(num_segments, num_splits, window_size, num_channels, extra_dim)
    split_targets = targets.reshape(num_segments, num_splits, window_size, num_channels, extra_dim)

    # Remove the last dimension if it's redundant
    split_inputs = split_inputs.squeeze(-1)
    split_targets = split_targets.squeeze(-1)

    # Merge first two dimensions (all segments become one large batch)
    split_inputs = split_inputs.reshape(-1, window_size, num_channels)
    split_targets = split_targets.reshape(-1, window_size, num_channels)
    
    return split_inputs, split_targets

def select_channels_per_patient(X_patient, Y_patient, patient_id):
  
    # Simplified hemisphere-only dictionary(0=left, 1=right)
    channel_dict = {
        "pat_402": 1,
        "pat_8902": 0,
        "pat_16202": 0,
        "pat_23902": 0,
        "pat_30802": 1,
        "pat_32702": 0,
        "pat_46702": 1,
        "pat_50802": 1,
        "pat_53402": 1,
        "pat_55202": 1,
        "pat_56402": 0,
        "pat_58602": 0,
        "pat_59102": 1,
        "pat_60002": 0,
        "pat_64702": 1,
        "pat_75202": 1,
        "pat_80702": 0,
        "pat_85202": 0,
        "pat_93402": 0,
        "pat_93902": 1,
        "pat_112802": 0,
        "pat_113902": 1,
        "pat_114702": 1,
        "pat_114902": 0,
        "pat_123902": 0,
    }

    if patient_id not in channel_dict:
        raise ValueError(f"Patient ID {patient_id} not found in channel_dict.")

    side_flag = channel_dict[patient_id]

    if side_flag == 0:
        # Left hemisphere: F7 (10), T7 (12), P7 (14)
        Fx = X_patient[:, :, 10]
        Tx = X_patient[:, :, 12]
        Px = X_patient[:, :, 14]
        Fy= Y_patient[:, :, 10]
        Ty= Y_patient[:, :, 12]
        Py= Y_patient[:, :, 14]
    elif side_flag == 1:
        # Right hemisphere: F8 (11), T8 (13), P8 (15)
        Fx = X_patient[:, :, 11]
        Tx = X_patient[:, :, 13]
        Px = X_patient[:, :, 15]
        Fy= Y_patient[:, :, 11]
        Ty= Y_patient[:, :, 13]
        Py= Y_patient[:, :, 15]
        
    else:
        raise ValueError(f"Invalid side_flag '{side_flag}' for patient {patient_id}")

    # Create bipolar montages
    Dsq_Csqx = Fx - Tx
    Psq_Csqx = Px - Tx
    Dsq_Csqy = Fy - Ty
    Psq_Csqy = Py - Ty

    # Final shape: (segments, points, 2)
    X_selected = np.stack([Dsq_Csqx, Psq_Csqx], axis=-1)
    Y_selected = np.stack([Dsq_Csqy, Psq_Csqy], axis=-1) if Y_patient is not None else None

    print(f"{patient_id} → Side: {'Left' if side_flag == 0 else 'Right'}")
    print("X_selected shape:", X_selected.shape)
    print("Y_selected shape:", Y_selected.shape if Y_selected is not None else "None")
    return X_selected, Y_selected

def plot_random_20_segments(X_selected, Y_selected, patient_id, save_dir, fs=256):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate random indices for the segments
    random_indices = np.random.choice(X_selected.shape[0], size=20, replace=False)  # Example: selecting 20 random segments

    for idx in random_indices:
        filename = f"{patient_id}_segment_{idx}.png"
        full_path = os.path.join(save_dir, filename)

        # Calculate time in seconds for each data point in the segment
        time = np.arange(X_selected.shape[1]) / fs  # Convert time points to seconds

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"{patient_id} - Segment {idx}", fontsize=14)

        # Plot the input data (X_selected)
        axs[0].plot(time, X_selected[idx, :, 0], label="Ch:1(F - T)", linewidth=1, alpha=0.8, color='#56B4E9')
        axs[0].plot(time, X_selected[idx, :, 1], label="Ch:2(P - T)", linewidth=0.8, alpha=0.6, color='black')
        axs[0].set_ylabel("Input")
        axs[0].set_title("Input Data")
        axs[0].legend()
        axs[0].grid(True)

        # Plot the target data (Y_selected), if available
        if Y_selected is not None:
            axs[1].plot(time, Y_selected[idx, :, 0], label="Ch:1 (F - T)", linewidth=1, alpha=0.8, color='#56B4E9')
            axs[1].plot(time, Y_selected[idx, :, 1], label="Ch:2 (P - T)", linewidth=0.8, alpha=0.6, color='black')
            axs[1].set_ylabel("Target")
            axs[1].set_title("Target Data")
            axs[1].legend()
            axs[1].grid(True)

        # Set x-axis label for time in seconds
        axs[1].set_xlabel("Time (s)")

        # Set the same y-axis limits for both subplots to ensure they are on the same scale
        y_min = min(np.min(X_selected[idx, :, 0]), np.min(X_selected[idx, :, 1]), np.min(Y_selected[idx, :, 0]), np.min(Y_selected[idx, :, 1]))
        y_max = max(np.max(X_selected[idx, :, 0]), np.max(X_selected[idx, :, 1]), np.max(Y_selected[idx, :, 0]), np.max(Y_selected[idx, :, 1]))
        axs[0].set_ylim([y_min, y_max])
        axs[1].set_ylim([y_min, y_max])

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(full_path)
        plt.close()

        print(f"Saved: {full_path}")


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Define custom loss function

class RRMSELoss(nn.Module):

    def __init__(self):
        super(RRMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))

        rms_y = torch.sqrt(torch.mean(y_true ** 2))+1e-8

        rrmse = torch.divide(rmse, rms_y)

        return rrmse
    
    
def plot_predictions(y_true_cpu, y_pred_cpu, x_inputs_cpu, 
                     num_windows, save_folder):

    save_dir = os.path.join(save_folder,"PLOTS", "Predictions")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_windows):
        random_index = torch.randint(0, len(y_true_cpu), (1,)).item()
        channel_idx = torch.randint(0, y_true_cpu.shape[2], (1,)).item()

        y_true_np = y_true_cpu[random_index, :, channel_idx].cpu().numpy()
        y_pred_np = y_pred_cpu[random_index, :, channel_idx].cpu().numpy()
        x_input_np = x_inputs_cpu[random_index, :, channel_idx].cpu().numpy()

        jitter = np.linspace(-0.2, 0.2, len(y_true_np))
        time_axis = np.arange(len(y_true_np)) / 256

        # ---- First Figure: Combined Plot ----
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis,x_input_np + jitter, label='Input Values', linestyle='-', linewidth=1.2, alpha=1, color='#D55E00')  # Dark orange
        plt.plot(time_axis,y_true_np + jitter, label='True Values', linestyle='-', linewidth=1, alpha=0.8, color='#56B4E9')  # Light blue
        plt.plot(time_axis,y_pred_np + jitter, label='Predicted Values', linestyle='-', linewidth=0.8, alpha=0.6, color='black')  # Black
        plt.title(f'Predictions vs True Values (Window {random_index}, Channel {channel_idx})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (µV)')
        plt.legend()
        plt.grid(True)

       
        save_path1 = os.path.join(save_dir, f'combined_plot-{i+1}.png')
        plt.savefig(save_path1)
        plt.close()

        print(f"Plot saved at: {save_path1}")

        # # ---- Second Figure: Separate Plots ----
        # fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # # Get global min/max for consistent scaling
        # y_min = min(x_input_np.min(), y_true_np.min(), y_pred_np.min())
        # y_max = max(x_input_np.max(), y_true_np.max(), y_pred_np.max())

        # # Plot Input Values
        # axes[0].plot(x_input_np + jitter, linestyle='-', linewidth=1.2, alpha=1, color='#10b981')
        # axes[0].set_title("Input Values")
        # axes[0].grid(True)
        # axes[0].set_ylim(y_min, y_max)

        # # Plot True Values
        # axes[1].plot(y_true_np + jitter, linestyle='-', linewidth=1, alpha=0.8, color='#3b82f6')
        # axes[1].set_title("True Values")
        # axes[1].grid(True)
        # axes[1].set_ylim(y_min, y_max)

        # # Plot Predicted Values
        # axes[2].plot(y_pred_np + jitter, linestyle='-', linewidth=0.8, alpha=0.6, color='#d97706')
        # axes[2].set_title("Predicted Values")
        # axes[2].grid(True)
        # axes[2].set_ylim(y_min, y_max)

        # plt.xlabel("Time Points")
        # plt.tight_layout()

        # # Save the second figure
        # save_path2 = os.path.join(save_folder, f'separated_plots_loss-{loss_function}_window-{window_label}_channel-{channel_idx}_{i+1}.png')
        # plt.savefig(save_path2)
        # plt.close()
     

def plot_loss(history,save_folder):
    save_dir=os.path.join(save_folder,"PLOTS","LOSS")
    os.makedirs(save_dir, exist_ok=True)


    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', linestyle='-', marker='o', color='#1f77b4', markersize=6, linewidth=1.5)

    # Plot Validation Loss
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='-', marker='s', color='#ff7f0e', markersize=6, linewidth=1.5)


    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

   
    save_path = os.path.join(save_dir, 'History_loss.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Loss plot saved at: {save_path}")