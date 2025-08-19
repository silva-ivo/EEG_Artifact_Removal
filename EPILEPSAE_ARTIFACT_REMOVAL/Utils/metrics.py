import torch
import os
import numpy as np
import pandas as pd

def compute_pcc(y_true, y_pred):
    """
    Compute Pearson Correlation Coefficient (PCC).
    """
    mean_y_true = torch.mean(y_true)
    mean_y_pred = torch.mean(y_pred)

    covariance = torch.mean((y_true - mean_y_true) * (y_pred - mean_y_pred))
    
    var_y_true = torch.var(y_true, unbiased=False)  # More stable than std
    var_y_pred = torch.var(y_pred, unbiased=False)

    denominator = torch.sqrt(var_y_true * var_y_pred)
    
    # Avoid division by zero
    pcc = covariance / torch.clamp(denominator, min=1e-8)
    return pcc

def compute_snr(y_true, y_pred):
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    """
    signal_power = torch.mean(y_true ** 2)
    noise_power = torch.mean((y_true - y_pred) ** 2)
    
    # Avoid division by zero
    snr = 10 * torch.log10(torch.clamp(signal_power / noise_power, min=1e-8))
    return snr

def compute_snr_diff(y_true, y_pred, y_noisy):
    """
    Compute SNR Difference (SNR_diff).
    """
    snr_clean = compute_snr(y_true, y_noisy)  # Original noisy signal SNR
    snr_denoised = compute_snr(y_true, y_pred)  # Denoised signal SNR
    
    snr_diff = snr_denoised - snr_clean
    return snr_diff

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE) using PyTorch."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def compute_rrmse(y_true, y_pred):
    """Compute Relative Root Mean Squared Error (RRMSE) using PyTorch."""
    return compute_rmse(y_true, y_pred) / torch.sqrt(torch.mean(y_true ** 2) + 1e-8) 

def compute_metrics_summary(all_pcc, all_snr_diff, all_rmse, all_rrmse):
    def safe_mean(arr):
        return np.mean(arr) if len(arr) > 0 else float('nan')

    def safe_sem(arr):
        return np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float('nan')

    # Convert PyTorch tensors to Python scalars if needed
    if isinstance(all_pcc, torch.Tensor):
        all_pcc = all_pcc.item()
    if isinstance(all_snr_diff, torch.Tensor):
        all_snr_diff = all_snr_diff.item()
    if isinstance(all_rmse, torch.Tensor):
        all_rmse = all_rmse.item()
    if isinstance(all_rrmse, torch.Tensor):
        all_rrmse = all_rrmse.item()

    # Convert single numbers to lists
    if isinstance(all_pcc, (int, float)):
        all_pcc = [all_pcc]
    if isinstance(all_snr_diff, (int, float)):
        all_snr_diff = [all_snr_diff]
    if isinstance(all_rmse, (int, float)):
        all_rmse = [all_rmse]
    if isinstance(all_rrmse, (int, float)):
        all_rrmse = [all_rrmse]

    metrics_summary = {
        "PCC": {"mean": safe_mean(all_pcc), "sem": safe_sem(all_pcc)},
        "SNR_Diff": {"mean": safe_mean(all_snr_diff), "sem": safe_sem(all_snr_diff)},
        "RMSE": {"mean": safe_mean(all_rmse), "sem": safe_sem(all_rmse)},
        "RRMSE": {"mean": safe_mean(all_rrmse), "sem": safe_sem(all_rrmse)},
    }

    return metrics_summary


def save_metrics_to_csv(metrics_summary, model_name,loss_fucntion_name, window_size_name, mode, folder="/data/home/silva/Documents/Pipline_2/Results", filename="metrics_summary.csv"):
    dir_path = os.path.join(folder, mode, model_name,loss_fucntion_name, window_size_name)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, filename)


    df = pd.DataFrame(metrics_summary).T  # Transpose for better structure
    df.to_csv(file_path, index=True)

    print(f" Metrics saved successfully at: {file_path}")

