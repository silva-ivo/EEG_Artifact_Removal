from Utils import dataloader
from Utils import utils as ut
from Utils import metrics   
from models import DAE_models
from train import trainer
import torch
from torch import nn
import itertools
import pandas as pd
import os
import json
import numpy as np




#LOADERS
data_base_dir = "DATADIRECTORY"  # Replace with your actual data directory path
# window_size = 15360
# window_size_name = "5s"
# batch_size = 64
#window_size = 15360 1minuto
#windows_size = 7680 30segundos
#windows_size = 2560 10segundos
#window_szie = 1280 5segundos
#window_size = 256 1segundos


# === Paths ===
model_config_file = "SAVECONFIGFILE"  # Replace with your actual model configuration file path
os.makedirs(model_config_file, exist_ok=True)

# === Model parameter space ===
dropout_rates = [0.1] #0.3 ,0

# === Training parameter space ===
window_size = {"5s": 1280} #"10s": 2560, "1min": 15360, "30s": 7680
batch_size = {"32": 32}
loss_function = {
    "RRMSELoss": ut.RRMSELoss(),
    #"MSELoss": nn.MSELoss(),
    #"MAELoss": nn.L1Loss(),
}
lr = {"0.001": 0.001} # "0.01": 0.01 "0.0001": 0.0001

# === Available models ===
models_dict = {
    #"SE_UNet_3": DAE_models.SE_UNet_3,
    #"SE_UNet_4": DAE_models.SE_UNet_4,
    #"SE_UNet_5": DAE_models.SE_UNet_5,
    #"SE_UNet_6": DAE_models.SE_UNet_6,
    "SE_UNet_7": DAE_models.SE_UNet_7,
}
reduction = {"16":16 }

for model_name, model_class in models_dict.items():
    for loss_key, loss_value in loss_function.items():
        for win_size_key in window_size.keys():
            window_size_value = window_size[win_size_key]
            all_inputs,all_targets,all_pat_ids = dataloader.load_all_patients(data_base_dir, window_size_value)
            for lr_key in lr.keys():
                for reduction_key, reduction_value in reduction.items():
                    batch_size_value = batch_size["32"]
                    lr_value = lr[lr_key]

                    print(f"Model {model_name}:")

                    print(f"  Window: {win_size_key}, Loss:{loss_key}, LR: {lr_key}")
                    all_pcc,all_snr_diff,all_rmse,all_rrmse,all_cpt = [],[],[],[],[]
                    inner_fold=0
                    # === Get dataloaders ===
                    for train_loader,val_loader in dataloader.get_nested_cv_loaders(all_inputs,all_targets):
                        print(f"Inner trainning loop:{inner_fold}")
                        # === Model ===
                        model = model_class(reduction=reduction_value,dropout=0.1)
                
                        
                        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

                        # === Train ===
                        model, history, x_input, y_true, y_pred = trainer.train_model(
                            model, model_name, train_loader, val_loader, win_size_key,
                            loss_value,loss_key, device, num_epochs=400,
                            early_stopping_patience=20, lr=lr_value
                        )

                        # === Metrics ===
                        pcc = metrics.compute_pcc(y_true, y_pred)
                        snr_diff = metrics.compute_snr_diff(y_true, y_pred, x_input)
                        rmse = metrics.compute_rmse(y_true, y_pred)
                        rrmse = metrics.compute_rrmse(y_true, y_pred)
                        cp_time=history['val_inference_time_ms']
                        # Convert the tensors to scalar values using .item() before appending
                        all_pcc.append(pcc.item() if isinstance(pcc, torch.Tensor) else pcc)
                        all_snr_diff.append(snr_diff.item() if isinstance(snr_diff, torch.Tensor) else snr_diff)
                        all_rmse.append(rmse.item() if isinstance(rmse, torch.Tensor) else rmse)
                        all_rrmse.append(rrmse.item() if isinstance(rrmse, torch.Tensor) else rrmse)
                        all_cpt.append(cp_time)
                        inner_fold += 1
                        
                    pcc_avg = np.mean(all_pcc)
                    pcc_std = np.std(all_pcc)
                    
                    snr_diff_avg = np.mean(all_snr_diff)
                    snr_diff_std = np.std(all_snr_diff)
                    
                    rmse_avg = np.mean(all_rmse)
                    rmse_std = np.std(all_rmse)
                    
                    rrmse_avg = np.mean(all_rrmse)
                    rrmse_std = np.std(all_rrmse)
                                            
                    cpt_avg = np.mean(all_cpt)
                    cpt_std = np.std(all_cpt)
                    
                    # === Save training result ===
                    result = {
                    "window_size": win_size_key,
                    "batch_size": "32",
                    "loss_function": loss_key,
                    "learning_rate": lr_key,
                    "pcc": f"{pcc_avg:.4f} ± {pcc_std:.4f}",
                    "snr_diff": f"{snr_diff_avg:.4f} ± {snr_diff_std:.4f}",
                    "rmse": f"{rmse_avg:.4f} ± {rmse_std:.4f}",
                    "rrmse": f"{rrmse_avg:.4f} ± {rrmse_std:.4f}",
                    "cpt(ms)": f"{cpt_avg:.4f} ± {cpt_std:.4f}",}

                    
                    os.makedirs(os.path.join(model_config_file, f"model{model_name}"), exist_ok=True)
                    df_result = pd.DataFrame([result])
                    model_results_file_id = os.path.join(model_config_file,f"model{model_name}", f"model{model_name}_results.csv")
                    df_result.to_csv(model_results_file_id, mode='a', index=False,
                                    header=not os.path.exists(model_results_file_id))

                        # # === Plots (optional) ===
                    ut.plot_loss(history,model_config_file, win_size_key,reduction_key,model_name,loss_key)
                    ut.plot_predictions(y_true, y_pred, x_input, 15, model_config_file,win_size_key,reduction_key,model_name,loss_key)

