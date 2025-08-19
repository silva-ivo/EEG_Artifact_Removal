from Utils import dataloader
from Utils import utils as ut
from Utils import metrics   
from models import cnn_models,DAE_models
from train import trainer
import torch
from torch import nn
import itertools
import pandas as pd
import os
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
model_config_file = "SAVEMODELCONFIGFILE"  #  model config file path for saving the model
os.makedirs(os.path.dirname(model_config_file), exist_ok=True)

# === Model parameter space ===
num_blocks_options = [5] #2,3,4
channels_options = [32,64, 128, 256, 512] #[32, 64], [32, 64, 128], [32, 64, 128, 256], ,[32, 64, 128, 256, 512, 1024]]
kernel_sizes_options = [9, 7, 5, 3, 3] #[7, 5], [7, 5, 3], [9, 7, 5, 3],[9, 7, 5, 5, 3, 3]]
use_residual_options = [True]
dropout_rates = [0.1] #[0,0.3]

# === Training parameter space ===
window_size = {"5s": 1280} #"30s": 7680, "1min": 15360,"10s": 2560 }
batch_size = {"32": 32}
loss_function = {
    "RRMSELoss": ut.RRMSELoss(),
    # "MSELoss": nn.MSELoss(),
     #"MAELoss": nn.L1Loss(),
}
lr = {"0.001": 0.001}#"0.01": 0.01,"0.0001": 0.0001

#REDUCTION FACTOR FOR THE SE LAYER
reduction={"4":4} #"16":16,"8":8

model_id=0
  
# === Iterate over all combinations of model parameters ===  
for num_blocks in num_blocks_options:
    for channels in channels_options:
        for kernels in kernel_sizes_options:
            # Filter out invalid configs
            for use_res in use_residual_options:
                for dropout in dropout_rates:
                    for reduction_key, reduction_value in reduction.items():
                        # === Save model config ===
                        model_config = {
                            "model_id": model_id,
                            "num_blocks": num_blocks,
                            "channels": str(channels),
                            "kernel_sizes": str(kernels),
                            "use_residual": use_res,
                            "dropout_rate": dropout,
                            "reduction": reduction_value
                        }
                        df_model_config = pd.DataFrame([model_config])
                        os.makedirs(os.path.join(model_config_file, f"model{model_id}"), exist_ok=True)
                        model_config_file_id = os.path.join(model_config_file,f"model{model_id}", f"model{model_id}_config.csv")
                        print(f"Saving model config to {model_config_file_id}")
                        df_model_config.to_csv(model_config_file_id, mode='a', index=False,
                                                header=not os.path.exists(model_config_file_id))
                    
                    
                        
                        # === For each valid training config ===
                        for win_size_key in window_size.keys():
                            window_size_value = window_size[win_size_key]
                            all_inputs,all_targets,all_pat_ids = dataloader.load_all_patients(data_base_dir, window_size_value)
                            for lr_key in lr.keys():
                                batch_size_value = batch_size["32"]
                                loss_function_value = loss_function["RRMSELoss"]
                                lr_value = lr[lr_key]

                                print(f"Model {model_id}:")
                                print(f"  Blocks: {num_blocks}, Channels: {channels}, Kernels: {kernels}")
                                print(f"  Residual: {use_res}, Dropout: {dropout}")
                                print(f"  Window: {win_size_key}, Loss: RRMSELoss, LR: {lr_key}")
                                print(f"  Reduction: {reduction_value}")

                                
                                
                                
                                # === Get dataloaders ===
                                all_pcc,all_snr_diff,all_rmse,all_rrmse,all_cpt = [],[],[],[],[]
                                inner_fold=0
                                
                                for train_loader,val_loader in dataloader.get_nested_cv_loaders(all_inputs,all_targets):
                                    print(f"Inner trainning loop:{inner_fold}")
                                    # === Model ===
                                    model = cnn_models.SE_ResNet1D(
                                        input_channels=2,
                                        num_blocks=num_blocks,
                                        channels=channels,
                                        kernel_sizes=kernels,
                                        reduction=reduction_value,
                                        use_residual=use_res,
                                        dropout_rate=dropout
                                    )
                                    model_name = f"model{model_id}"
                                    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

                                    # === Train ===
                                    model, history, x_input, y_true, y_pred = trainer.train_model(
                                        model, model_name, train_loader, val_loader, win_size_key,
                                        loss_function_value,"RRMSELoss", device, num_epochs=400,
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
                                    
                                    # === Save results ===
                                    result = {
                                        'model_id': model_id,
                                        "pcc": pcc,
                                        "snr_diff": snr_diff,
                                        "rmse": rmse,
                                        "rrmse": rrmse,
                                        "cp_time": cp_time,}
                                    
                                    
                                    df_result = pd.DataFrame([result])
                                    model_results_file_id = os.path.join(model_config_file,f"model{model_id}_results.csv")
                                    df_result.to_csv(model_results_file_id, mode='a', index=False,
                                                    header=not os.path.exists(model_results_file_id))

                                        # # === Plots (optional) ===
                                    ut.plot_loss(history,model_config_file)
                                    ut.plot_predictions(y_true, y_pred, x_input, 20, model_config_file)
                                    inner_fold += 1
                                    model_id += 1
                                    
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
                                "model_id": model_id,
                                "window_size": win_size_key,
                                "batch_size": "32",
                                "loss_function": "RRMSELoss",
                                "learning_rate": lr_key,
                                "pcc": f"{pcc_avg:.4f} ± {pcc_std:.4f}",
                                "snr_diff": f"{snr_diff_avg:.4f} ± {snr_diff_std:.4f}",
                                "rmse": f"{rmse_avg:.4f} ± {rmse_std:.4f}",
                                "rrmse": f"{rrmse_avg:.4f} ± {rrmse_std:.4f}",
                                "cpt(ms)": f"{cpt_avg:.4f} ± {cpt_std:.4f}",}

                                
                                model_config_file=os.makedirs(os.path.join(model_config_file, f"model{model_id}"), exist_ok=True)
                                df_result = pd.DataFrame([result])
                                model_results_file_id = os.path.join(model_config_file,f"model{model_id}", f"model{model_id}_results.csv")
                                df_result.to_csv(model_results_file_id, mode='a', index=False,
                                                header=not os.path.exists(model_results_file_id))

                                    # # === Plots (optional) ===
                                ut.plot_loss(history,model_config_file)
                                ut.plot_predictions(y_true, y_pred, x_input, 15, model_config_file)

                        #model_id += 1
                        
                        
