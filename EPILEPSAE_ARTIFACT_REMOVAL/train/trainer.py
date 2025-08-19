import torch
import torch.optim as optim
import os
import time



def train_model(model, model_id, train_loader, val_loader,
                loss_function,device, num_epochs,
                early_stopping_patience, lr):
    
    
    model_save_path = "put_your_model_save_path_here"  # Replace with your actual model save path
    os.makedirs(model_save_path, exist_ok=True)
    
    model_folder = os.path.join(model_save_path, f"{model_id}")
    os.makedirs(model_folder, exist_ok=True)
    save_path_bestmodel = os.path.join(model_folder, "best_model.pth")
    
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr)

    best_val_loss = float('inf')
    patience_counter = early_stopping_patience

    history = {
        'loss': [],
        'val_loss': [],
        'val_inference_time_ms': None  # Store only best epoch time
    }

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch, idx_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        epoch_loss = train_loss / len(train_loader)
        history['loss'].append(epoch_loss)

        # Validation phase
        y_pred_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)  
        y_true_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)
        x_inputs_tensor = torch.empty(0, *y_pred.shape[1:]).to(device)

        model.eval()
        val_loss = 0.0
        len_val_loss = 0
        total_inference_time = 0.0

        with torch.no_grad():
            for X_val, y_val, idx_batch in val_loader:
                X_val, y_val = X_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)

                # Measure inference time (GPU synced)
                torch.cuda.synchronize()
                start_time = time.time()
                y_pred = model(X_val)
                torch.cuda.synchronize()
                end_time = time.time()
                
                batch_time_ms = (end_time - start_time) * 1000
                total_inference_time += batch_time_ms

                loss = criterion(y_pred, y_val)
                if loss.item() < 1e3:
                    val_loss += loss.item()
                    len_val_loss += 1

                y_pred_tensor = torch.cat((y_pred_tensor, y_pred), dim=0)
                y_true_tensor = torch.cat((y_true_tensor, y_val), dim=0)
                x_inputs_tensor = torch.cat((x_inputs_tensor, X_val), dim=0)

        epoch_val_loss = val_loss / len_val_loss
        avg_inference_time = total_inference_time / len(val_loader)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {epoch_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Avg Inference Time: {avg_inference_time:.2f} ms")

        # Save best model and inference time
        if epoch_val_loss < best_val_loss:
            print(f"[INFO] Saving model with val_loss = {epoch_val_loss:.4f}")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path_bestmodel)
            history['val_inference_time_ms'] = avg_inference_time
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Restoring best model weights.")
            break

    # Load best model before returning
    model.load_state_dict(torch.load(save_path_bestmodel))

    return model, history, x_inputs_tensor.cpu(), y_true_tensor.cpu(), y_pred_tensor.cpu()
