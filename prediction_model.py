import numpy as np            # Import the NumPy library for numerical operations
import pandas as pd           # Import the pandas library for data manipulation
import torch                  # Import the PyTorch library
import torch.nn as nn         # Import PyTorch's neural network module
import os                     # Import the OS library for operating system related tasks
import joblib                 # Import joblib for saving/loading Python objects
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for creating and loading datasets
from sklearn.preprocessing import MinMaxScaler     # Import MinMaxScaler for feature scaling

# For regression metrics (to evaluate model performance)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# ----------------------
# 1. Load Data
# ----------------------
df = pd.read_csv("./data/binance_btcusdt_1HOUR.csv")
df = df.iloc[200:].reset_index(drop=True)

# ----------------------
# 2. Select columns
# ----------------------
features = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    # MACD
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    # RSI
    "RSI_30",
    # Hull MAs
    "HMA_20",
    "HMA_50",
    "HMA_200",
    # Bollinger (length=50)
    "BBL_50_2.0",
    "BBM_50_2.0",
    "BBU_50_2.0",
    "BBB_50_2.0",
    "BBP_50_2.0",
    # Bollinger (length=200)
    "BBL_200_2.0",
    "BBM_200_2.0",
    "BBU_200_2.0",
    "BBB_200_2.0",
    "BBP_200_2.0",
    # OBV
    "OBV"
]
data = df[features].values
if np.isnan(data).any():
    print("Data contains NaN values. Handling NaNs...")
    data = np.nan_to_num(data, nan=0.0)

# ----------------------
# 3. Scale data
# ----------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X_all = data_scaled
close_idx = features.index("close")
y_all = data_scaled[:, close_idx]

# ----------------------
# create_sliding_windows
# ----------------------
def create_sliding_windows(X, y, lookback=30):
    X_slid, y_slid = [], []
    for i in range(len(X) - lookback):
        X_slid.append(X[i : i + lookback])
        y_slid.append(y[i + lookback])
    return np.array(X_slid), np.array(y_slid)

lookback = 24
X_slid, y_slid = create_sliding_windows(X_all, y_all, lookback=lookback)

# ----------------------
# 5. Train/Val/Test Split
# ----------------------
train_size = int(0.7 * len(X_slid))
val_size = int(0.15 * len(X_slid))
test_size = len(X_slid) - train_size - val_size

X_train = X_slid[:train_size]
y_train = y_slid[:train_size]

X_val = X_slid[train_size : train_size + val_size]
y_val = y_slid[train_size : train_size + val_size]

X_test = X_slid[train_size + val_size :]
y_test = y_slid[train_size + val_size :]

# ----------------------
# TimeSeriesDataset
# ----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------
# 7. LSTM Model Definition
# ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

# ----------------------
# 8. Train Function with Adaptive LR and Early Stopping
# ----------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=50,
    device="cpu",
    early_stopping_patience=10
):
    """
    Trains the model for a given number of epochs, uses a scheduler for adaptive LR,
    and implements early stopping based on validation loss.
    """
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, device=device)
        
        # Step the scheduler with the validation loss for ReduceLROnPlateau
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {np.mean(train_losses):.6f} "
              f"Val Loss: {val_loss:.6f}")
        
        # Early Stopping check
        if val_loss < best_val_loss - 1e-7:  # A tiny delta to check for improvement
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    # Load the best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

# ----------------------
# 9. Evaluate Function (MSE)
# ----------------------
def evaluate(model, data_loader, criterion, device="cpu"):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            losses.append(loss.item())
    
    return np.mean(losses)

# ----------------------
# Additional function: get_predictions
# ----------------------
def get_predictions(model, data_loader, device="cpu"):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    return np.array(all_targets), np.array(all_preds)

# ----------------------
# 10. Main Execution
# ----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate our LSTM model
    model = LSTMModel(input_dim=27, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    
    criterion = nn.MSELoss()
    
    # Adam optimizer with a small initial LR
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 1) Define the scheduler for Adaptive Learning Rate
    #    - We use ReduceLROnPlateau to reduce the LR when val_loss stops improving.
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,          # Scale the learning rate by 0.1
        patience=5,          # Wait 5 epochs with no improvement before reducing LR
        verbose=True,
        min_lr=1e-7          # Don't go below this learning rate
    )
    
    # 2) Train the model with early stopping and LR scheduler
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=100,
        device=device,
        early_stopping_patience=10   # Stop if no improvement for 10 epochs
    )
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, device=device)
    print("Test Loss (MSE):", test_loss)
    
    # # Save model
    # save_dir = "saved_models"
    # os.makedirs(save_dir, exist_ok=True)
    # model_path = os.path.join(save_dir, "lstm_model.pth")
    # torch.save(model.state_dict(), model_path)
    # print(f"Model weights saved to {model_path}")

    # # Save the scaler
    # scaler_path = os.path.join(save_dir, "scaler.joblib")
    # joblib.dump(scaler, scaler_path)
    # print(f"Scaler saved to {scaler_path}")

    # Compute additional metrics
    y_true, y_pred = get_predictions(model, test_loader, device=device)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nAdditional Test Metrics:")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    
    # Save metrics to CSV
    metrics_dict = {
        "mse": [mse],
        "rmse": [rmse],
        "mae": [mae],
        "r2": [r2]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv("test_metrics.csv", index=False)
    print("\nSaved test_metrics.csv")
    
    # Save predictions vs. targets to CSV
    df_results = pd.DataFrame({
        "y_true_scaled": y_true,
        "y_pred_scaled": y_pred
    })
    df_results.to_csv("predictions_scaled.csv", index=False)
    print("Saved predictions_scaled.csv")
    
    # Optionally invert-scale predictions to get real prices
    all_unscaled_preds = []
    all_unscaled_true = []
    
    for true_val_scaled, pred_val_scaled in zip(y_true, y_pred):
        dummy_row_true = np.zeros((1, len(features)))
        dummy_row_true[0, close_idx] = true_val_scaled
        
        dummy_row_pred = np.zeros((1, len(features)))
        dummy_row_pred[0, close_idx] = pred_val_scaled
        
        unscaled_true_val = scaler.inverse_transform(dummy_row_true)[0, close_idx]
        unscaled_pred_val = scaler.inverse_transform(dummy_row_pred)[0, close_idx]
        
        all_unscaled_true.append(unscaled_true_val)
        all_unscaled_preds.append(unscaled_pred_val)
    
    df_unscaled = pd.DataFrame({
        "y_true_unscaled": all_unscaled_true,
        "y_pred_unscaled": all_unscaled_preds
    })
    df_unscaled.to_csv("predictions_unscaled.csv", index=False)
    print("Saved predictions_unscaled.csv")

    print("\nDone!")
