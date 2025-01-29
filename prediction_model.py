import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------
# 1. Load and Prepare Data
# ---------------------------
df = pd.read_csv("./data/binance_btcusdt_15MIN.csv")

# If you have not yet created these columns in your CSV, uncomment below:
"""
if "pct_change_current" not in df.columns:
    df["pct_change_current"] = df["close"].pct_change() * 100

if "pct_change_future" not in df.columns:
    df["pct_change_future"] = ((df["close"].shift(-1) - df["close"]) / df["close"]) * 100
    df.dropna(subset=["pct_change_future"], inplace=True)
    df.reset_index(drop=True, inplace=True)
"""

df = df.iloc[300:].reset_index(drop=True)  # optional skip of early data

# ---------------------------
# 2. Select Features & Label
# ---------------------------
# "pct_change_current" is now an input feature
# "pct_change_future" is our label
features = [
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
    # MACD
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    # RSI
    "RSI_30",
    # Hull MAs
    "HMA_20", "HMA_50", "HMA_200",
    # Bollinger (length=50)
    "BBL_50_2.0", "BBM_50_2.0", "BBU_50_2.0", "BBB_50_2.0", "BBP_50_2.0",
    # Bollinger (length=200)
    "BBL_200_2.0", "BBM_200_2.0", "BBU_200_2.0", "BBB_200_2.0", "BBP_200_2.0",
    # OBV
    "OBV",
    # NEW Feature
    "pct_change_current"
]

# Create the NumPy array of features
X_all = df[features].values

# Our target (regression) is the future price percentage change
y_all = df["pct_change_future"].values

# ---------------------------
# 3. Handle NaNs & Scaling
# ---------------------------
if np.isnan(X_all).any():
    print("Feature data contains NaN values. Replacing them with 0.0...")
    X_all = np.nan_to_num(X_all, nan=0.0)

if np.isnan(y_all).any():
    print("Label data contains NaN values. Dropping or replacing them with 0.0...")
    # Here we simply remove any row with a NaN label
    valid_mask = ~np.isnan(y_all)
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]

# Scale the features (X) using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)

# ---------------------------
# 4. Create Sliding Windows
# ---------------------------
def create_sliding_windows(X, y, lookback=24):
    X_slid, y_slid = [], []
    for i in range(len(X) - lookback):
        X_slid.append(X[i : i + lookback])
        y_slid.append(y[i + lookback])
    return np.array(X_slid), np.array(y_slid)

# Increase the lookback as desired
lookback = 96
X_slid, y_slid = create_sliding_windows(X_scaled, y_all, lookback)

# ---------------------------
# 5. Train/Val/Test Split
# ---------------------------
train_size = int(0.7 * len(X_slid))
val_size   = int(0.15 * len(X_slid))
test_size  = len(X_slid) - train_size - val_size

X_train = X_slid[:train_size]
y_train = y_slid[:train_size]

X_val = X_slid[train_size : train_size + val_size]
y_val = y_slid[train_size : train_size + val_size]

X_test = X_slid[train_size + val_size :]
y_test = y_slid[train_size + val_size :]

print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

# ---------------------------
# 6. Dataset Definition
# ---------------------------
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
val_dataset   = TimeSeriesDataset(X_val, y_val)
test_dataset  = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------
# 7. LSTM Model for Regression
# ---------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # input_dim = number of features
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # A single linear layer to go from hidden_dim -> 1
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch_size, lookback, input_dim)
        batch_size = x.size(0)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out: (batch_size, lookback, hidden_dim)

        # Take the last time step's output
        out_last = out[:, -1, :]  # (batch_size, hidden_dim)

        # Pass it through a fully connected layer
        out = self.fc(out_last)   # (batch_size, 1)
        return out

# ---------------------------
# 8. Loss & Optimizer
#    For regression, we typically use MSELoss
# ---------------------------
criterion = nn.MSELoss()
lr = 1e-3  # Adjust if needed

# Instantiate the LSTM model
# input_dim = number of features
input_dim = X_train.shape[2]  # i.e. 27 + 1 new feature, etc.
model = SimpleLSTM(input_dim=input_dim, hidden_dim=64, num_layers=1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------------------
# 9. Training & Evaluation
# ---------------------------
def evaluate(model, data_loader, criterion, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # shape: (batch_size, 1)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            losses.append(loss.item())
    return np.mean(losses)

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device="cpu",
):
    model.to(device)
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # (batch_size, 1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        train_loss_avg = np.mean(train_losses)
        val_loss_avg   = evaluate(model, val_loader, criterion, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss_avg:.6f} "
              f"Val Loss: {val_loss_avg:.6f}")

        # Track the best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("\nTraining finished.")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    return model

def get_predictions(model, data_loader, device="cpu"):
    model.eval()
    all_targets = []
    all_preds   = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())
    return np.array(all_targets), np.array(all_preds)

# ---------------------------
# 10. Main Execution
# ---------------------------
if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        device=device
    )
    
    # Evaluate on the test set
    y_true, y_pred = get_predictions(model, test_loader, device=device)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    print("\nRegression Metrics on Test Set:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")

    # Save metrics
    metrics_dict = {
        "mse": [mse],
        "mae": [mae],
        "r2":  [r2]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv("test_metrics_regression.csv", index=False)
    print("\nSaved test_metrics_regression.csv")
    
    # Save predictions vs. true
    df_results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    df_results.to_csv("predictions_regression.csv", index=False)
    print("Saved predictions_regression.csv")

    print("\nDone!")
