import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight

# ---------------------------
# 1. Load and Prepare Data
# ---------------------------
df = pd.read_csv("./data/binance_btcusdt_15MIN.csv")
df = df.iloc[300:].reset_index(drop=True)

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
    "OBV"
]

# If label doesn't exist, create it
if "label" not in df.columns:
    df["next_close"] = df["close"].shift(-1)
    df["price_change"] = df["next_close"] - df["close"]
    df["label"] = (df["price_change"] > 0).astype(int)
    df.dropna(subset=["next_close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Binary 'label' column created.")
else:
    print("Label column already exists. Proceeding with existing labels.")

# ---------------------------
# 2. Scale Data (Features Only)
# ---------------------------
data = df[features].values
if np.isnan(data).any():
    print("Data contains NaN values. Handling NaNs...")
    data = np.nan_to_num(data, nan=0.0)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X_all = data_scaled
y_all = df["label"].values  # 0 or 1

# ---------------------------
# 3. Create Sliding Windows
# ---------------------------
def create_sliding_windows(X, y, lookback=24):
    X_slid, y_slid = [], []
    for i in range(len(X) - lookback):
        X_slid.append(X[i : i + lookback])
        y_slid.append(y[i + lookback])
    return np.array(X_slid), np.array(y_slid)

lookback = 96
X_slid, y_slid = create_sliding_windows(X_all, y_all, lookback)

# ---------------------------
# 4. Train/Val/Test Split
# ---------------------------
train_size = int(0.7 * len(X_slid))
val_size = int(0.15 * len(X_slid))
test_size = len(X_slid) - train_size - val_size

X_train = X_slid[:train_size]
y_train = y_slid[:train_size]

X_val = X_slid[train_size : train_size + val_size]
y_val = y_slid[train_size : train_size + val_size]

X_test = X_slid[train_size + val_size :]
y_test = y_slid[train_size + val_size :]

print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

# ---------------------------
# 5. Dataset Definition
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
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------
# 6. Simple CNN Model
#     - 2 Convolution Blocks
#     - No BatchNorm
#     - No Dropout
# ---------------------------
class SimpleCNN(nn.Module):
    """
    A small 2-layer CNN for debugging/troubleshooting.
    Removes BatchNorm and Dropout.
    """
    def __init__(self, input_dim=27, num_filters=64, kernel_size=3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same"
        )
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            padding="same"
        )
        
        self.fc = nn.Linear(num_filters * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, lookback, input_dim)
        # permute to (batch_size, input_dim, lookback)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)    # (batch_size, num_filters, lookback)
        x = self.relu(x)
        
        x = self.conv2(x)    # (batch_size, num_filters*2, lookback)
        x = self.relu(x)
        
        # global average pooling over time dimension
        x = torch.mean(x, dim=-1)  # (batch_size, num_filters*2)
        
        x = self.fc(x)            # (batch_size, 1)
        return x

# ---------------------------
# 7. Loss & Optimizer
# ---------------------------
criterion = nn.BCEWithLogitsLoss()

# Lower learning rate if training is unstable, or higher if stuck
# We'll set it to 1e-4 to see if we can move away from 0.693 easily
lr = 1e-4

# ---------------------------
# 8. Training & Evaluation
# ---------------------------
def evaluate(model, data_loader, criterion, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1).float()
            
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
            y_batch = y_batch.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        train_loss_avg = np.mean(train_losses)
        val_loss_avg = evaluate(model, val_loader, criterion, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss_avg:.6f} "
              f"Val Loss: {val_loss_avg:.6f}")

        # We won't do early stopping for debugging:
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict()

    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Print final debug info
    print("\nDebug Info:")
    print(f"Final Train Loss: {train_loss_avg:.6f}")
    print(f"Best Val Loss:    {best_val_loss:.6f}")
    return model

def get_predictions(model, data_loader, device="cpu"):
    model.eval()
    all_targets = []
    all_logits = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch)
            all_logits.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    return np.array(all_targets), np.array(all_logits)

def check_label_distribution(y, dataset_name="Dataset"):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"\nLabel distribution in {dataset_name}:")
    for label, count in distribution.items():
        percentage = (count / len(y)) * 100
        print(f"Label {int(label)}: {count} samples ({percentage:.2f}%)")

# ---------------------------
# 9. Main Execution
# ---------------------------
if __name__ == "__main__":
    # Print label distribution
    check_label_distribution(y_train, "Training Set")
    check_label_distribution(y_val, "Validation Set")
    check_label_distribution(y_test, "Test Set")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the simpler CNN model
    model = SimpleCNN(input_dim=27, num_filters=64, kernel_size=3)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model (no early stopping)
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
    y_true, y_logits = get_predictions(model, test_loader, device=device)
    
    # Convert logits to probabilities
    y_prob = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(int)
    
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)
    
    print("\nClassification Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # Save classification metrics
    metrics_dict = {
        "accuracy":  [accuracy],
        "precision": [precision],
        "recall":    [recall],
        "f1_score":  [f1],
        "roc_auc":   [roc_auc]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv("test_metrics_classification.csv", index=False)
    print("\nSaved test_metrics_classification.csv")
    
    # Save predictions
    df_results = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob.flatten(),
        "y_pred": y_pred.flatten()
    })
    df_results.to_csv("predictions_classification.csv", index=False)
    print("Saved predictions_classification.csv")
    
    print("\nDone!")
