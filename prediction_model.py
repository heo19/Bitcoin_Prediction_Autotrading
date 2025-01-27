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

# ---------------------------------------
# 1. Load Data
# ---------------------------------------
df = pd.read_csv("./data/binance_btcusdt_1HOUR.csv")  # Read CSV data into a pandas DataFrame
df = df.iloc[200:].reset_index(drop=True)             # Slice the DataFrame from row 200 onward and reset the index

# ---------------------------------------
# 2. Select columns
# ---------------------------------------
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
data = df[features].values    # Extract the numpy array of the selected features from the DataFrame
if np.isnan(data).any():
    print("Data contains NaN values. Handling NaNs...")
    data = np.nan_to_num(data, nan=0.0)  # Replace NaN values with 0.0

# ---------------------------------------
# 3. Scale data
# ---------------------------------------
scaler = MinMaxScaler()         # Instantiate a MinMaxScaler object
data_scaled = scaler.fit_transform(data)  # Fit the scaler to the data and transform it

# Let X be all (27) features
X_all = data_scaled

# Let y be the single column for 'close' in scaled form
close_idx = features.index("close")   # Get the index of "close" in the features list
y_all = data_scaled[:, close_idx]     # Extract just the scaled "close" values as our target

# ---------------------------------------
# create_sliding_windows
# ---------------------------------------
def create_sliding_windows(X, y, lookback=30):
    """
    Creates input (X_slid) and target (y_slid) sequences using a specified lookback window.
    For each position i, it takes X[i : i+lookback] as inputs and y[i+lookback] as the label.
    """
    X_slid, y_slid = [], []
    for i in range(len(X) - lookback):        # Iterate until we have enough data for the lookback window
        X_slid.append(X[i : i + lookback])    # Append a window of size 'lookback' to X_slid
        y_slid.append(y[i + lookback])        # The label is the value that comes after the lookback period
    return np.array(X_slid), np.array(y_slid)

# Use the function
lookback = 24                                  # Number of past time steps to consider
X_slid, y_slid = create_sliding_windows(X_all, y_all, lookback=lookback)

# ---------------------------------------
# 5. Train/Val/Test Split
# ---------------------------------------
train_size = int(0.7 * len(X_slid))       # 70% of data for training
val_size = int(0.15 * len(X_slid))        # 15% of data for validation
test_size = len(X_slid) - train_size - val_size  # Remainder for testing

X_train = X_slid[:train_size]            # Slicing the first 'train_size' samples for training
y_train = y_slid[:train_size]

X_val = X_slid[train_size : train_size + val_size]  # Next 'val_size' samples for validation
y_val = y_slid[train_size : train_size + val_size]

X_test = X_slid[train_size + val_size :]  # Remaining samples for test
y_test = y_slid[train_size + val_size :]

# ---------------------------------------
# TimeSeriesDataset
# ---------------------------------------
class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data. 
    Stores inputs (X) and targets (y) and returns them by index.
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)    # Return total number of samples

    def __getitem__(self, idx):
        # Return the sample and target at position idx as tensors
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

# ---------------------------------------
# 6. Create Datasets / Loaders
# ---------------------------------------
train_dataset = TimeSeriesDataset(X_train, y_train)  # Create dataset for training data
val_dataset = TimeSeriesDataset(X_val, y_val)        # Create dataset for validation data
test_dataset = TimeSeriesDataset(X_test, y_test)     # Create dataset for test data

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   # Loader for training, with shuffling
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)      # Loader for validation
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    # Loader for testing

# ---------------------------------------
# 7. LSTM Model Definition
# ---------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=64, num_layers=1, dropout=0.0):
        """
        An LSTM model for time series regression.
        input_dim  : number of input features
        hidden_dim : number of units in LSTM hidden layer
        num_layers : number of LSTM layers
        dropout    : dropout probability
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Fully connected layer to map LSTM output to a single value
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]                                     # Get the batch size from the input
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  # Initialize cell state

        out, (hn, cn) = self.lstm(x, (h0, c0))                      # Forward pass through LSTM
        out = out[:, -1, :]                                         # Take the last time-step's output
        out = self.fc(out)                                          # Pass the last output through the FC layer
        return out.squeeze(-1)                                      # Squeeze the last dimension for final predictions

# ---------------------------------------
# 8. Train Function
# ---------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device="cpu"):
    """
    Trains the model for a given number of epochs and prints out the training and validation loss.
    """
    model.to(device)  # Move model to specified device (CPU or GPU)
    
    for epoch in range(num_epochs):
        model.train()         # Set model to training mode
        train_losses = []     # List to store training losses each batch
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)  # Move batch to device
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()         # Reset gradients
            outputs = model(X_batch)      # Get predictions from the model
            loss = criterion(outputs, y_batch)  # Calculate loss
            loss.backward()               # Backpropagation
            optimizer.step()              # Update parameters
            
            train_losses.append(loss.item())  # Store current batch loss
        
        val_loss = evaluate(model, val_loader, criterion, device=device)  # Compute validation loss
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {np.mean(train_losses):.6f} "
              f"Val Loss: {val_loss:.6f}")
    
    return model

# ---------------------------------------
# 9. Evaluate Function (Returns MSE only, for convenience in training)
# ---------------------------------------
def evaluate(model, data_loader, criterion, device="cpu"):
    """
    Evaluates the model on the given data_loader using the provided criterion (MSELoss in this case).
    Returns the average loss across all batches.
    """
    model.eval()     # Set model to evaluation mode
    losses = []      # List to store losses
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)  # Move data to device
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)      # Perform forward pass
            loss = criterion(outputs, y_batch)  # Calculate loss
            losses.append(loss.item())    # Store loss
    
    return np.mean(losses)  # Return mean loss

# ---------------------------------------
# Additional function: get_predictions
#    This will help us compute metrics like RMSE, MAE, R², etc.
# ---------------------------------------
def get_predictions(model, data_loader, device="cpu"):
    """
    Returns:
       all_targets: ground truth values (as a numpy array)
       all_preds:   model predictions (as a numpy array)
    """
    model.eval()         # Set model to evaluation mode
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch)             # Model predictions
            all_preds.extend(preds.cpu().numpy())   # Append predictions to list (convert to CPU)
            all_targets.extend(y_batch.cpu().numpy())# Append true values
    
    return np.array(all_targets), np.array(all_preds)

# ---------------------------------------
# 10. Main Execution
# ---------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device: GPU if available, else CPU
    model = LSTMModel(input_dim=27, hidden_dim=64, num_layers=1, dropout=0.0).to(device)  # Instantiate our LSTM model
    
    criterion = nn.MSELoss()                  # Mean Squared Error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer with learning rate 0.001
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    # Evaluate on test set (basic MSE)
    test_loss = evaluate(model, test_loader, criterion, device=device)
    print("Test Loss (MSE):", test_loss)
    
    # Save model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)               # Create directory if it doesn't exist

    model_path = os.path.join(save_dir, "lstm_model.pth")
    torch.save(model.state_dict(), model_path)         # Save the model's weights
    print(f"Model weights saved to {model_path}")

    # 4. SAVE the scaler
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)                   # Save the fitted MinMaxScaler
    print(f"Scaler saved to {scaler_path}")

    # ---------------------------------------
    # Compute more metrics
    # ---------------------------------------
    y_true, y_pred = get_predictions(model, test_loader, device=device)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred)  # MSE
    rmse = math.sqrt(mse)                     # RMSE
    mae = mean_absolute_error(y_true, y_pred) # MAE
    r2 = r2_score(y_true, y_pred)             # R²
    
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
    df_metrics = pd.DataFrame(metrics_dict)        # Create a DataFrame with the metrics
    df_metrics.to_csv("test_metrics.csv", index=False)  # Save metrics to CSV file
    print("\nSaved test_metrics.csv")
    
    # Save predictions vs. targets to CSV
    df_results = pd.DataFrame({
        "y_true_scaled": y_true,
        "y_pred_scaled": y_pred
    })
    df_results.to_csv("predictions_scaled.csv", index=False)  # Save scaled predictions
    print("Saved predictions_scaled.csv")
    
    # Optionally invert-scale predictions if you need real price
    all_unscaled_preds = []
    all_unscaled_true = []
    
    for true_val_scaled, pred_val_scaled in zip(y_true, y_pred):
        # Create a dummy row of the same shape as the input to 'scaler'
        dummy_row_true = np.zeros((1, len(features)))        # A row of zeros
        dummy_row_true[0, close_idx] = true_val_scaled       # Place the scaled true close in the "close" position
        
        dummy_row_pred = np.zeros((1, len(features)))        # Another row of zeros
        dummy_row_pred[0, close_idx] = pred_val_scaled       # Place the scaled predicted close in the "close" position
        
        # Inverse transform each dummy row
        unscaled_true_val = scaler.inverse_transform(dummy_row_true)[0, close_idx]  # Recover actual close price
        unscaled_pred_val = scaler.inverse_transform(dummy_row_pred)[0, close_idx]
        
        all_unscaled_true.append(unscaled_true_val)          # Collect actual unscaled close
        all_unscaled_preds.append(unscaled_pred_val)         # Collect predicted unscaled close
    
    df_unscaled = pd.DataFrame({
        "y_true_unscaled": all_unscaled_true,
        "y_pred_unscaled": all_unscaled_preds
    })
    df_unscaled.to_csv("predictions_unscaled.csv", index=False)  # Save unscaled predictions
    print("Saved predictions_unscaled.csv")

    print("\nDone!")
