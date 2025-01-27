# Bitcoin Close Price Prediction Using LSTM

## Overview

This project utilizes a Long Short-Term Memory (LSTM) neural network to predict the next timeframe's closing price of Bitcoin (BTC/USDT) using historical hourly data from Binance. By incorporating various technical indicators and leveraging time series data, the model aims to provide accurate forecasts to inform trading strategies and investment decisions.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Saving and Loading Models](#saving-and-loading-models)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Preprocessing:** Handles missing values and scales data using Min-Max scaling.
- **Feature Engineering:** Incorporates various technical indicators such as MACD, RSI, Hull Moving Averages, Bollinger Bands, and OBV.
- **Time Series Windowing:** Creates sliding windows to structure the data for time series forecasting.
- **Custom Dataset:** Implements a PyTorch `Dataset` for efficient data handling.
- **LSTM Model:** Defines an LSTM-based neural network tailored for regression tasks.
- **Training and Validation:** Includes training loops with validation to monitor performance.
- **Evaluation Metrics:** Computes MSE, RMSE, MAE, and R² to assess model performance.
- **Model Saving:** Saves trained model weights and scaler for future use.
- **Prediction Outputs:** Generates and saves both scaled and unscaled predictions for analysis.

## Dataset

The project uses hourly Bitcoin (BTC/USDT) trading data from Binance, stored in the `binance_btcusdt_1HOUR.csv` file. The dataset includes various price metrics, volume data, and technical indicators essential for predicting the closing price.

### Features

- **Price and Volume:**
  - `open`, `high`, `low`, `close`
  - `volume`, `quote_asset_volume`
  - `number_of_trades`
  - `taker_buy_base_volume`, `taker_buy_quote_volume`

- **Technical Indicators:**
  - **MACD:** `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9`
  - **RSI:** `RSI_30`
  - **Hull Moving Averages:** `HMA_20`, `HMA_50`, `HMA_200`
  - **Bollinger Bands:**
    - Length=50: `BBL_50_2.0`, `BBM_50_2.0`, `BBU_50_2.0`, `BBB_50_2.0`, `BBP_50_2.0`
    - Length=200: `BBL_200_2.0`, `BBM_200_2.0`, `BBU_200_2.0`, `BBB_200_2.0`, `BBP_200_2.0`
  - **OBV:** `OBV`

## Prerequisites

- **Python:** Version 3.7 or higher
- **Libraries:**
  - `numpy`
  - `pandas`
  - `torch` (PyTorch)
  - `scikit-learn`
  - `joblib`

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/bitcoin-close-prediction.git
    cd bitcoin-close-prediction
    ```

2. **Create a Virtual Environment (Optional but Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *If `requirements.txt` is not provided, install manually:*

    ```bash
    pip install numpy pandas torch scikit-learn joblib
    ```

## Usage

The main script performs data loading, preprocessing, model training, evaluation, and saving of results. Follow the steps below to execute the project.

### Data Preparation

1. **Load the Data:**

    The script reads the CSV file containing Binance BTC/USDT hourly data.

    ```python
    df = pd.read_csv("./data/binance_btcusdt_1HOUR.csv")
    df = df.iloc[200:].reset_index(drop=True)
    ```

    *Note: The first 200 rows are skipped to remove initial anomalies or missing values.*

2. **Select Relevant Features:**

    Only specific columns are selected for modeling.

    ```python
    features = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "RSI_30",
        "HMA_20", "HMA_50", "HMA_200",
        "BBL_50_2.0", "BBM_50_2.0", "BBU_50_2.0",
        "BBB_50_2.0", "BBP_50_2.0",
        "BBL_200_2.0", "BBM_200_2.0", "BBU_200_2.0",
        "BBB_200_2.0", "BBP_200_2.0",
        "OBV"
    ]
    data = df[features].values
    ```

3. **Handle Missing Values:**

    Any `NaN` values in the dataset are replaced with `0.0`.

    ```python
    if np.isnan(data).any():
        print("Data contains NaN values. Handling NaNs...")
        data = np.nan_to_num(data, nan=0.0)
    ```

4. **Scale the Data:**

    Features are scaled to the range [0, 1] using `MinMaxScaler`.

    ```python
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    ```

5. **Define Inputs and Targets:**

    - **Inputs (`X_all`):** All selected features.
    - **Target (`y_all`):** Scaled "close" price.

    ```python
    X_all = data_scaled
    close_idx = features.index("close")
    y_all = data_scaled[:, close_idx]
    ```

6. **Create Sliding Windows:**

    Constructs sequences of past `lookback` time steps to predict the next "close" price.

    ```python
    lookback = 24
    X_slid, y_slid = create_sliding_windows(X_all, y_all, lookback=lookback)
    ```

### Model Training

1. **Train/Validation/Test Split:**

    - **Training:** 70%
    - **Validation:** 15%
    - **Testing:** 15%

    ```python
    train_size = int(0.7 * len(X_slid))
    val_size = int(0.15 * len(X_slid))
    test_size = len(X_slid) - train_size - val_size

    X_train = X_slid[:train_size]
    y_train = y_slid[:train_size]

    X_val = X_slid[train_size:train_size+val_size]
    y_val = y_slid[train_size:train_size+val_size]

    X_test = X_slid[train_size+val_size:]
    y_test = y_slid[train_size+val_size:]
    ```

2. **Create Datasets and DataLoaders:**

    Utilizes PyTorch's `Dataset` and `DataLoader` for efficient data handling.

    ```python
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    ```

3. **Define the LSTM Model:**

    An LSTM model with configurable parameters.

    ```python
    model = LSTMModel(input_dim=27, hidden_dim=64, num_layers=1, dropout=0.0).to(device)
    ```

4. **Train the Model:**

    Executes the training loop for a specified number of epochs.

    ```python
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    ```

### Evaluation

1. **Evaluate on Test Set:**

    Computes Mean Squared Error (MSE) on the test data.

    ```python
    test_loss = evaluate(model, test_loader, criterion, device=device)
    print("Test Loss (MSE):", test_loss)
    ```

2. **Compute Additional Metrics:**

    Calculates RMSE, MAE, and R² for a comprehensive evaluation.

    ```python
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
    ```

3. **Save Metrics and Predictions:**

    Outputs are saved for further analysis.

    ```python
    df_metrics.to_csv("test_metrics.csv", index=False)
    df_results.to_csv("predictions_scaled.csv", index=False)
    df_unscaled.to_csv("predictions_unscaled.csv", index=False)
    ```

## Model Architecture

The LSTM model is defined using PyTorch's `nn.Module`. It consists of:

- **LSTM Layer:** Processes the input sequences.
- **Fully Connected Layer:** Maps the LSTM outputs to a single prediction value.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=64, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
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
        out = out[:, -1, :]   # last time-step
        out = self.fc(out)
        return out.squeeze(-1)
