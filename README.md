# Bitcoin Up/Down Classification Using a 1D CNN

## Overview

This project demonstrates a **1D Convolutional Neural Network (CNN)** approach to predict whether Bitcoin’s price will move **up or down** on the next 15-minute bar, given historical data and technical indicators. By transforming the raw time series data into sliding windows and feeding these sequences into a CNN, the model aims to classify each subsequent time step as either a price increase (label `1`) or a price decrease (label `0`).

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
- [Saving and Loading Models](#saving-and-loading-models) *(Optional Section)*
- [Contributing](#contributing)
- [License](#license)

## Features

- **Time-Series Windowing**: Utilizes a sliding-window approach (configurable `lookback`) to generate sequences for classification.
- **Technical Indicators**: Incorporates MACD, RSI, Hull Moving Averages, Bollinger Bands, and OBV to enrich predictive signals.
- **MinMax Scaling**: Normalizes feature data for stable training.
- **Simple 1D CNN**: A straightforward 2-layer convolutional model (no BatchNorm or Dropout in the default code) for quick debugging and experimentation.
- **Train/Validation/Test Splits**: Automatically partitions the data for a robust evaluation of model performance.
- **Classification Metrics**: Outputs accuracy, precision, recall, F1-score, and ROC AUC on the test set.
- **CSV Logging**: Saves metrics and predictions (`y_true`, `y_prob`, `y_pred`) to CSV files for post-analysis.

## Dataset

The project expects a CSV file (e.g., `binance_btcusdt_15MIN.csv`) containing:

- **Candle Data**:  
  - `open`, `high`, `low`, `close`, `volume`  
  - `quote_asset_volume`, `number_of_trades`  
  - `taker_buy_base_volume`, `taker_buy_quote_volume`
- **Indicators**:  
  - MACD (`MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9`)  
  - RSI (`RSI_30`)  
  - Hull Moving Averages (`HMA_20`, `HMA_50`, `HMA_200`)  
  - Bollinger Bands (e.g., `BBL_50_2.0`, `BBM_50_2.0`, `BBU_50_2.0`, etc. for length=50 and 200)  
  - On-Balance Volume (`OBV`)
- **Binary Label**:  
  - `label` column (`0` if the next bar’s close is lower, `1` if higher).  
  - If the script does not find a `label` column, it creates one automatically by comparing `next_close` with the current `close`.

By default, the script uses a **15-minute** interval dataset of BTC/USDT from Binance. You can adapt it to other intervals or symbols as long as the CSV includes the same features.

## Prerequisites

- **Python**: 3.7 or higher recommended
- **Libraries**:
  - `numpy`
  - `pandas`
  - `torch` (PyTorch)
  - `scikit-learn` (for MinMaxScaler, metrics)
  - `pandas_ta` (if you are generating technical indicators separately)

## Installation

1. **Clone the Repository** (or download the project folder):
    ```bash
    git clone https://github.com/yourusername/btc-cnn-classification.git
    cd btc-cnn-classification
    ```

2. **(Optional) Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Or: venv\Scripts\activate (Windows)
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    If you don’t have a `requirements.txt`, install manually:
    ```bash
    pip install numpy pandas torch scikit-learn
    ```

## Usage

### Data Preparation

1. **Place Your CSV in the `data/` Directory**  
   Make sure it’s named `binance_btcusdt_15MIN.csv` (or adjust the script accordingly).

2. **Verify Column Names**  
   Ensure your file includes the core candle columns, technical indicators, and the `label` column (or allow the script to generate it).

3. **Check for NaNs**  
   The script replaces any `NaN` with `0.0`. Modify if you need a different imputation strategy.

### Model Training

1. **Run the Script**:
    ```bash
    python prediction_model.py
    ```

2. **Sliding Window**:
   - The code uses a **`lookback`** window (e.g., 96) to form a sequence of 96 time steps for each training sample.
   - Adjust `lookback` in the script if needed.

3. **Monitor Training**:
   - You’ll see `Train Loss` and `Val Loss` every epoch.
   - By default, the script runs 50 epochs with no early stopping (for debugging).

### Evaluation

After the final epoch, the script:

- Evaluates on the **test set** and prints classification metrics:
  - **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **ROC AUC**.
- Saves:
  - **test_metrics_classification.csv**: The final metrics.
  - **predictions_classification.csv**: A row for each test sample with columns:  
    `y_true`, `y_prob`, `y_pred`.

## Model Architecture

```text
SimpleCNN
 ├── Conv1d (in_channels=27, out_channels=64, kernel_size=3)
 ├── ReLU
 ├── Conv1d (in_channels=64, out_channels=128, kernel_size=3)
 ├── ReLU
 ├── Global Average Pooling (dim = -1)
 └── FC (128 -> 1)  (Logit output for binary classification)
