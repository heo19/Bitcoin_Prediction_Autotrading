import pandas as pd
import os
from binance.client import Client
import datetime
import pandas_ta as ta

# ====== SETTINGS ======
API_KEY = "YOUR API KEY"
API_SECRET = "YOUR API SECRET"
SYMBOL = "BTCUSDT"                 # The trading pair you want (e.g., BTCUSDT)
INTERVAL = Client.KLINE_INTERVAL_1HOUR  # or KLINE_INTERVAL_5MINUTE, etc.

START_DATE = "2020-01-01 00:00:00"
END_DATE = "2025-01-01 23:59:59"
CSV_FILENAME = "binance_btcusdt_1HOUR.csv"  # CSV file name to save

# Initialize client
client = Client(API_KEY, API_SECRET, tld='us')

print(f"Fetching {SYMBOL} candlesticks from {START_DATE} to {END_DATE}...")
candles = client.get_historical_klines(
    SYMBOL,
    INTERVAL,
    START_DATE,
    END_DATE
)

# Convert into DataFrame
df = pd.DataFrame(candles, columns=[
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_asset_volume",
    "number_of_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore"
])

# Convert numeric columns
numeric_cols = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "taker_buy_base_volume",
    "taker_buy_quote_volume"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)

# Convert timestamps
df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

df.drop(columns=["ignore"], inplace=True)
df = df.sort_values("open_time").reset_index(drop=True)

print("Candlestick DataFrame sample (first 5 rows):")
print(df.head())

# Ensure the columns are numeric before applying technical analysis
df["open"] = pd.to_numeric(df["open"])
df["high"] = pd.to_numeric(df["high"])
df["low"] = pd.to_numeric(df["low"])
df["close"] = pd.to_numeric(df["close"])

# Apply technical analysis indicators
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.rsi(length=30, append=True)
df.ta.hma(length=20, append=True)
df.ta.hma(length=50, append=True)
df.ta.hma(length=200, append=True)
df.ta.bbands(length=50, std=2, append=True)
df.ta.bbands(length=200, std=2, append=True)
df.ta.obv(append=True)

print("\nDataFrame with indicators (tail 5 rows):")
print(df.tail(5))

# ========== 5) SAVE TO CSV ==========

os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", CSV_FILENAME)
df.to_csv(output_path, index=False)

print(f"\nAll data + indicators saved to: {output_path}")