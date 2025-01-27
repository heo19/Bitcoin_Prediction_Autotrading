import pandas as pd
import os
from binance.client import Client
import datetime
import pandas_ta as ta

# ====== SETTINGS ======
API_KEY = "YOUR API KEY"                     # << Replace with your own Binance API Key
API_SECRET = "YOUR API SECRET"               # << Replace with your own Binance API Secret
SYMBOL = "BTCUSDT"                           # e.g., BTCUSDT
INTERVAL = Client.KLINE_INTERVAL_1HOUR       # e.g., 1H, 15m, 5m, etc.

START_DATE = "2020-01-01 00:00:00"           # Adjust as needed
END_DATE   = "2025-01-01 23:59:59"           # Adjust as needed
CSV_FILENAME = "binance_btcusdt_1HOUR.csv"   # CSV file name to save

# ---------- 1) INITIALIZE CLIENT ----------
client = Client(API_KEY, API_SECRET, tld='us')

# ---------- 2) FETCH HISTORICAL DATA ----------
print(f"Fetching {SYMBOL} candlesticks from {START_DATE} to {END_DATE}...")
candles = client.get_historical_klines(
    SYMBOL,
    INTERVAL,
    START_DATE,
    END_DATE
)

# ---------- 3) BUILD DATAFRAME ----------
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

# ---------- 4) APPLY TECHNICAL INDICATORS ----------
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.rsi(length=30, append=True)
df.ta.hma(length=20, append=True)
df.ta.hma(length=50, append=True)
df.ta.hma(length=200, append=True)
df.ta.bbands(length=50,  std=2, append=True)
df.ta.bbands(length=200, std=2, append=True)
df.ta.obv(append=True)

print("\nDataFrame with indicators (tail 5 rows):")
print(df.tail(5))

# ---------- 5) CREATE NEXT_CLOSE, PRICE_CHANGE, AND BINARY LABEL ----------
#  next_close  : next row's close price
#  price_change: difference between next close and current close
#  label       : 1 if price goes up, else 0

# Shift the 'close' column up by one row to get the *next* close
df["next_close"] = df["close"].shift(-1)

# Calculate price change
df["price_change"] = df["next_close"] - df["close"]

# Create binary label: 1 if next close is higher than current close, else 0
df["label"] = (df["price_change"] > 0).astype(int)

# Drop the last row because it doesn't have a valid next_close
df.dropna(subset=["next_close"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- 6) SAVE TO CSV ----------
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", CSV_FILENAME)
df.to_csv(output_path, index=False)

print(f"\nAll data + indicators + binary label saved to: {output_path}")
print("\nFinal DataFrame sample (tail 5 rows):")
print(df.tail(5))
