import yfinance as yf
import pandas as pd
import numpy as np

# --- Parameters ---
ticker = '^GSPC'  # S&P 500 index
start_date = '2005-01-01'
end_date = '2025-01-01'
output_file = 'GSPC_clean.csv'

# --- Download data from Yahoo Finance ---
data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

# --- Keep only the Close column and reset index ---
data = data[['Close']].copy()
data.reset_index(inplace=True)

# --- Drop any rows with missing or invalid data ---
data = data.dropna(subset=['Close'])

# --- Compute daily log returns ---
data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))

# --- Drop first row with NaN log return ---
data = data.iloc[1:].copy()

# --- Save cleaned CSV ---
data.to_csv(output_file, index=False)

print(f"Done! Cleaned CSV saved as '{output_file}'")
