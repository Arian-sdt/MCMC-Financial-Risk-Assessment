import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load your CSV ---
data = pd.read_csv('GSPC_clean.csv')  # replace with your file name

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# --- 2. Plot daily log returns ---
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['LogReturn'], color='blue', linewidth=0.8)
plt.title('Daily Log Returns of S&P 500')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.grid(True)
plt.tight_layout()

# --- 3. Show the plot ---
plt.show()
