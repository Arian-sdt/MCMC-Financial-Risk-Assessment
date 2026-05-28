import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("merged_garch_sv_forecasts.csv")
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 1. Volatility targeting
# =========================
target_vol = 0.01  # target daily volatility (~1%)

df["w_garch"] = target_vol / df["PredVolMean_GARCH"]
df["w_sv"] = target_vol / df["PredVolMean_SV"]

# cap leverage (important)
df["w_garch"] = df["w_garch"].clip(-3, 3)
df["w_sv"] = df["w_sv"].clip(-3, 3)

# =========================
# 2. Strategy returns
# =========================
df["ret_garch"] = df["w_garch"] * df["ActualReturn_GARCH"]
df["ret_sv"] = df["w_sv"] * df["ActualReturn_GARCH"]

# buy & hold baseline
df["ret_bh"] = df["ActualReturn_GARCH"]

# =========================
# 3. Cumulative returns
# =========================
df["cum_garch"] = (1 + df["ret_garch"]).cumprod()
df["cum_sv"] = (1 + df["ret_sv"]).cumprod()
df["cum_bh"] = (1 + df["ret_bh"]).cumprod()

# =========================
# 4. Plot
# =========================
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["cum_garch"], label="GARCH strategy")
plt.plot(df["Date"], df["cum_sv"], label="SV strategy")
plt.plot(df["Date"], df["cum_bh"], label="Buy & Hold")
plt.title("Volatility Timing Strategy Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

threshold = df["PredVolMean_GARCH"].quantile(0.7)

df["pos_garch"] = (df["PredVolMean_GARCH"] < threshold).astype(int)
df["pos_sv"] = (df["PredVolMean_SV"] < threshold).astype(int)

df["ret_garch_regime"] = df["pos_garch"] * df["ActualReturn_GARCH"]
df["ret_sv_regime"] = df["pos_sv"] * df["ActualReturn_GARCH"]

df["cum_garch_regime"] = (1 + df["ret_garch_regime"]).cumprod()
df["cum_sv_regime"] = (1 + df["ret_sv_regime"]).cumprod()

plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["cum_garch_regime"], label="GARCH regime")
plt.plot(df["Date"], df["cum_sv_regime"], label="SV regime")
plt.plot(df["Date"], df["cum_bh"], label="Buy & Hold")
plt.legend()
plt.title("Regime Switching Strategy")
plt.grid(True)
plt.show()