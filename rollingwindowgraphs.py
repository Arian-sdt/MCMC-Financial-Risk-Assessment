import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# File paths
# ==========================================
garch_csv = "rolling_bayesian_garch_forecasts.csv"
sv_csv = "rolling_bayesian_sv_forecasts.csv"

if not Path(garch_csv).exists():
    raise FileNotFoundError(f"Cannot find {garch_csv}")

if not Path(sv_csv).exists():
    raise FileNotFoundError(f"Cannot find {sv_csv}")

# ==========================================
# Load data
# ==========================================
garch = pd.read_csv(garch_csv)
sv = pd.read_csv(sv_csv)

garch["Date"] = pd.to_datetime(garch["Date"])
sv["Date"] = pd.to_datetime(sv["Date"])

# Sort just in case
garch = garch.sort_values("Date").reset_index(drop=True)
sv = sv.sort_values("Date").reset_index(drop=True)

# ==========================================
# Merge on common dates
# ==========================================
merged = pd.merge(
    garch,
    sv,
    on="Date",
    suffixes=("_GARCH", "_SV")
)

if merged.empty:
    raise ValueError("No overlapping dates found between the two forecast CSV files.")

# ==========================================
# 1. GARCH forecast vs actual |return|
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(merged["Date"], merged["PredVolMean_GARCH"], label="Predicted GARCH volatility")
plt.fill_between(
    merged["Date"],
    merged["PredVolQ025_GARCH"],
    merged["PredVolQ975_GARCH"],
    alpha=0.25,
    label="GARCH 95% posterior interval"
)
plt.plot(
    merged["Date"],
    merged["ActualAbsReturn_GARCH"],
    label="Actual |return|",
    alpha=0.3,
    linewidth=1
)
plt.title("Rolling Bayesian GARCH One-Step-Ahead Volatility Forecast")
plt.xlabel("Date")
plt.ylabel("Volatility / |Return|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 2. SV forecast vs actual |return|
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(merged["Date"], merged["PredVolMean_SV"], label="Predicted SV volatility")
plt.fill_between(
    merged["Date"],
    merged["PredVolQ025_SV"],
    merged["PredVolQ975_SV"],
    alpha=0.25,
    label="SV 95% posterior interval"
)
plt.plot(merged["Date"], merged["ActualAbsReturn_SV"], label="Actual |return|",
    alpha=0.3,
    linewidth=1)
plt.title("Rolling Bayesian SV One-Step-Ahead Volatility Forecast")
plt.xlabel("Date")
plt.ylabel("Volatility / |Return|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 3. GARCH vs SV vs actual on same plot
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(merged["Date"], merged["PredVolMean_GARCH"], label="GARCH predicted volatility")
plt.plot(merged["Date"], merged["PredVolMean_SV"], label="SV predicted volatility")
plt.plot(merged["Date"], merged["ActualAbsReturn_GARCH"], label="Actual |return|", alpha=0.8)
plt.title("GARCH vs SV Rolling One-Step-Ahead Volatility Forecasts")
plt.xlabel("Date")
plt.ylabel("Volatility / |Return|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 4. GARCH acceptance rates
# ==========================================
if "WindowAcceptRate" in merged.columns and "PilotAcceptRate_GARCH" in merged.columns:
    plt.figure(figsize=(12, 4))
    plt.plot(merged["Date"], merged["WindowAcceptRate"], label="GARCH main MH acceptance")
    plt.plot(merged["Date"], merged["PilotAcceptRate_GARCH"], label="GARCH pilot RW acceptance")
    plt.title("Acceptance Rates Across Rolling GARCH Windows")
    plt.xlabel("Date")
    plt.ylabel("Acceptance rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. SV acceptance rates
# ==========================================
sv_accept_cols = {"ParamAcceptRate", "LatentAcceptRate", "PilotAcceptRate_SV"}
if sv_accept_cols.issubset(set(merged.columns)):
    plt.figure(figsize=(12, 4))
    plt.plot(merged["Date"], merged["ParamAcceptRate"], label="SV parameter acceptance")
    plt.plot(merged["Date"], merged["LatentAcceptRate"], label="SV latent acceptance")
    plt.plot(merged["Date"], merged["PilotAcceptRate_SV"], label="SV pilot acceptance")
    plt.title("Acceptance Rates Across Rolling SV Windows")
    plt.xlabel("Date")
    plt.ylabel("Acceptance rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. Save merged comparison file
# ==========================================
merged.to_csv("merged_garch_sv_forecasts.csv", index=False)
print("Saved merged comparison file to merged_garch_sv_forecasts.csv")