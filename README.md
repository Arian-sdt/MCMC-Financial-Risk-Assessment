# MCMC Financial Risk Assessment

Bayesian volatility and tail-risk modeling for S&P 500 returns using Markov Chain Monte Carlo (MCMC), Bayesian GARCH, and stochastic volatility models.

## Overview

This project models financial market risk using daily S&P 500 price data from Yahoo Finance. It estimates volatility with Bayesian methods, compares GARCH and stochastic volatility forecasts, simulates market paths, and evaluates simple volatility-based exposure strategies.

The project focuses on:

- Bayesian GARCH(1,1) parameter estimation
- Bayesian stochastic volatility modeling
- MCMC-based posterior sampling
- Rolling one-step-ahead volatility forecasts
- Simulated market paths from fitted models
- Volatility targeting and regime-based exposure strategies
- Visualization of returns, prices, forecasts, and model performance

## Dataset

The repository includes `GSPC_clean.csv`, which contains cleaned S&P 500 data from January 4, 2005 to December 31, 2024.

Columns:

```text
Date
Close
LogReturn
```

The data is generated from Yahoo Finance using `scrape.py`.

## Project Structure

```text
MCMC-Financial-Risk-Assessment/
├── scrape.py                         # Downloads S&P 500 data from Yahoo Finance
├── data_clean.py                     # Prints summary statistics for log returns
├── graph_close.py                    # Plots S&P 500 close prices
├── graph_logReturns.py               # Plots daily log returns
├── maingarch.py                      # Bayesian GARCH(1,1) MCMC estimation
├── mainsv.py                         # Bayesian stochastic volatility MCMC estimation
├── garchslidingwindow.py             # Rolling Bayesian GARCH volatility forecasts
├── svslidingwindow.py                # Rolling Bayesian stochastic volatility forecasts
├── rollingwindowgraphs.py            # Compares and plots GARCH vs SV rolling forecasts
├── generate_paths_garch.py           # Simulates market paths using fitted GARCH parameters
├── generate_paths_sv.py              # Simulates market paths using fitted SV parameters
├── Exposure.py                       # Tests volatility targeting and regime strategies
├── GSPC_clean.csv                    # Cleaned S&P 500 dataset
├── rolling_bayesian_garch_forecasts.csv
├── rolling_bayesian_sv_forecasts.csv
├── merged_garch_sv_forecasts.csv
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Arian-sdt/MCMC-Financial-Risk-Assessment.git
cd MCMC-Financial-Risk-Assessment
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib tqdm yfinance
```

## Usage

### 1. Download and clean S&P 500 data

```bash
python3 scrape.py
```

This downloads S&P 500 close prices from Yahoo Finance, calculates daily log returns, and saves the result to:

```text
GSPC_clean.csv
```

### 2. Inspect return statistics

```bash
python3 data_clean.py
```

This prints the mean, standard deviation, and variance of the log returns.

### 3. Plot the data

Plot close prices:

```bash
python3 graph_close.py
```

Plot daily log returns:

```bash
python3 graph_logReturns.py
```

### 4. Run Bayesian GARCH estimation

```bash
python3 maingarch.py
```

This estimates a Bayesian GARCH(1,1) model using MCMC and prints posterior summaries for:

- omega
- alpha
- beta
- rho = alpha + beta
- delta = alpha / (alpha + beta)

### 5. Run Bayesian stochastic volatility estimation

```bash
python3 mainsv.py
```

This estimates a stochastic volatility model with latent log-volatility states and Student-t returns.

It prints posterior summaries for:

- mu
- phi
- sigma2

### 6. Generate rolling volatility forecasts

Run the rolling Bayesian GARCH forecast:

```bash
python3 garchslidingwindow.py
```

Output:

```text
rolling_bayesian_garch_forecasts.csv
```

Run the rolling Bayesian stochastic volatility forecast:

```bash
python3 svslidingwindow.py
```

Output:

```text
rolling_bayesian_sv_forecasts.csv
```

Both scripts use rolling windows to produce one-step-ahead volatility forecasts over the most recent five years of data.

### 7. Compare GARCH and stochastic volatility forecasts

```bash
python3 rollingwindowgraphs.py
```

This merges the GARCH and SV forecast outputs, plots the forecast comparison, and saves:

```text
merged_garch_sv_forecasts.csv
```

### 8. Simulate market paths

Simulate paths using fitted GARCH parameters:

```bash
python3 generate_paths_garch.py
```

Simulate paths using fitted stochastic volatility parameters:

```bash
python3 generate_paths_sv.py
```

These scripts plot real S&P 500 prices against simulated price paths.

### 9. Evaluate exposure strategies

```bash
python3 Exposure.py
```

This script compares:

- GARCH volatility targeting
- SV volatility targeting
- Buy-and-hold
- GARCH regime switching
- SV regime switching

The strategies use predicted volatility from `merged_garch_sv_forecasts.csv`.

## Methodology

### Bayesian GARCH

The GARCH model estimates time-varying volatility using:

```text
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```

The model uses transformed parameters to enforce valid constraints:

- omega > 0
- alpha > 0
- beta > 0
- alpha + beta < 1

Posterior sampling is performed with Metropolis-Hastings methods.

### Stochastic Volatility

The stochastic volatility model represents returns as:

```text
r_t = exp(h_t / 2) * epsilon_t
```

where `h_t` is a latent log-volatility process and the return shocks follow a Student-t distribution.

The latent volatility follows an AR(1)-style process:

```text
h_t = mu + phi * (h_{t-1} - mu) + eta_t
```

This allows volatility to evolve over time while accounting for heavy-tailed financial returns.

## Outputs

The project produces several forecast and comparison files:

```text
rolling_bayesian_garch_forecasts.csv
rolling_bayesian_sv_forecasts.csv
merged_garch_sv_forecasts.csv
```

These contain predicted volatility means, posterior forecast intervals, actual returns, absolute returns, squared returns, and MCMC acceptance rates.

## Dependencies

Required Python packages:

```text
numpy
pandas
matplotlib
tqdm
yfinance
```

The project also uses built-in Python modules such as:

```text
math
pathlib
```

## Notes

Some scripts can take a long time to run because they perform repeated MCMC sampling over rolling windows. For faster experimentation, reduce values such as `n_total`, `burn`, or `window_size` inside the rolling forecast scripts.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice. Model outputs depend on historical data, statistical assumptions, and simulation settings, and should not be treated as guaranteed predictions.

## Author

Created by [Arian-sdt](https://github.com/Arian-sdt).
```
