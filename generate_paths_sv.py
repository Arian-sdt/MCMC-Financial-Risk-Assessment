import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Load data
# ==============================
data = pd.read_csv('GSPC_clean.csv')
data['Date'] = pd.to_datetime(data['Date'])

x = data['LogReturn'].astype(float).to_numpy()
x = x[np.isfinite(x)]
x = x - np.mean(x)

# keep dates aligned with x in case of dropped non-finite values
dates = data.loc[np.isfinite(data['LogReturn'].astype(float).to_numpy()), 'Date'].reset_index(drop=True)

# ==============================
# 2. Posterior mean parameters
# ==============================
mu = -10.5950
phi = 0.2151
sigma2_h = 3.5289
nu = 8.0

# ==============================
# 3. Simulate SV returns
# ==============================
def simulate_sv(T, mu, phi, sigma2_h, nu, S0):
    h = np.zeros(T)
    r = np.zeros(T)

    # stationary initialization for h_0
    h[0] = np.random.normal(mu, np.sqrt(sigma2_h / (1.0 - phi**2)))

    # first return
    eps0 = np.random.standard_t(df=nu)
    r[0] = np.exp(h[0] / 2.0) * eps0

    for t in range(1, T):
        # latent log-volatility
        h[t] = mu + phi * (h[t-1] - mu) + np.random.normal(0.0, np.sqrt(sigma2_h))

        # return
        eps_t = np.random.standard_t(df=nu)
        r[t] = np.exp(h[t] / 2.0) * eps_t

    # price path
    S = np.zeros(T)
    S[0] = S0
    for t in range(1, T):
        S[t] = S[t-1] * np.exp(r[t])

    return S, h, r

# ==============================
# 4. Generate simulated paths
# ==============================
T = len(x)
S0 = data['Close'].iloc[0]

n_paths = 5
simulated_paths = []
simulated_h = []
simulated_r = []

for _ in range(n_paths):
    S_sim, h_sim, r_sim = simulate_sv(T, mu, phi, sigma2_h, nu, S0)
    simulated_paths.append(S_sim)
    simulated_h.append(h_sim)
    simulated_r.append(r_sim)

# ==============================
# 5. Plot real + simulated price paths
# ==============================
plt.figure(figsize=(12, 6))

plt.plot(dates, data['Close'].iloc[:T].to_numpy(), color='green', linewidth=2, label='Real')

for i, S_sim in enumerate(simulated_paths):
    plt.plot(dates, S_sim, linestyle='--', alpha=0.7, label=f'SV Sim {i+1}')

plt.title('Real vs Simulated Stochastic Volatility Market Paths')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 6. Plot simulated volatility paths
# ==============================
plt.figure(figsize=(12, 6))

for i, h_sim in enumerate(simulated_h):
    vol_sim = np.exp(h_sim / 2.0)
    plt.plot(dates, vol_sim, linestyle='--', alpha=0.7, label=f'SV Vol {i+1}')

plt.title('Simulated Stochastic Volatility Paths')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 7. Plot one SV volatility path against absolute returns
# ==============================
vol_sim = np.exp(simulated_h[0] / 2.0)
abs_returns = np.abs(x)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(dates, vol_sim, color='red')
plt.title(r'Estimated / Simulated Volatility $\exp(h_t/2)$')
plt.ylabel('Volatility')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(dates, abs_returns, color='black')
plt.title(r'Absolute Returns $|r_t|$')
plt.xlabel('Date')
plt.ylabel(r'$|r_t|$')
plt.grid(True)

plt.tight_layout()
plt.show()