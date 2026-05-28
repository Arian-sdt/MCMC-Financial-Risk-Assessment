import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Load data
# ==============================
data = pd.read_csv('GSPC_clean.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Demeaned log returns
x = data['LogReturn'].astype(float).to_numpy()
x = x[np.isfinite(x)]
x = x - np.mean(x)

# ==============================
# 2. Posterior mean parameters
# ==============================
omega = 2.84e-6
alpha = 0.1294
beta  = 0.8455

# ==============================
# 3. Simulate GARCH(1,1) returns
# ==============================
def simulate_garch(T, omega, alpha, beta, S0):
    r = np.zeros(T)
    sigma2 = np.zeros(T)

    # Start at unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, T):
        eps = np.random.normal()
        r[t] = np.sqrt(sigma2[t-1]) * eps
        sigma2[t] = omega + alpha * r[t]**2 + beta * sigma2[t-1]

    # Convert returns to price path
    S = np.zeros(T)
    S[0] = S0
    for t in range(1, T):
        S[t] = S[t-1] * np.exp(r[t])

    return S

# ==============================
# 4. Generate simulated paths
# ==============================
T = len(data)
S0 = data['Close'].iloc[0]

n_paths = 5
simulated_paths = []

for _ in range(n_paths):
    S_sim = simulate_garch(T, omega, alpha, beta, S0)
    simulated_paths.append(S_sim)

# ==============================
# 5. Plot real + simulated paths
# ==============================
plt.figure(figsize=(12,6))

# Real path
plt.plot(data['Date'], data['Close'], color='green', linewidth=2, label='Real')

# Simulated paths
for i, S_sim in enumerate(simulated_paths):
    plt.plot(data['Date'], S_sim, linestyle='--', alpha=0.7, label=f'Sim {i+1}')

plt.title('Real vs Simulated GARCH(1,1) Market Paths')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()