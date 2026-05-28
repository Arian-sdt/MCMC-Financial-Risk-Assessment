import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import lgamma
from pathlib import Path
from tqdm import tqdm

# =========================
# Data loading / preprocessing
# =========================
def load_returns_csv_with_dates(csv_path: str):
    df = pd.read_csv(csv_path)
    if "LogReturn" not in df.columns:
        raise ValueError("CSV must contain a 'LogReturn' column.")
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.loc[np.isfinite(df["LogReturn"].astype(float))].copy()
    df["LogReturn"] = df["LogReturn"].astype(float)
    df["LogReturn"] = df["LogReturn"] - df["LogReturn"].mean()
    df = df.reset_index(drop=True)
    return df


# =========================
# Basic utilities
# =========================
def logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


# =========================
# Parameter transforms
# =========================
def transform_u_to_params(u: np.ndarray):
    """
    u = (u1,u2,u3) in R^3
    mu     = u1
    phi    = logistic(u2) in (0,1)
    sigma2 = exp(u3) > 0
    """
    u1, u2, u3 = float(u[0]), float(u[1]), float(u[2])
    mu = u1
    phi = logistic(u2)
    sigma2 = np.exp(u3)
    return mu, phi, sigma2


def log_jacobian(u: np.ndarray) -> float:
    _, phi, sigma2 = transform_u_to_params(u)
    if phi <= 0 or phi >= 1 or sigma2 <= 0:
        return -np.inf
    return np.log(phi * (1.0 - phi)) + np.log(sigma2)


# =========================
# Priors
# =========================
def log_normal_pdf(x: float, mean: float, var: float) -> float:
    if var <= 0:
        return -np.inf
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var)


def log_beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1:
        return -np.inf
    return (
        lgamma(a + b)
        - lgamma(a)
        - lgamma(b)
        + (a - 1.0) * np.log(x)
        + (b - 1.0) * np.log(1.0 - x)
    )


def log_inv_gamma_pdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return -np.inf
    return a * np.log(b) - lgamma(a) - (a + 1.0) * np.log(x) - b / x


def make_log_prior():
    mu_mean, mu_var = 0.0, 10.0
    phi_a, phi_b = 20.0, 1.5
    sigma2_a, sigma2_b = 2.0, 0.1

    def log_prior(mu: float, phi: float, sigma2: float) -> float:
        lp = 0.0
        lp += log_normal_pdf(mu, mu_mean, mu_var)
        lp += log_beta_pdf(phi, phi_a, phi_b)
        lp += log_inv_gamma_pdf(sigma2, sigma2_a, sigma2_b)
        return lp

    return log_prior, dict(
        mu_mean=mu_mean,
        mu_var=mu_var,
        phi_a=phi_a,
        phi_b=phi_b,
        sigma2_a=sigma2_a,
        sigma2_b=sigma2_b,
    )


# =========================
# Student-t observation density
# =========================
def student_t_logpdf_standard(z: float, nu: float) -> float:
    if nu <= 0:
        return -np.inf
    return (
        lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - ((nu + 1.0) / 2.0) * np.log(1.0 + (z * z) / nu)
    )


def obs_loglik_t(x_t: float, h_t: float, nu: float) -> float:
    scale = np.exp(0.5 * h_t)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    z = x_t / scale
    return student_t_logpdf_standard(z, nu) - np.log(scale)


# =========================
# Latent AR(1) prior for h
# =========================
def log_h_prior(h: np.ndarray, mu: float, phi: float, sigma2: float) -> float:
    if sigma2 <= 0 or phi <= 0 or phi >= 1:
        return -np.inf

    T = len(h)
    if T == 0:
        return -np.inf

    var0 = sigma2 / (1.0 - phi * phi)
    if var0 <= 0 or not np.isfinite(var0):
        return -np.inf

    lp = log_normal_pdf(h[0], mu, var0)

    for t in range(1, T):
        mean_t = mu + phi * (h[t - 1] - mu)
        lp += log_normal_pdf(h[t], mean_t, sigma2)

    return lp


# =========================
# Full posterior pieces
# =========================
def log_obs_likelihood(x: np.ndarray, h: np.ndarray, nu: float) -> float:
    if len(x) != len(h):
        return -np.inf
    ll = 0.0
    for t in range(len(x)):
        term = obs_loglik_t(x[t], h[t], nu)
        if not np.isfinite(term):
            return -np.inf
        ll += term
    return ll


def make_log_posterior_u(x: np.ndarray, h: np.ndarray, nu: float):
    log_prior, prior_hypers = make_log_prior()

    def log_post(u: np.ndarray) -> float:
        mu, phi, sigma2 = transform_u_to_params(u)

        lp = log_prior(mu, phi, sigma2)
        if not np.isfinite(lp):
            return -np.inf

        lh = log_h_prior(h, mu, phi, sigma2)
        if not np.isfinite(lh):
            return -np.inf

        ll = log_obs_likelihood(x, h, nu)
        if not np.isfinite(ll):
            return -np.inf

        lj = log_jacobian(u)
        if not np.isfinite(lj):
            return -np.inf

        return ll + lh + lp + lj

    return log_post, prior_hypers


# =========================
# Local conditional log posterior for one h_t
# =========================
def local_logpost_h_t(
    t: int,
    h_t_star: float,
    h: np.ndarray,
    x: np.ndarray,
    mu: float,
    phi: float,
    sigma2: float,
    nu: float
) -> float:
    T = len(h)

    val = obs_loglik_t(x[t], h_t_star, nu)
    if not np.isfinite(val):
        return -np.inf

    if t == 0:
        var0 = sigma2 / (1.0 - phi * phi)
        val += log_normal_pdf(h_t_star, mu, var0)
    else:
        mean_t = mu + phi * (h[t - 1] - mu)
        val += log_normal_pdf(h_t_star, mean_t, sigma2)

    if t < T - 1:
        mean_next = mu + phi * (h_t_star - mu)
        val += log_normal_pdf(h[t + 1], mean_next, sigma2)

    return val


# =========================
# Pilot RW Metropolis for parameter block
# =========================
def pilot_rw_metropolis_sv(
    x: np.ndarray,
    h: np.ndarray,
    nu: float,
    n_burn=1000,
    n_keep=500,
    step=0.03,
    seed=1
):
    rng = np.random.default_rng(seed)

    mu0 = float(np.log(np.var(x) + 1e-8))
    phi0 = 0.95
    sigma20 = 0.02

    u = np.array([
        mu0,
        np.log(phi0 / (1.0 - phi0)),
        np.log(sigma20)
    ], dtype=float)

    log_post, _ = make_log_posterior_u(x, h, nu)
    lp = log_post(u)

    cov = (step ** 2) * np.eye(3)

    kept = []
    accepted = 0
    total = n_burn + n_keep

    for i in range(total):
        u_star = u + rng.multivariate_normal(np.zeros(3), cov)
        lp_star = log_post(u_star)

        if np.log(rng.uniform()) < (lp_star - lp):
            u, lp = u_star, lp_star
            accepted += 1

        if i >= n_burn:
            kept.append(u.copy())

    kept = np.asarray(kept)
    emp_cov = np.cov(kept.T)
    emp_cov = np.atleast_2d(emp_cov)

    return u, lp, emp_cov + 1e-6 * np.eye(3), accepted / total


# =========================
# Main SV sampler with adaptive RW for params
# =========================
def adaptive_sv_mcmc(
    x: np.ndarray,
    n_total=3000,
    nu=8.0,
    h_step=1.0,
    param_scale=0.10,
    adapt_every=200,
    target_param_accept=0.25,
    burn_adapt=1500,
    seed=42,
    store_h=True
):
    rng = np.random.default_rng(seed)
    T = len(x)

    h = np.log(x * x + 1e-6)

    u, lp, base_cov, pilot_acc = pilot_rw_metropolis_sv(
        x=x,
        h=h,
        nu=nu,
        n_burn=1000,
        n_keep=500,
        step=0.03,
        seed=seed
    )

    log_post_u, prior_hypers = make_log_posterior_u(x, h, nu)

    samples_u = np.empty((n_total, 3), dtype=float)
    samples_h_last = np.empty(n_total, dtype=float)

    if store_h:
        samples_h = np.empty((n_total, T), dtype=float)
    else:
        samples_h = None

    param_accept = 0
    latent_accept = 0
    latent_total = 0

    block_accept = 0
    current_scale = param_scale
    current_cov = current_scale * base_cov + 1e-6 * np.eye(3)

    recent_u = []

    for it in range(n_total):
        mu, phi, sigma2 = transform_u_to_params(u)

        # latent updates
        for t in range(T):
            current = h[t]
            star = current + rng.normal(0.0, h_step)

            log_curr = local_logpost_h_t(t, current, h, x, mu, phi, sigma2, nu)
            log_star = local_logpost_h_t(t, star, h, x, mu, phi, sigma2, nu)

            if np.log(rng.uniform()) < (log_star - log_curr):
                h[t] = star
                latent_accept += 1

            latent_total += 1

        # parameter update
        log_post_u, _ = make_log_posterior_u(x, h, nu)
        lp = log_post_u(u)

        u_star = u + rng.multivariate_normal(np.zeros(3), current_cov)
        lp_star = log_post_u(u_star)

        if np.isfinite(lp_star) and np.log(rng.uniform()) < (lp_star - lp):
            u = u_star
            lp = lp_star
            param_accept += 1
            block_accept += 1

        samples_u[it] = u
        samples_h_last[it] = h[-1]

        if store_h:
            samples_h[it] = h.copy()

        recent_u.append(u.copy())

        if (it + 1) % adapt_every == 0 and (it + 1) <= burn_adapt:
            block_acc_rate = block_accept / adapt_every

            if block_acc_rate > target_param_accept:
                current_scale *= 1.15
            else:
                current_scale *= 0.90

            arr = np.asarray(recent_u)
            if len(arr) >= 10:
                emp_cov = np.cov(arr.T)
                emp_cov = np.atleast_2d(emp_cov)
                current_cov = current_scale * emp_cov + 1e-6 * np.eye(3)

            block_accept = 0
            recent_u = []

    mu_draws = np.empty(n_total)
    phi_draws = np.empty(n_total)
    sigma2_draws = np.empty(n_total)

    for i in range(n_total):
        mu_i, phi_i, sigma2_i = transform_u_to_params(samples_u[i])
        mu_draws[i] = mu_i
        phi_draws[i] = phi_i
        sigma2_draws[i] = sigma2_i

    out = {
        "samples_u": samples_u,
        "samples_h": samples_h,
        "mu": mu_draws,
        "phi": phi_draws,
        "sigma2": sigma2_draws,
        "h_last": samples_h_last,
        "h_final": h.copy(),
        "pilot_accept_rate": pilot_acc,
        "param_accept_rate": param_accept / n_total,
        "latent_accept_rate": latent_accept / max(1, latent_total),
        "prior_hypers": prior_hypers,
        "nu_fixed": nu,
        "final_param_scale": current_scale,
    }
    return out


# =========================
# Rolling Bayesian SV forecast
# =========================
def rolling_bayesian_sv_forecast(
    df: pd.DataFrame,
    years_back=5,
    window_size=100,
    burn=1000,
    n_total=3000,
    nu=8.0,
    h_step=1.0,
    param_scale=0.10,
    adapt_every=200,
    target_param_accept=0.25,
    burn_adapt=1500,
    seed=42
):
    end_date = df["Date"].max()
    start_date = end_date - pd.DateOffset(years=years_back)
    df_last = df.loc[df["Date"] >= start_date].copy().reset_index(drop=True)

    x_all = df_last["LogReturn"].to_numpy()
    dates_all = df_last["Date"].to_numpy()

    if len(x_all) <= window_size:
        raise ValueError("Not enough observations in the last 5 years for the chosen window size.")

    results = []
    rng = np.random.default_rng(seed)

    n_forecasts = len(x_all) - window_size

    for i in tqdm(range(n_forecasts), desc="Rolling Bayesian SV"):
        x_window = x_all[i:i + window_size]
        next_return = x_all[i + window_size]
        next_date = dates_all[i + window_size]

        out = adaptive_sv_mcmc(
            x=x_window,
            n_total=n_total,
            nu=nu,
            h_step=h_step,
            param_scale=param_scale,
            adapt_every=adapt_every,
            target_param_accept=target_param_accept,
            burn_adapt=burn_adapt,
            seed=int(rng.integers(0, 1_000_000_000)),
            store_h=False
        )

        mu_draws = out["mu"][burn:]
        phi_draws = out["phi"][burn:]
        sigma2_draws = out["sigma2"][burn:]
        h_last_draws = out["h_last"][burn:]

        sigma_next_draws = np.empty_like(mu_draws)

        for j in range(len(mu_draws)):
            mean_next = mu_draws[j] + phi_draws[j] * (h_last_draws[j] - mu_draws[j])
            h_next = mean_next + np.random.normal(0.0, np.sqrt(sigma2_draws[j]))
            sigma_next_draws[j] = np.exp(h_next / 2.0)

        pred_vol_mean = float(np.mean(sigma_next_draws))
        pred_vol_q025 = float(np.quantile(sigma_next_draws, 0.025))
        pred_vol_q975 = float(np.quantile(sigma_next_draws, 0.975))

        results.append({
            "Date": next_date,
            "PredVolMean": pred_vol_mean,
            "PredVolQ025": pred_vol_q025,
            "PredVolQ975": pred_vol_q975,
            "ActualReturn": next_return,
            "ActualAbsReturn": abs(next_return),
            "ActualSqReturn": next_return ** 2,
            "ParamAcceptRate": out["param_accept_rate"],
            "LatentAcceptRate": out["latent_accept_rate"],
            "PilotAcceptRate": out["pilot_accept_rate"],
        })

    return pd.DataFrame(results)


# =========================
# Plotting
# =========================
def plot_rolling_sv_forecast(results_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["Date"], results_df["PredVolMean"], label="Predicted SV volatility")
    plt.fill_between(
        results_df["Date"],
        results_df["PredVolQ025"],
        results_df["PredVolQ975"],
        alpha=0.25,
        label="95% posterior interval"
    )
    plt.plot(results_df["Date"], results_df["ActualAbsReturn"], label="Actual |return|")
    plt.title("Rolling Bayesian SV One-Step-Ahead Volatility Forecast")
    plt.xlabel("Date")
    plt.ylabel("Volatility / |Return|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(results_df["Date"], results_df["ParamAcceptRate"], label="Parameter acceptance")
    plt.plot(results_df["Date"], results_df["LatentAcceptRate"], label="Latent acceptance")
    plt.plot(results_df["Date"], results_df["PilotAcceptRate"], label="Pilot acceptance", alpha=0.8)
    plt.title("Acceptance Rates Across Rolling SV Windows")
    plt.xlabel("Date")
    plt.ylabel("Acceptance rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# Run
# =========================
if __name__ == "__main__":
    csv_path = "GSPC_clean.csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Cannot find {csv_path} in current directory.")

    df = load_returns_csv_with_dates(csv_path)

    results = rolling_bayesian_sv_forecast(
        df=df,
        years_back=5,
        window_size=100,
        burn=1000,
        n_total=3000,
        nu=8.0,
        h_step=1.0,
        param_scale=0.10,
        adapt_every=200,
        target_param_accept=0.25,
        burn_adapt=1500,
        seed=42
    )

    print(results.head())
    print("\nAverage parameter acceptance rate:", results["ParamAcceptRate"].mean())
    print("Average latent acceptance rate:", results["LatentAcceptRate"].mean())
    print("Average pilot acceptance rate:", results["PilotAcceptRate"].mean())

    plot_rolling_sv_forecast(results)

    results.to_csv("rolling_bayesian_sv_forecasts.csv", index=False)