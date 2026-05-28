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

    # demean full series once
    df["LogReturn"] = df["LogReturn"] - df["LogReturn"].mean()
    df = df.reset_index(drop=True)

    return df


# =========================
# Transforms
# =========================
def logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def transform_u_to_params(u: np.ndarray):
    """
    u = (u1,u2,u3) in R^3
    omega = exp(u1) > 0
    rho   = logistic(u2) in (0,1) = alpha+beta
    delta = logistic(u3) in (0,1) = alpha/(alpha+beta)
    """
    u1, u2, u3 = float(u[0]), float(u[1]), float(u[2])
    omega = np.exp(u1)
    rho = logistic(u2)
    delta = logistic(u3)
    alpha = rho * delta
    beta = rho * (1.0 - delta)
    return omega, rho, delta, alpha, beta


def log_jacobian(u: np.ndarray) -> float:
    omega, rho, delta, _, _ = transform_u_to_params(u)
    if omega <= 0 or rho <= 0 or rho >= 1 or delta <= 0 or delta >= 1:
        return -np.inf
    return (
        np.log(omega)
        + np.log(rho * (1.0 - rho))
        + np.log(delta * (1.0 - delta))
    )


# =========================
# Priors
# =========================
def log_gamma_pdf(x: float, shape: float, rate: float) -> float:
    if x <= 0:
        return -np.inf
    return shape * np.log(rate) - lgamma(shape) + (shape - 1.0) * np.log(x) - rate * x


def log_beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1:
        return -np.inf
    return (
        lgamma(a + b) - lgamma(a) - lgamma(b)
        + (a - 1.0) * np.log(x)
        + (b - 1.0) * np.log(1.0 - x)
    )


def make_log_prior(var_x: float):
    omega_shape = 2.0
    omega_rate = 200_000.0   # mean = 1e-5
    rho_a, rho_b = 2.0, 2.0
    delta_a, delta_b = 2.0, 2.0

    def log_prior(omega: float, rho: float, delta: float) -> float:
        lp = 0.0
        lp += log_gamma_pdf(omega, shape=omega_shape, rate=omega_rate)
        lp += log_beta_pdf(rho, a=rho_a, b=rho_b)
        lp += log_beta_pdf(delta, a=delta_a, b=delta_b)
        return lp

    return log_prior, dict(
        omega_shape=omega_shape,
        omega_rate=omega_rate,
        rho_a=rho_a,
        rho_b=rho_b,
        delta_a=delta_a,
        delta_b=delta_b,
    )


# =========================
# GARCH(1,1) Gaussian log-likelihood
# =========================
def garch_loglik_gaussian(x: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    T = len(x)
    if omega <= 0 or alpha <= 0 or beta <= 0 or (alpha + beta) >= 1.0:
        return -np.inf

    unc = omega / (1.0 - alpha - beta)
    sig2 = unc if (np.isfinite(unc) and unc > 0) else float(np.var(x))

    ll = 0.0
    for t in range(T):
        if t > 0:
            sig2 = omega + alpha * (x[t - 1] ** 2) + beta * sig2

        if not np.isfinite(sig2) or sig2 <= 1e-14:
            return -np.inf

        ll += -0.5 * (np.log(2.0 * np.pi) + np.log(sig2) + (x[t] ** 2) / sig2)

    return ll


def last_filtered_variance(x: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    """
    Compute sigma_T^2 at the end of the window.
    """
    T = len(x)
    unc = omega / (1.0 - alpha - beta)
    sig2 = unc if (np.isfinite(unc) and unc > 0) else float(np.var(x))

    for t in range(T):
        if t > 0:
            sig2 = omega + alpha * (x[t - 1] ** 2) + beta * sig2

    return sig2


def forecast_next_variance(x_window: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    """
    One-step-ahead forecast:
    sigma_{T+1}^2 = omega + alpha * x_T^2 + beta * sigma_T^2
    """
    sig2_T = last_filtered_variance(x_window, omega, alpha, beta)
    sig2_next = omega + alpha * (x_window[-1] ** 2) + beta * sig2_T
    return sig2_next


# =========================
# Posterior in u-space
# =========================
def make_log_posterior_u(x: np.ndarray):
    var_x = float(np.var(x))
    log_prior, prior_hypers = make_log_prior(var_x)

    def log_post(u: np.ndarray) -> float:
        omega, rho, delta, alpha, beta = transform_u_to_params(u)
        ll = garch_loglik_gaussian(x, omega, alpha, beta)
        if not np.isfinite(ll):
            return -np.inf
        lp = log_prior(omega, rho, delta)
        if not np.isfinite(lp):
            return -np.inf
        lj = log_jacobian(u)
        if not np.isfinite(lj):
            return -np.inf
        return ll + lp + lj

    return log_post, prior_hypers


# =========================
# Multivariate Student-t proposal
# =========================
def mv_student_t_rvs(mean: np.ndarray, scale: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    p = len(mean)
    L = np.linalg.cholesky(scale)
    z = L @ rng.normal(size=p)
    s = rng.chisquare(df) / df
    return mean + z / np.sqrt(s)


def mv_student_t_logpdf(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, df: float) -> float:
    p = len(mean)
    L = np.linalg.cholesky(scale)
    diff = x - mean
    y = np.linalg.solve(L, diff)
    quad = float(y @ y)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    c = (
        lgamma((df + p) / 2.0)
        - lgamma(df / 2.0)
        - 0.5 * logdet
        - (p / 2.0) * np.log(df * np.pi)
    )
    return c - ((df + p) / 2.0) * np.log(1.0 + quad / df)


# =========================
# Pilot Random-Walk Metropolis
# =========================
def pilot_rw_metropolis(x: np.ndarray, log_post, n_burn=1000, n_keep=500, step=0.05, seed=1):
    rng = np.random.default_rng(seed)

    omega0 = 7e-6
    rho0 = 0.95
    delta0 = 0.2
    u = np.array([
        np.log(omega0),
        np.log(rho0 / (1.0 - rho0)),
        np.log(delta0 / (1.0 - delta0))
    ], dtype=float)

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
    M = kept.mean(axis=0)
    S = np.cov(kept.T) + 1e-6 * np.eye(3)

    return u, lp, M, S, accepted / total


# =========================
# Adaptive Independent MH
# =========================
def adaptive_independent_mh(
    x: np.ndarray,
    n_total=4000,
    df=10,
    adapt_every=500,
    seed=2
):
    log_post, prior_hypers = make_log_posterior_u(x)
    rng = np.random.default_rng(seed)

    u, lp, M, S, pilot_acc = pilot_rw_metropolis(
        x, log_post,
        n_burn=1000,
        n_keep=500,
        step=0.05,
        seed=seed
    )

    samples_u = np.empty((n_total, 3), dtype=float)
    accept = 0
    adapt_buf = []

    for t in range(n_total):
        u_star = mv_student_t_rvs(M, S, df, rng)
        lp_star = log_post(u_star)

        if np.isfinite(lp_star):
            logq_curr = mv_student_t_logpdf(u, M, S, df)
            logq_star = mv_student_t_logpdf(u_star, M, S, df)
            loga = (lp_star - lp) + (logq_curr - logq_star)
        else:
            loga = -np.inf

        if np.log(rng.uniform()) < min(0.0, loga):
            u, lp = u_star, lp_star
            accept += 1

        samples_u[t] = u
        adapt_buf.append(u.copy())

        if (t + 1) % adapt_every == 0:
            arr = np.asarray(adapt_buf)
            M = arr.mean(axis=0)
            S = np.cov(arr.T) + 1e-6 * np.eye(3)

    omega = np.empty(n_total)
    rho = np.empty(n_total)
    delta = np.empty(n_total)
    alpha = np.empty(n_total)
    beta = np.empty(n_total)

    for i in range(n_total):
        om, rh, de, a, b = transform_u_to_params(samples_u[i])
        omega[i], rho[i], delta[i], alpha[i], beta[i] = om, rh, de, a, b

    out = {
        "samples_u": samples_u,
        "omega": omega,
        "rho": rho,
        "delta": delta,
        "alpha": alpha,
        "beta": beta,
        "pilot_accept_rate": pilot_acc,
        "accept_rate": accept / n_total,
        "prior_hypers": prior_hypers,
    }
    return out


# =========================
# Rolling Bayesian forecast
# =========================
def rolling_bayesian_garch_forecast(
    df: pd.DataFrame,
    years_back=5,
    window_size=100,
    burn=1000,
    n_total=4000,
    adapt_every=500,
    df_prop=10,
    seed=42
):
    # select last 5 years by calendar date
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

    for i in tqdm(range(n_forecasts), desc="Rolling Bayesian GARCH"):
        x_window = x_all[i:i + window_size]
        next_return = x_all[i + window_size]
        next_date = dates_all[i + window_size]

        out = adaptive_independent_mh(
            x_window,
            n_total=n_total,
            df=df_prop,
            adapt_every=adapt_every,
            seed=int(rng.integers(0, 1_000_000_000))
        )

        omega_draws = out["omega"][burn:]
        alpha_draws = out["alpha"][burn:]
        beta_draws = out["beta"][burn:]

        sigma_next_draws = np.empty_like(omega_draws)

        for j in range(len(omega_draws)):
            sig2_next = forecast_next_variance(
                x_window,
                omega_draws[j],
                alpha_draws[j],
                beta_draws[j]
            )
            sigma_next_draws[j] = np.sqrt(sig2_next)

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
            "WindowAcceptRate": out["accept_rate"],
            "PilotAcceptRate": out["pilot_accept_rate"],
        })

    return pd.DataFrame(results)


# =========================
# Plotting
# =========================
def plot_rolling_forecast(results_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["Date"], results_df["PredVolMean"], label="Predicted volatility")
    plt.fill_between(
        results_df["Date"],
        results_df["PredVolQ025"],
        results_df["PredVolQ975"],
        alpha=0.25,
        label="95% posterior interval"
    )
    plt.plot(results_df["Date"], results_df["ActualAbsReturn"], label="Actual |return|")
    plt.title("Rolling Bayesian GARCH One-Step-Ahead Volatility Forecast")
    plt.xlabel("Date")
    plt.ylabel("Volatility / |Return|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(results_df["Date"], results_df["WindowAcceptRate"], label="Main MH acceptance")
    plt.plot(results_df["Date"], results_df["PilotAcceptRate"], label="Pilot RW acceptance", alpha=0.8)
    plt.title("Acceptance Rates Across Rolling Windows")
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

    results = rolling_bayesian_garch_forecast(
        df=df,
        years_back=5,
        window_size=100,
        burn=1000,
        n_total=4000,
        adapt_every=500,
        df_prop=10,
        seed=42
    )

    print(results.head())
    print("\nAverage main-window acceptance rate:", results["WindowAcceptRate"].mean())
    print("Average pilot acceptance rate:", results["PilotAcceptRate"].mean())

    plot_rolling_forecast(results)

    # optional save
    results.to_csv("rolling_bayesian_garch_forecasts.csv", index=False)