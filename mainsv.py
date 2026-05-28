import numpy as np
import pandas as pd
from math import lgamma
from pathlib import Path
from tqdm import tqdm

# =========================
# Data loading / preprocessing
# =========================
def load_returns_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "LogReturn" not in df.columns:
        raise ValueError("CSV must contain a 'LogReturn' column.")
    x = df["LogReturn"].astype(float).to_numpy()
    x = x[np.isfinite(x)]
    x = x - x.mean()  # demean
    return x


# =========================
# Basic utilities
# =========================
def logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def safe_log(x: float) -> float:
    if x <= 0 or not np.isfinite(x):
        return -np.inf
    return np.log(x)


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
    """
    |d(mu,phi,sigma2)/d(u1,u2,u3)| = 1 * phi(1-phi) * sigma2
    """
    mu, phi, sigma2 = transform_u_to_params(u)
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
    """
    Inverse-Gamma(a,b) with density proportional to x^(-a-1) exp(-b/x)
    """
    if x <= 0:
        return -np.inf
    return a * np.log(b) - lgamma(a) - (a + 1.0) * np.log(x) - b / x


def make_log_prior():
    # weakly informative priors
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
        mu_mean=mu_mean, mu_var=mu_var,
        phi_a=phi_a, phi_b=phi_b,
        sigma2_a=sigma2_a, sigma2_b=sigma2_b
    )


# =========================
# Student-t observation density
# =========================
def student_t_logpdf_standard(z: float, nu: float) -> float:
    """
    Standard Student-t with df=nu, location 0, scale 1
    """
    if nu <= 0:
        return -np.inf
    return (
        lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - ((nu + 1.0) / 2.0) * np.log(1.0 + (z * z) / nu)
    )


def obs_loglik_t(x_t: float, h_t: float, nu: float) -> float:
    """
    x_t = exp(h_t/2) * eps_t, eps_t ~ t_nu(0,1)
    so log density = log t(z) - 0.5 h_t, where z = x_t / exp(h_t/2)
    """
    scale = np.exp(0.5 * h_t)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    z = x_t / scale
    return student_t_logpdf_standard(z, nu) - np.log(scale)


# =========================
# Latent AR(1) prior for h
# =========================
def log_h_prior(h: np.ndarray, mu: float, phi: float, sigma2: float) -> float:
    """
    h_1 ~ N(mu, sigma2 / (1-phi^2))
    h_t | h_{t-1} ~ N(mu + phi(h_{t-1}-mu), sigma2), t>=2
    """
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
    """
    Only uses terms involving h_t:
    - observation density at t
    - AR(1) prior terms involving h_t
    """
    T = len(h)

    val = obs_loglik_t(x[t], h_t_star, nu)
    if not np.isfinite(val):
        return -np.inf

    # prior contribution from h_t | h_{t-1}
    if t == 0:
        var0 = sigma2 / (1.0 - phi * phi)
        val += log_normal_pdf(h_t_star, mu, var0)
    else:
        mean_t = mu + phi * (h[t - 1] - mu)
        val += log_normal_pdf(h_t_star, mean_t, sigma2)

    # prior contribution from h_{t+1} | h_t
    if t < T - 1:
        mean_next = mu + phi * (h_t_star - mu)
        val += log_normal_pdf(h[t + 1], mean_next, sigma2)

    return val


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
# Pilot RW Metropolis for parameter block
# =========================
def pilot_rw_metropolis_sv(
    x: np.ndarray,
    h: np.ndarray,
    nu: float,
    n_burn=3000,
    n_keep=1000,
    step=0.05,
    seed=1
):
    rng = np.random.default_rng(seed)

    # sensible initialization
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

    for i in tqdm(range(total), desc="Pilot RW (SV params)"):
        u_star = u + rng.multivariate_normal(np.zeros(3), cov)
        lp_star = log_post(u_star)

        if np.log(rng.uniform()) < (lp_star - lp):
            u, lp = u_star, lp_star
            accepted += 1

        if i >= n_burn:
            kept.append(u.copy())

    kept = np.asarray(kept)
    M = kept.mean(axis=0)
    S = 0.005 * np.cov(kept.T) + 1e-6 * np.eye(3)

    return u, lp, M, S, accepted / total


# =========================
# Main SV sampler
# =========================
def adaptive_sv_mcmc(
    x: np.ndarray,
    n_total=20000,
    nu=8.0,                # fixed Student-t df
    h_step=1.5,           # RW std for latent states
    df_prop=10.0,          # Student-t proposal df for parameter block
    adapt_every=1000,
    seed=42
):
    rng = np.random.default_rng(seed)
    T = len(x)

    # initialize h with log-squared returns
    h = np.log(x * x + 1e-6)

    # pilot for parameter block
    u, lp, M, S, pilot_acc = pilot_rw_metropolis_sv(
        x, h, nu, n_burn=3000, n_keep=1000, step=0.05, seed=seed
    )

    log_prior, prior_hypers = make_log_prior()

    samples_u = np.empty((n_total, 3), dtype=float)
    samples_h_mean = np.empty(n_total, dtype=float)
    samples_h_last = np.empty(n_total, dtype=float)

    param_accept = 0
    latent_accept = 0
    latent_total = 0

    adapt_buf = []

    for it in tqdm(range(n_total), desc="Adaptive SV MCMC"):
        # ---------------------------------
        # 1. Update latent path h one site at a time
        # ---------------------------------
        mu, phi, sigma2 = transform_u_to_params(u)

        for t in range(T):
            current = h[t]
            star = current + rng.normal(0.0, h_step)

            log_curr = local_logpost_h_t(t, current, h, x, mu, phi, sigma2, nu)
            log_star = local_logpost_h_t(t, star,    h, x, mu, phi, sigma2, nu)

            if np.log(rng.uniform()) < (log_star - log_curr):
                h[t] = star
                latent_accept += 1

            latent_total += 1

        # ---------------------------------
        # 2. Update parameter block u using adaptive independent MH
        # ---------------------------------
        log_post_u, _ = make_log_posterior_u(x, h, nu)

        u_star = mv_student_t_rvs(M, S, df_prop, rng)
        lp_star = log_post_u(u_star)

        if np.isfinite(lp_star):
            logq_curr = mv_student_t_logpdf(u,      M, S, df_prop)
            logq_star = mv_student_t_logpdf(u_star, M, S, df_prop)
            loga = (lp_star - log_post_u(u)) + (logq_curr - logq_star)
        else:
            loga = -np.inf

        if np.log(rng.uniform()) < min(0.0, loga):
            u = u_star
            param_accept += 1

        # store
        samples_u[it] = u
        samples_h_mean[it] = float(np.mean(h))
        samples_h_last[it] = float(h[-1])
        adapt_buf.append(u.copy())

        # adapt proposal
        if (it + 1) % adapt_every == 0:
            arr = np.asarray(adapt_buf)
            M = arr.mean(axis=0)
            S = 0.005 * np.cov(arr.T) + 1e-6 * np.eye(3)

    # convert parameter draws
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
        "mu": mu_draws,
        "phi": phi_draws,
        "sigma2": sigma2_draws,
        "h_final": h.copy(),
        "h_mean_trace": samples_h_mean,
        "h_last_trace": samples_h_last,
        "pilot_accept_rate": pilot_acc,
        "param_accept_rate": param_accept / n_total,
        "latent_accept_rate": latent_accept / max(1, latent_total),
        "prior_hypers": prior_hypers,
        "nu_fixed": nu,
    }
    return out


# =========================
# Summaries
# =========================
def summarize(arr: np.ndarray, name: str):
    q2p5, q50, q97p5 = np.quantile(arr, [0.025, 0.5, 0.975])
    print(f"{name}: mean={arr.mean():.6g}, median={q50:.6g}, 95% CI=[{q2p5:.6g}, {q97p5:.6g}]")


# =========================
# Run
# =========================
if __name__ == "__main__":
    csv_path = "GSPC_clean.csv"  # change if needed
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Cannot find {csv_path} in current directory.")

    x = load_returns_csv(csv_path)

    print("Data stats (demeaned):")
    print("  mean:", float(x.mean()))
    print("  std :", float(x.std()))
    print("  var :", float(x.var()))

    out = adaptive_sv_mcmc(
        x,
        n_total=20000,
        nu=8.0,          # fixed Student-t df
        h_step=1.5,
        df_prop=10.0,
        adapt_every=1000,
        seed=42
    )

    print("\nPrior hyperparameters used:")
    print(out["prior_hypers"])
    print("Fixed nu:", out["nu_fixed"])

    print("\nAcceptance rates:")
    print("  pilot RW (params):", out["pilot_accept_rate"])
    print("  main MH  (params):", out["param_accept_rate"])
    print("  latent h updates  :", out["latent_accept_rate"])

    burn = 5000
    print(f"\nPosterior summaries (after burn-in {burn}):")
    summarize(out["mu"][burn:], "mu")
    summarize(out["phi"][burn:], "phi")
    summarize(out["sigma2"][burn:], "sigma2")