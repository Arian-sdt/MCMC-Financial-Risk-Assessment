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
    x = x - x.mean()  # demean (recommended)
    return x

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
    # |d(omega,rho,delta)/d(u)| = omega * rho(1-rho) * delta(1-delta)
    omega, rho, delta, _, _ = transform_u_to_params(u)
    # numerical safety
    if omega <= 0 or rho <= 0 or rho >= 1 or delta <= 0 or delta >= 1:
        return -np.inf
    return (np.log(omega)
            + np.log(rho * (1.0 - rho))
            + np.log(delta * (1.0 - delta)))

# =========================
# Priors
# =========================
def log_gamma_pdf(x: float, shape: float, rate: float) -> float:
    # Gamma(shape=k, rate=beta)
    if x <= 0:
        return -np.inf
    return shape*np.log(rate) - lgamma(shape) + (shape-1.0)*np.log(x) - rate*x

def log_beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0 or x >= 1:
        return -np.inf
    return (lgamma(a+b) - lgamma(a) - lgamma(b)
            + (a-1.0)*np.log(x) + (b-1.0)*np.log(1.0-x))

def make_log_prior(var_x: float):
    """
    Tune omega prior rate to the return scale.
    Your var ~ 1.46e-4. A typical omega is ~ few e-6.
    We set E[omega]=1e-5 via Gamma(2, rate=2e5) by default.
    """
    omega_shape = 2.0
    omega_rate  = 200_000.0  # mean = 1e-5
    rho_a, rho_b = 2.0, 2.0
    delta_a, delta_b = 2.0, 2.0

    def log_prior(omega: float, rho: float, delta: float) -> float:
        lp = 0.0
        lp += log_gamma_pdf(omega, shape=omega_shape, rate=omega_rate)
        lp += log_beta_pdf(rho, a=rho_a, b=rho_b)
        lp += log_beta_pdf(delta, a=delta_a, b=delta_b)
        return lp

    return log_prior, dict(
        omega_shape=omega_shape, omega_rate=omega_rate,
        rho_a=rho_a, rho_b=rho_b, delta_a=delta_a, delta_b=delta_b
    )

# =========================
# GARCH(1,1) Gaussian log-likelihood
# =========================
def garch_loglik_gaussian(x: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    T = len(x)
    if omega <= 0 or alpha <= 0 or beta <= 0 or (alpha + beta) >= 1.0:
        return -np.inf

    # initialize sigma^2 with unconditional variance if possible
    unc = omega / (1.0 - alpha - beta)
    sig2 = unc if (np.isfinite(unc) and unc > 0) else float(np.var(x))

    ll = 0.0
    for t in range(T):
        if t > 0:
            sig2 = omega + alpha * (x[t-1] ** 2) + beta * sig2

        if not np.isfinite(sig2) or sig2 <= 1e-14:
            return -np.inf

        ll += -0.5 * (np.log(2.0*np.pi) + np.log(sig2) + (x[t]**2)/sig2)
    return ll

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
    # z ~ N(0, scale)
    L = np.linalg.cholesky(scale)
    z = L @ rng.normal(size=p)
    # s ~ Chi2(df)/df
    s = rng.chisquare(df) / df
    return mean + z / np.sqrt(s)

def mv_student_t_logpdf(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, df: float) -> float:
    p = len(mean)
    L = np.linalg.cholesky(scale)
    diff = x - mean
    y = np.linalg.solve(L, diff)
    quad = float(y @ y)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    c = (lgamma((df + p)/2.0) - lgamma(df/2.0)
         - 0.5*logdet - (p/2.0)*np.log(df*np.pi))
    return c - ((df + p)/2.0)*np.log(1.0 + quad/df)

# =========================
# Pilot Random-Walk Metropolis (for M, Sigma)
# =========================
def pilot_rw_metropolis(x: np.ndarray, log_post, n_burn=3000, n_keep=1000, step=0.05, seed=1):
    rng = np.random.default_rng(seed)

    # sensible initialization based on your scale
    # var ~ 1.46e-4, take omega0 ~ 7e-6, rho0 ~ 0.95, delta0 ~ 0.2
    omega0 = 7e-6
    rho0 = 0.95
    delta0 = 0.2
    u = np.array([
        np.log(omega0),
        np.log(rho0 / (1.0 - rho0)),
        np.log(delta0 / (1.0 - delta0))
    ], dtype=float)

    lp = log_post(u)
    cov = (step**2) * np.eye(3)
    kept = []

    accepted = 0
    total = n_burn + n_keep

    for i in tqdm(range(total), desc="Pilot RW"):
        u_star = u + rng.multivariate_normal(np.zeros(3), cov)
        lp_star = log_post(u_star)

        if np.log(rng.uniform()) < (lp_star - lp):
            u, lp = u_star, lp_star
            accepted += 1

        if i >= n_burn:
            kept.append(u.copy())

    kept = np.asarray(kept)
    M = kept.mean(axis=0)
    S = np.cov(kept.T) + 1e-6*np.eye(3)  # jitter

    return u, lp, M, S, accepted/total

# =========================
# Adaptive Independent MH
# =========================
def adaptive_independent_mh(
    x: np.ndarray,
    n_total=50_000,
    df=10,
    adapt_every=1000,
    seed=2
):
    log_post, prior_hypers = make_log_posterior_u(x)
    rng = np.random.default_rng(seed)

    # Pilot
    u, lp, M, S, pilot_acc = pilot_rw_metropolis(x, log_post, n_burn=3000, n_keep=1000, step=0.05, seed=seed)

    samples_u = np.empty((n_total, 3), dtype=float)
    accept = 0
    adapt_buf = []

    for t in tqdm(range(n_total), desc="Adaptive MH"):
        u_star = mv_student_t_rvs(M, S, df, rng)
        lp_star = log_post(u_star)

        if np.isfinite(lp_star):
            logq_curr = mv_student_t_logpdf(u,      M, S, df)
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
            S = np.cov(arr.T) + 1e-6*np.eye(3)

    # Convert draws to parameters
    omega = np.empty(n_total)
    rho   = np.empty(n_total)
    delta = np.empty(n_total)
    alpha = np.empty(n_total)
    beta  = np.empty(n_total)

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

    out = adaptive_independent_mh(
        x,
        n_total=50_000,     # start smaller; increase to 200k later
        df=10,              # Student-t df
        adapt_every=1000,
        seed=42
    )

    print("\nPrior hyperparameters used:")
    print(out["prior_hypers"])

    print("\nAcceptance rates:")
    print("  pilot RW:", out["pilot_accept_rate"])
    print("  main MH :", out["accept_rate"])

    # Burn-in discard for summaries (adjust as you like)
    burn = 5_000
    print(f"\nPosterior summaries (after burn-in {burn}):")
    summarize(out["omega"][burn:], "omega")
    summarize(out["alpha"][burn:], "alpha")
    summarize(out["beta"][burn:],  "beta")
    summarize(out["rho"][burn:],   "rho (=alpha+beta)")
    summarize(out["delta"][burn:], "delta (=alpha/(alpha+beta))")