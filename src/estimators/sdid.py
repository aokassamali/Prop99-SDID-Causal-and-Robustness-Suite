import numpy as np
import pandas as pd
from scipy.optimize import minimize

def solve_simplex_ridge(A: np.ndarray, b: np.ndarray, l2: float) -> np.ndarray:
    n_features = A.shape[1]

    x0 = np.full(n_features, 1 / n_features)

    def objective(x):
        r = A @ x - b
        return np.mean(r * r) + (l2 * np.sum(x * x))
    
    constraints = [{"type" : "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_features)]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options= {'ftol': 1e-9, 'disp': False}
    )

    return res.x


def fit_sdid(Y: pd.DataFrame, meta: dict, zeta: float = 1.0, eta: float = 0.1) -> dict:
    t0 = meta['treat_start_idx']
    
    Y1 = Y.iloc[-1, :]
    Y0 = Y.iloc[:-1, :]
    
    y1_pre = Y1.iloc[:t0].to_numpy()
    y1_post = Y1.iloc[t0:].to_numpy()
    
    Y0_pre = Y0.iloc[:, :t0].to_numpy()
    Y0_post = Y0.iloc[:, t0:].to_numpy()

    y1_pre_centered = y1_pre - y1_pre.mean()
    Y0_pre_centered = Y0_pre - Y0_pre.mean(axis=1, keepdims=True)

    omega = solve_simplex_ridge(Y0_pre_centered.T, y1_pre_centered, l2=zeta)

    target_post_mean = Y0_post.mean(axis=1)
    target_centered = target_post_mean - target_post_mean.mean()

    Y0_pre_time_centered = Y0_pre - Y0_pre.mean(axis=0, keepdims=True)

    lam = solve_simplex_ridge(Y0_pre_time_centered, target_centered, l2=eta)

    y_synth = omega @ Y0.to_numpy()

    g_t = Y1.to_numpy() - y_synth

    g_pre = g_t[:t0]
    g_post = g_t[t0:]

    tau_hat = np.mean(g_post) - np.sum(lam * g_pre)

    return {
        "tau": float(tau_hat),
        
        # Unit Weights (mapped to State Names)
        "omega": pd.Series(omega, index=Y0.index, name="Unit_Weights"),
        
        # Time Weights (mapped to Years)
        "lambda": pd.Series(lam, index=Y.columns[:t0], name="Time_Weights"),

        #Synthetic Series
        "y_synth": pd.Series(y_synth, index=Y.columns, name="Synthetic"),
        "gap": pd.Series(g_t, index=Y.columns, name = "Gap"),
        "g_pre": pd.Series(g_pre, index=Y.columns[:t0], name="Gap_Pre"),
        "g_post": pd.Series(g_post, index=Y.columns[t0:], name="Gap_Post"),

        # Scalar Metrics
        "diagnostics": {
            "omega_effective": 1.0 / np.sum(omega**2),
            "lambda_effective": 1.0 / np.sum(lam**2),
            "omega_max": np.max(omega),
            "lambda_entropy": -np.sum(lam * np.log(lam + 1e-9))
        }
    }
