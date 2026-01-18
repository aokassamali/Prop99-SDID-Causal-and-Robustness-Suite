import pandas as pd
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from metrics.diagnostics import rmspe, weight_stats


def fit_scm(Y: pd.DataFrame, meta: dict) -> dict:

    T_pre_idx = meta['treat_start_idx']
    years = Y.columns.astype(int)
    
    treated_row = Y.iloc[-1]
    donor_df = Y.iloc[:-1]
    
    y1_pre = treated_row.iloc[:T_pre_idx].to_numpy()
    Y0_pre = donor_df.iloc[:, :T_pre_idx].to_numpy()
    
    n_donors = len(donor_df)
    w_init = np.full(n_donors, 1/n_donors)
    
    res = minimize(
        fun=lambda w: np.mean((y1_pre - (w @ Y0_pre))**2),
        x0=w_init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    )
    
    w_star = res.x

    y_synth_val = w_star @ donor_df.to_numpy()
    
    att_val = treated_row.to_numpy() - y_synth_val

    diag_stats = {
            "rmspe_pre": rmspe(y1_pre, y_synth_val[:T_pre_idx]),
            "rmspe_post": rmspe(treated_row.iloc[T_pre_idx:], y_synth_val[T_pre_idx:]),
            **weight_stats(w_star)  # Merge the weight dictionary here
        }
    
    diag_stats["rmspe_ratio"] = diag_stats["rmspe_post"] / diag_stats["rmspe_pre"]
    
    return {
            "weights": pd.Series(w_star, index=donor_df.index, name="SCM_Weights"),
            "y_synth": pd.Series(y_synth_val, index=years, name="Synthetic"),
            "y_treated" : treated_row,
            "att_series": pd.Series(att_val, index=years, name="ATT"),
            "att_avg": np.mean(att_val[T_pre_idx:]),
            "diagnostics": diag_stats
        }