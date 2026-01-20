# src/robustness/placebos.py
from __future__ import annotations

import inspect
import numpy as np
import pandas as pd


def _rmspe(g: np.ndarray) -> float:
    return float(np.sqrt(np.mean(g * g)))


def _gap_from_res(res: dict, Y: pd.DataFrame) -> pd.Series | None:
    if "gap" in res and isinstance(res["gap"], pd.Series):
        return res["gap"]
    if "att_series" in res and isinstance(res["att_series"], pd.Series):
        return res["att_series"]
    if "y_treated" in res and "y_synth" in res and isinstance(res["y_treated"], pd.Series) and isinstance(res["y_synth"], pd.Series):
        yt = res["y_treated"].to_numpy()
        ys = res["y_synth"].to_numpy()
        return pd.Series(yt - ys, index=res["y_synth"].index, name="Gap")
    if "y_synth" in res and isinstance(res["y_synth"], pd.Series):
        return pd.Series(Y.iloc[-1].to_numpy() - res["y_synth"].to_numpy(), index=res["y_synth"].index, name="Gap")
    return None


def _unit_weights(res: dict) -> pd.Series | None:
    for k in ["weights", "omega"]:
        if k in res and isinstance(res[k], pd.Series):
            return res[k]
    return None


def _time_weights(res: dict) -> pd.Series | None:
    for k in ["lambda", "lam"]:
        if k in res and isinstance(res[k], pd.Series):
            return res[k]
    return None


def _call_fit(fit_fn: callable, Y: pd.DataFrame, meta: dict, sdid_zeta: float, sdid_eta: float) -> dict:
    sig = inspect.signature(fit_fn)
    if "zeta" in sig.parameters and "eta" in sig.parameters:
        return fit_fn(Y, meta, float(sdid_zeta), float(sdid_eta))
    return fit_fn(Y, meta)


def placebo_in_space(
    Y: pd.DataFrame,
    meta: dict,
    fit_fn: callable,
    effect_key: str = "att_avg",
    sdid_zeta: float = 1.0,
    sdid_eta: float = 0.1,
) -> pd.DataFrame:
    t0 = int(meta["treat_start_idx"])
    rows = []
    donors_only = Y.iloc[:-1]

    for u in donors_only.index:
        Yp = donors_only.drop(index=u)
        Yp = pd.concat([Yp, donors_only.loc[[u]]], axis=0)

        res = _call_fit(fit_fn, Yp, meta, sdid_zeta, sdid_eta)
        eff = float(res[effect_key])

        gap = _gap_from_res(res, Yp)
        if gap is not None:
            g = gap.to_numpy()
            g_pre = g[:t0]
            g_post = g[t0:]
            pre = _rmspe(g_pre)
            post = _rmspe(g_post)
            ratio = float(post / (pre + 1e-12))
        else:
            diag = res.get("diagnostics", {})
            pre = float(diag.get("rmspe_pre", np.nan))
            post = float(diag.get("rmspe_post", np.nan))
            ratio = float(diag.get("rmspe_ratio", np.nan))

        w = _unit_weights(res)
        max_w = float(w.max()) if w is not None and len(w) else np.nan
        eff_d = float(1.0 / np.sum(np.square(w.to_numpy()))) if w is not None and len(w) else np.nan

        lam = _time_weights(res)
        max_l = float(lam.max()) if lam is not None and len(lam) else np.nan
        eff_y = float(1.0 / np.sum(np.square(lam.to_numpy()))) if lam is not None and len(lam) else np.nan

        rows.append({
            "placebo_unit": u,
            effect_key: eff,
            "pre_rmspe": pre,
            "post_rmspe": post,
            "rmspe_ratio": ratio,
            "max_weight": max_w,
            "eff_donors": eff_d,
            "max_lambda": max_l,
            "eff_years": eff_y,
        })

    return pd.DataFrame(rows)


def placebo_in_time(
    Y: pd.DataFrame,
    meta: dict,
    fit_fn: callable,
    training_years: int,
    effect_key: str = "att_avg",
    sdid_zeta: float = 1.0,
    sdid_eta: float = 0.1,
) -> pd.DataFrame:
    base_t0 = int(meta["treat_start_idx"])
    Y_intime = Y.iloc[:, :base_t0]  # use ONLY real pre columns
    max_idx = Y_intime.shape[1]
    candidate_indices = range(int(training_years), max_idx - 1)

    rows = []
    for u in candidate_indices:
        meta_u = dict(meta)
        meta_u["treat_start_idx"] = int(u)
        fake_year = Y_intime.columns[u]

        res = _call_fit(fit_fn, Y_intime, meta_u, sdid_zeta, sdid_eta)
        eff = float(res[effect_key])

        gap = _gap_from_res(res, Y_intime)
        if gap is not None:
            g = gap.to_numpy()
            g_pre = g[:u]
            g_post = g[u:]
            pre = _rmspe(g_pre)
            post = _rmspe(g_post)
            ratio = float(post / (pre + 1e-12))
        else:
            diag = res.get("diagnostics", {})
            pre = float(diag.get("rmspe_pre", np.nan))
            post = float(diag.get("rmspe_post", np.nan))
            ratio = float(diag.get("rmspe_ratio", np.nan))

        w = _unit_weights(res)
        max_w = float(w.max()) if w is not None and len(w) else np.nan
        eff_d = float(1.0 / np.sum(np.square(w.to_numpy()))) if w is not None and len(w) else np.nan

        lam = _time_weights(res)
        max_l = float(lam.max()) if lam is not None and len(lam) else np.nan
        eff_y = float(1.0 / np.sum(np.square(lam.to_numpy()))) if lam is not None and len(lam) else np.nan

        rows.append({
            "treated_year": fake_year,
            effect_key: eff,
            "pre_rmspe": pre,
            "post_rmspe": post,
            "rmspe_ratio": ratio,
            "max_weight": max_w,
            "eff_donors": eff_d,
            "max_lambda": max_l,
            "eff_years": eff_y,
        })

    return pd.DataFrame(rows)
