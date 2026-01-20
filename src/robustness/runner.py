# src/robustness/runner.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import inspect
import numpy as np
import pandas as pd

from robustness.perturbations import Perturbation


def _years(cols) -> np.ndarray:
    return np.array([int(c) for c in cols])


def _extract_gap(res: Dict[str, Any], Y: pd.DataFrame) -> Optional[pd.Series]:
    # 1) SDID provides "gap" explicitly (preferred)
    if "gap" in res and isinstance(res["gap"], pd.Series):
        return res["gap"]

    # 2) SCM provides "att_series" (treated - synthetic)
    if "att_series" in res and isinstance(res["att_series"], pd.Series):
        return res["att_series"]

    # 3) Fallback: y_treated - y_synth if present
    if "y_treated" in res and "y_synth" in res and isinstance(res["y_treated"], pd.Series) and isinstance(res["y_synth"], pd.Series):
        yt = res["y_treated"].to_numpy()
        ys = res["y_synth"].to_numpy()
        return pd.Series(yt - ys, index=res["y_synth"].index, name="Gap")

    # 4) Final fallback: compute from passed-in Y and y_synth
    if "y_synth" in res and isinstance(res["y_synth"], pd.Series):
        return pd.Series(Y.iloc[-1].to_numpy() - res["y_synth"].to_numpy(), index=res["y_synth"].index, name="Gap")

    return None


def _extract_unit_weights(res: Dict[str, Any]) -> Optional[pd.Series]:
    for k in ["weights", "omega", "unit_weights", "w"]:
        if k in res and isinstance(res[k], pd.Series):
            return res[k]
    return None


def _extract_time_weights(res: Dict[str, Any]) -> Optional[pd.Series]:
    for k in ["lambda", "lam", "time_weights"]:
        if k in res and isinstance(res[k], pd.Series):
            return res[k]
    return None


def _rmspe(g: np.ndarray) -> float:
    return float(np.sqrt(np.mean(g * g)))


def _post_mask(years: np.ndarray, t0: int, params: dict) -> np.ndarray:
    m = np.zeros_like(years, dtype=bool)
    m[t0:] = True
    if params.get("name") == "last_k":
        start_year = int(params["start_year"])
        m = m & (years >= start_year)
    return m


def _compute_metrics(
    method: str,
    res: Dict[str, Any],
    Y: pd.DataFrame,
    meta: dict,
    params: dict,
    prefer_exact_effect_key: bool,
) -> Tuple[float, float, float, float]:
    t0 = int(meta["treat_start_idx"])
    years = _years(Y.columns)

    gap = _extract_gap(res, Y)

    # If gap extraction failed, fall back to estimator-supplied metrics.
    if gap is None:
        if method == "scm":
            effect = float(res["att_avg"])
        else:
            effect = float(res["tau"])
        diag = res.get("diagnostics", {})
        return (
            effect,
            float(diag.get("rmspe_pre", np.nan)),
            float(diag.get("rmspe_post", np.nan)),
            float(diag.get("rmspe_ratio", np.nan)),
        )

    g = np.asarray(gap, dtype=float).reshape(-1)
    g_pre = g[:t0]
    m_post = _post_mask(years, t0, params)
    g_post = g[m_post]

    pre = _rmspe(g_pre)
    post = _rmspe(g_post)
    ratio = float(post / (pre + 1e-12))

    if method == "sdid":
        lam = _extract_time_weights(res)
        if lam is None:
            effect = float(np.mean(g_post))
        else:
            intercept = float(np.sum(lam.to_numpy() * g_pre))
            effect = float(np.mean(g_post) - intercept)
        if prefer_exact_effect_key and params.get("name") == "full_post" and "tau" in res:
            effect = float(res["tau"])
    else:
        effect = float(np.mean(g_post))
        if prefer_exact_effect_key and params.get("name") == "full_post" and "att_avg" in res:
            effect = float(res["att_avg"])

    return effect, pre, post, ratio


def _call_estimator(method: str, fit_fn: callable, Y: pd.DataFrame, meta: dict, params: dict, defaults: dict) -> Dict[str, Any]:
    if method == "sdid":
        z = float(params.get("zeta", defaults.get("sdid_zeta", 1.0)))
        e = float(params.get("eta", defaults.get("sdid_eta", 0.1)))

        sig = inspect.signature(fit_fn)
        if "zeta" in sig.parameters and "eta" in sig.parameters:
            return fit_fn(Y, meta, z, e)
        return fit_fn(Y, meta)

    return fit_fn(Y, meta)


def _row(
    method: str,
    ptype: str,
    pid: str,
    effect: float,
    base_effect: float,
    pre: float,
    post: float,
    ratio: float,
    unit_w: Optional[pd.Series],
    time_w: Optional[pd.Series],
    params: dict,
) -> Dict[str, Any]:
    max_w = float(unit_w.max()) if unit_w is not None and len(unit_w) else np.nan
    eff_d = float(1.0 / np.sum(np.square(unit_w.to_numpy()))) if unit_w is not None and len(unit_w) else np.nan

    max_l = float(time_w.max()) if time_w is not None and len(time_w) else np.nan
    eff_y = float(1.0 / np.sum(np.square(time_w.to_numpy()))) if time_w is not None and len(time_w) else np.nan

    return {
        "method": method,
        "perturbation_type": ptype,
        "perturbation_id": pid,
        "effect": float(effect),
        "effect_delta": float(effect - base_effect),
        "rmspe_pre": float(pre),
        "rmspe_post": float(post),
        "rmspe_ratio": float(ratio),
        "max_weight": max_w,
        "eff_donors": eff_d,
        "max_lambda": max_l,
        "eff_years": eff_y,
        "params": params,
    }


def run_robustness(
    Y: pd.DataFrame,
    meta: dict,
    estimators: Dict[str, callable],
    config: dict,
) -> pd.DataFrame:
    defaults = {
        "sdid_zeta": float(config.get("sdid_zeta", 1.0)),
        "sdid_eta": float(config.get("sdid_eta", 0.1)),
    }

    rows: List[Dict[str, Any]] = []
    baselines: Dict[str, Dict[str, Any]] = {}

    # Baselines
    for method, fit_fn in estimators.items():
        res0 = _call_estimator(method, fit_fn, Y, meta, params={}, defaults=defaults)
        eff0, pre0, post0, ratio0 = _compute_metrics(
            method, res0, Y, meta, params={"name": "full_post"}, prefer_exact_effect_key=True
        )
        unit_w0 = _extract_unit_weights(res0)
        time_w0 = _extract_time_weights(res0)

        baselines[method] = {"effect": float(eff0), "unit_w": unit_w0, "time_w": time_w0}

        rows.append(_row(
            method, "baseline", "baseline",
            eff0, eff0, pre0, post0, ratio0,
            unit_w0, time_w0,
            params={"name": "full_post"},
        ))

    from robustness.perturbations import (
        iter_leave_one_out_donors,
        iter_drop_topk_donors,
        iter_time_windows,
        iter_sdid_hyperparams,
    )

    per_method: Dict[str, List[Perturbation]] = {m: [] for m in estimators.keys()}

    # LOO for all methods
    loo = iter_leave_one_out_donors(Y, meta)
    for m in per_method:
        per_method[m].extend(loo)

    # Drop top-k for all methods (if baseline unit weights exist)
    for m in per_method:
        w0 = baselines[m]["unit_w"]
        if w0 is None:
            continue
        per_method[m].extend(iter_drop_topk_donors(
            Y, meta, baseline_weights=w0, ks=tuple(config.get("drop_topk", (1, 2, 3)))
        ))

    # Window sensitivity for all methods
    win = iter_time_windows(
        Y, meta,
        pre_start_years=tuple(config.get("pre_start_years", (1970, 1975, 1980))),
        post_aggs=config.get("post_aggs", [{"name": "full_post"}, {"name": "last_k", "start_year": 1995}]),
    )
    for m in per_method:
        per_method[m].extend(win)

    # Hyperparams: SDID only
    if "sdid" in per_method:
        per_method["sdid"].extend(iter_sdid_hyperparams(
            Y, meta,
            zetas=tuple(config.get("zeta_grid", (0.01, 0.1, 1.0, 10.0, 100.0))),
            etas=tuple(config.get("eta_grid", (0.001, 0.01, 0.1, 1.0, 10.0))),
        ))

    # Run perturbations
    for method, perts in per_method.items():
        fit_fn = estimators[method]
        base_effect = baselines[method]["effect"]

        perts = sorted(perts, key=lambda p: (p.perturbation_type, p.perturbation_id))
        for p in perts:
            res = _call_estimator(method, fit_fn, p.Y, p.meta, p.params, defaults=defaults)
            eff, pre, post, ratio = _compute_metrics(
                method, res, p.Y, p.meta, params=p.params, prefer_exact_effect_key=False
            )
            unit_w = _extract_unit_weights(res)
            time_w = _extract_time_weights(res)

            rows.append(_row(
                method, p.perturbation_type, p.perturbation_id,
                eff, base_effect, pre, post, ratio,
                unit_w, time_w,
                params=p.params,
            ))

    return pd.DataFrame(rows)


def summarize_robustness(df: pd.DataFrame, baseline_key: str = "baseline") -> dict:
    out: Dict[str, Any] = {}

    def _sign(x: float) -> int:
        return 0 if abs(x) < 1e-12 else (1 if x > 0 else -1)

    for method in sorted(df["method"].unique()):
        d = df[df["method"] == method].copy()
        base = d[(d["perturbation_type"] == "baseline") & (d["perturbation_id"] == baseline_key)]
        if base.empty:
            continue

        base_eff = float(base["effect"].iloc[0])
        base_sign = _sign(base_eff)

        loo = d[d["perturbation_type"] == "loo_donor"]
        same_sign = float(np.mean([_sign(x) == base_sign for x in loo["effect"].to_numpy()])) if len(loo) else np.nan
        med_abs_delta = float(np.median(np.abs(loo["effect_delta"].to_numpy()))) if len(loo) else np.nan

        win = d[d["perturbation_type"] == "window"]
        win_min = float(win["effect"].min()) if len(win) else np.nan
        win_max = float(win["effect"].max()) if len(win) else np.nan

        hyp = d[d["perturbation_type"] == "hyperparam"]
        hyp_min = float(hyp["effect"].min()) if len(hyp) else np.nan
        hyp_max = float(hyp["effect"].max()) if len(hyp) else np.nan

        out[method] = {
            "baseline_effect": base_eff,
            "baseline_sign": int(base_sign),
            "loo_sign_stability": same_sign,
            "loo_median_abs_delta": med_abs_delta,
            "window_effect_min": win_min,
            "window_effect_max": win_max,
            "hyperparam_effect_min": hyp_min,
            "hyperparam_effect_max": hyp_max,
            "baseline_max_weight": float(base["max_weight"].iloc[0]) if "max_weight" in base else np.nan,
            "baseline_eff_donors": float(base["eff_donors"].iloc[0]) if "eff_donors" in base else np.nan,
            "baseline_max_lambda": float(base["max_lambda"].iloc[0]) if "max_lambda" in base else np.nan,
            "baseline_eff_years": float(base["eff_years"].iloc[0]) if "eff_years" in base else np.nan,
            "flags": {
                "loo_sign_stable_ge_90pct": bool(same_sign >= 0.90) if not np.isnan(same_sign) else False,
                "no_sign_flip_in_windows": bool(_sign(win_min) == base_sign and _sign(win_max) == base_sign) if len(win) else False,
            },
        }

    return out
