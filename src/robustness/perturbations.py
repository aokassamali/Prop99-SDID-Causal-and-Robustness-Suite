# src/robustness/perturbations.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Perturbation:
    perturbation_type: str
    perturbation_id: str
    Y: pd.DataFrame
    meta: dict
    params: dict


def _years(cols) -> np.ndarray:
    return np.array([int(c) for c in cols])


def iter_leave_one_out_donors(Y: pd.DataFrame, meta: dict) -> List[Perturbation]:
    donors = list(Y.index[:-1])  # treated is last
    out: List[Perturbation] = []
    for d in donors:
        out.append(Perturbation(
            perturbation_type="loo_donor",
            perturbation_id=f"donor={d}",
            Y=Y.drop(index=d),
            meta=meta,
            params={"dropped_units": [d]},
        ))
    return out


def iter_drop_topk_donors(
    Y: pd.DataFrame,
    meta: dict,
    baseline_weights: pd.Series,
    ks=(1, 2, 3),
) -> List[Perturbation]:
    w = baseline_weights.dropna().sort_values(ascending=False)
    out: List[Perturbation] = []
    for k in ks:
        dropped = list(w.index[:k])
        if not dropped:
            continue
        # Only drop donors (never drop the treated-last row)
        drop_idx = [u for u in dropped if u in Y.index[:-1]]
        if not drop_idx:
            continue
        out.append(Perturbation(
            perturbation_type="drop_topk",
            perturbation_id=f"k={int(k)}",
            Y=Y.drop(index=drop_idx),
            meta=meta,
            params={"k": int(k), "dropped_units": dropped},
        ))
    return out


def iter_time_windows(
    Y: pd.DataFrame,
    meta: dict,
    pre_start_years=(1970, 1975, 1980),
    post_aggs=None,
) -> List[Perturbation]:
    # If meta lacks treat_start_year, infer it from original columns + treat_start_idx
    base_t0 = int(meta["treat_start_idx"])
    treat_year = int(_years(Y.columns)[base_t0])

    if post_aggs is None:
        post_aggs = [{"name": "full_post"}, {"name": "last_k", "start_year": 1995}]

    out: List[Perturbation] = []
    years = _years(Y.columns)

    for pre0 in pre_start_years:
        mask = years >= int(pre0)
        Yp = Y.loc[:, mask]
        years_p = _years(Yp.columns)

        if treat_year not in set(years_p):
            continue

        meta_p = dict(meta)
        meta_p["treat_start_idx"] = int(np.where(years_p == treat_year)[0][0])

        for pa in post_aggs:
            pid = f"pre_start={int(pre0)}|post={pa['name']}"
            if pa.get("name") == "last_k":
                pid += f"|start={int(pa['start_year'])}"
            out.append(Perturbation(
                perturbation_type="window",
                perturbation_id=pid,
                Y=Yp,
                meta=meta_p,
                params={"pre_start_year": int(pre0), **pa},
            ))

    return out


def iter_sdid_hyperparams(
    Y: pd.DataFrame,
    meta: dict,
    zetas=(0.01, 0.1, 1.0, 10.0, 100.0),
    etas=(0.001, 0.01, 0.1, 1.0, 10.0),
) -> List[Perturbation]:
    out: List[Perturbation] = []
    for z in zetas:
        for e in etas:
            out.append(Perturbation(
                perturbation_type="hyperparam",
                perturbation_id=f"zeta={z}|eta={e}",
                Y=Y,
                meta=meta,
                params={"zeta": float(z), "eta": float(e)},
            ))
    return out
