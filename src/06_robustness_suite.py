# scripts/04_robustness_suite.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tyro

from data.prop99 import load_json_parquet
from estimators.scm import fit_scm
from estimators.sdid import fit_sdid

from robustness.runner import run_robustness, summarize_robustness
from robustness.placebos import placebo_in_space, placebo_in_time


def _rank_and_pvals(effects: pd.Series, treated_effect: float) -> dict:
    abs_eff = float(abs(treated_effect))
    rank_abs = 1 + int((effects.abs() > abs_eff).sum())
    count_ge_abs = int((effects.abs() >= abs_eff).sum())
    p_two = float((1 + count_ge_abs) / (1 + len(effects)))
    count_le = int((effects <= float(treated_effect)).sum())  # one-sided negative
    p_one_neg = float((1 + count_le) / (1 + len(effects)))
    return {"n_placebos": int(len(effects)), "rank_abs": int(rank_abs), "p_two": p_two, "p_one_neg": p_one_neg}


def _prefit_filters(df: pd.DataFrame, treated_effect: float, treated_pre_rmspe: float) -> dict:
    out = {}

    df_elig = df[df["rmspe_pre"] <= treated_pre_rmspe]
    out["elig_pre_rmspe_le_treated"] = {
        "cutoff": float(treated_pre_rmspe),
        **_rank_and_pvals(df_elig["effect"], treated_effect),
    }

    cut = float(df["rmspe_pre"].quantile(0.50))
    df_p50 = df[df["rmspe_pre"] <= cut]
    out["bottom50_pre_rmspe"] = {"cutoff": cut, **_rank_and_pvals(df_p50["effect"], treated_effect)}

    return out


def _heatmap_sdid_hyperparams(df: pd.DataFrame, out_path: Path):
    hyp = df[(df["method"] == "sdid") & (df["perturbation_type"] == "hyperparam")].copy()
    if hyp.empty:
        return

    hyp["zeta"] = hyp["params_json"].apply(lambda s: json.loads(s).get("zeta", np.nan))
    hyp["eta"] = hyp["params_json"].apply(lambda s: json.loads(s).get("eta", np.nan))

    pivot = hyp.pivot_table(index="zeta", columns="eta", values="effect", aggfunc="mean")
    Z = pivot.to_numpy()
    zetas = pivot.index.to_numpy()
    etas = pivot.columns.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Z, aspect="auto", origin="lower")

    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels([str(e) for e in etas], rotation=45, ha="right")
    ax.set_yticks(range(len(zetas)))
    ax.set_yticklabels([str(z) for z in zetas])

    ax.set_xlabel("eta")
    ax.set_ylabel("zeta")
    ax.set_title("SDID robustness: effect over (zeta, eta)")
    fig.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main(
    data_dir: str = "data/processed",
    training_years: int = 10,
    plot_heatmap: bool = True,
    seed: int = 0,
):
    np.random.seed(seed)

    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    Y, meta = load_json_parquet(data_dir)

    estimators = {"scm": fit_scm, "sdid": fit_sdid}
    config = {
        "sdid_zeta": 1.0,
        "sdid_eta": 0.1,
        "drop_topk": (1, 2, 3),
        "pre_start_years": (1970, 1975, 1980),
        "post_aggs": [{"name": "full_post"}, {"name": "last_k", "start_year": 1995}],
        "zeta_grid": (0.01, 0.1, 1.0, 10.0, 100.0),
        "eta_grid": (0.001, 0.01, 0.1, 1.0, 10.0),
    }

    # Core robustness grid (baseline + perturbations)
    df = run_robustness(Y, meta, estimators=estimators, config=config)

    # Flatten params to JSON for CSV stability
    df = df.sort_values(["method", "perturbation_type", "perturbation_id"]).reset_index(drop=True)
    df["params_json"] = df["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    df = df.drop(columns=["params"])

    # Baseline effects + treated pre_rmspe for filtering
    base_scm_row = df[(df["method"] == "scm") & (df["perturbation_type"] == "baseline")].iloc[0]
    base_sdid_row = df[(df["method"] == "sdid") & (df["perturbation_type"] == "baseline")].iloc[0]
    base_scm = float(base_scm_row["effect"])
    base_sdid = float(base_sdid_row["effect"])

    treated_pre_scm = float(base_scm_row["rmspe_pre"])
    treated_pre_sdid = float(base_sdid_row["rmspe_pre"])

    # Generate placebo tables (SCM uses att_avg; SDID uses tau)
    ps_scm = placebo_in_space(Y, meta, fit_scm, effect_key="att_avg")
    pt_scm = placebo_in_time(Y, meta, fit_scm, training_years=training_years, effect_key="att_avg")

    ps_sdid = placebo_in_space(Y, meta, fit_sdid, effect_key="tau", sdid_zeta=1.0, sdid_eta=0.1)
    pt_sdid = placebo_in_time(Y, meta, fit_sdid, training_years=training_years, effect_key="tau", sdid_zeta=1.0, sdid_eta=0.1)

    def _to_rows(method: str, ptype: str, dfp: pd.DataFrame, treated_effect: float) -> pd.DataFrame:
        if method == "scm":
            effcol = "att_avg"
        else:
            effcol = "tau"
        out = dfp.rename(columns={
            effcol: "effect",
            "pre_rmspe": "rmspe_pre",
            "post_rmspe": "rmspe_post",
        }).copy()

        out["method"] = method
        out["perturbation_type"] = ptype
        out["perturbation_id"] = out["placebo_unit"].astype(str) if "placebo_unit" in out.columns else out["treated_year"].astype(str)
        out["effect_delta"] = out["effect"] - treated_effect
        out["params_json"] = ""

        # Ensure all weight columns exist
        for c in ["max_weight", "eff_donors", "max_lambda", "eff_years"]:
            if c not in out.columns:
                out[c] = np.nan

        keep = [
            "method","perturbation_type","perturbation_id",
            "effect","effect_delta","rmspe_pre","rmspe_post","rmspe_ratio",
            "max_weight","eff_donors","max_lambda","eff_years",
            "params_json",
        ]
        extras = [c for c in ["placebo_unit","treated_year"] if c in out.columns]
        return out[keep + extras]

    df_placebos = pd.concat([
        _to_rows("scm", "placebo_space", ps_scm, base_scm),
        _to_rows("scm", "placebo_time", pt_scm, base_scm),
        _to_rows("sdid", "placebo_space", ps_sdid, base_sdid),
        _to_rows("sdid", "placebo_time", pt_sdid, base_sdid),
    ], ignore_index=True)

    df_full = pd.concat([df, df_placebos], ignore_index=True)
    df_full = df_full.sort_values(["method", "perturbation_type", "perturbation_id"]).reset_index(drop=True)

    grid_path = report_dir / "tables" / "robustness_grid.csv"
    df_full.to_csv(grid_path, index=False)

    # Summary flags + main_results.json
    summary = summarize_robustness(df_full, baseline_key="baseline")
    summary_path = report_dir / "tables" / "robustness_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    def _placebo_pvals(method: str, treated_effect: float, treated_pre_rmspe: float) -> dict:
        space = df_full[(df_full["method"] == method) & (df_full["perturbation_type"] == "placebo_space")]
        out = {"space_unfiltered": _rank_and_pvals(space["effect"], treated_effect)}
        out["space_filtered"] = _prefit_filters(space, treated_effect, treated_pre_rmspe)

        time = df_full[(df_full["method"] == method) & (df_full["perturbation_type"] == "placebo_time")]
        out["time_recap"] = {
            "n": int(len(time)),
            "mean": float(time["effect"].mean()) if len(time) else np.nan,
            "median": float(time["effect"].median()) if len(time) else np.nan,
            "most_extreme_abs": float(time["effect"].abs().max()) if len(time) else np.nan,
        }
        return out

    placebo_summary = {
        "scm": _placebo_pvals("scm", base_scm, treated_pre_scm),
        "sdid": _placebo_pvals("sdid", base_sdid, treated_pre_sdid),
    }

    main_results = {
        "baseline": {"scm_att": base_scm, "sdid_tau": base_sdid, "difference": float(base_sdid - base_scm)},
        "placebos": placebo_summary,
        "robustness_highlights": summary,
    }
    main_path = report_dir / "tables" / "main_results.json"
    with open(main_path, "w") as f:
        json.dump(main_results, f, indent=2)

    if plot_heatmap:
        _heatmap_sdid_hyperparams(df_full, report_dir / "figures" / "robustness_heatmap.png")

    print("Wrote:")
    print(" -", grid_path)
    print(" -", summary_path)
    print(" -", main_path)


if __name__ == "__main__":
    tyro.cli(main)
