from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tyro

from estimators.sdid import fit_sdid
from data.prop99 import load_json_parquet
from robustness.placebos import placebo_in_space, placebo_in_time


def _compute_actual_metrics(Y: pd.DataFrame, meta: dict) -> dict:
    t0 = meta["treat_start_idx"]
    actual = fit_sdid(Y, meta)  
    tau = float(actual["tau"])
    gap = actual["gap"]

    pre = gap.iloc[:t0].to_numpy()
    post = gap.iloc[t0:].to_numpy()

    pre_rmspe = float(np.sqrt(np.mean(pre**2)))
    post_rmspe = float(np.sqrt(np.mean(post**2)))
    ratio = float(post_rmspe / (pre_rmspe + 1e-12))

    return {
        "tau": tau,
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "rmspe_ratio": ratio,
    }


def _rank_and_pvals(df: pd.DataFrame, tau_treated: float) -> dict:
    # rank_abs = 1 + sum(|tau_placebo| > |tau_treated|)
    abs_tau = float(abs(tau_treated))
    rank_abs = 1 + int((df["tau"].abs() > abs_tau).sum())

    # p_two = (1 + count_ge_abs) / (1 + n_placebos)
    count_ge_abs = int((df["tau"].abs() >= abs_tau).sum())
    p_two = float((1 + count_ge_abs) / (1 + len(df)))

    # p_one_neg = (1 + count_le) / (1 + n_placebos)  (one-sided: tau <= tau_treated)
    count_le = int((df["tau"] <= float(tau_treated)).sum())
    p_one_neg = float((1 + count_le) / (1 + len(df)))

    return {
        "n_placebos": int(len(df)),
        "rank_abs": int(rank_abs),
        "p_two": p_two,
        "p_one_neg": p_one_neg,
    }


def _filtered_views(df: pd.DataFrame, pre_rmspe_treated: float) -> dict:
    views = {}

    df_a = df[df["pre_rmspe"] <= pre_rmspe_treated].copy()
    views["elig_pre_rmspe_le_treated"] = {"df": df_a, "cutoff": float(pre_rmspe_treated)}

    cut = float(df["pre_rmspe"].quantile(0.50))
    df_b = df[df["pre_rmspe"] <= cut].copy()
    views["bottom50_pre_rmspe"] = {"df": df_b, "cutoff": cut}

    return views


def _maybe_save_tau_hist(df: pd.DataFrame, tau_treated: float, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["tau"].to_numpy(), bins=20, alpha=0.7, color="gray")
    ax.axvline(float(tau_treated), color="red", linestyle="--", linewidth=2, label=f"treated tau={tau_treated:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Placebo tau")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main(
    data_dir: str = "data/processed",
    training_years: int = 10,     # for in-time placebos
    plot: bool = False,           # optional histograms
):
    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)


    Y, meta = load_json_parquet(data_dir)

    # Compute treated-unit metrics once (CA assumed treated unit in your data convention)
    treated = _compute_actual_metrics(Y, meta)
    tau_CA = treated["tau"]
    pre_rmspe_CA = treated["pre_rmspe"]
    post_rmspe_CA = treated["post_rmspe"]
    ratio_CA = treated["rmspe_ratio"]

    # In-space placebos
    df_space = placebo_in_space(Y, meta, fit_sdid, effect_key="tau")

    # Unfiltered inference
    space_unf = _rank_and_pvals(df_space, tau_CA)

    # Filtered inference (two variants)
    space_views = _filtered_views(df_space, pre_rmspe_CA)
    space_filt = {}
    for k, v in space_views.items():
        space_filt[k] = {
            "cutoff": v["cutoff"],
            **_rank_and_pvals(v["df"], tau_CA),
        }

    # Include treated row in the CSV so CA "ranks among donors" in the table
    treated_row = {
        "placebo_unit": "CA (treated)",
        "tau": tau_CA,
        "pre_rmspe": pre_rmspe_CA,
        "post_rmspe": post_rmspe_CA,
        "rmspe_ratio": ratio_CA,
        "rank_abs": space_unf["rank_abs"],
        "p_two": space_unf["p_two"],
        "p_one_neg": space_unf["p_one_neg"],
    }
    df_space_out = pd.concat([df_space, pd.DataFrame([treated_row])], ignore_index=True)

    df_space_path = report_dir / "tables" / "sdid_placebo_space.csv"
    df_space_out.to_csv(df_space_path, index=False)

    space_summary = {
        "treated": {
            "tau": tau_CA,
            "pre_rmspe": pre_rmspe_CA,
            "post_rmspe": post_rmspe_CA,
            "rmspe_ratio": ratio_CA,
        },
        "unfiltered": space_unf,
        "filtered": space_filt,
    }
    with open(report_dir / "tables" / "sdid_placebo_space_summary.json", "w") as f:
        json.dump(space_summary, f, indent=2)

    if plot:
        _maybe_save_tau_hist(
            df_space, tau_CA,
            report_dir / "figures" / "sdid_placebo_space_tau.png",
            title="SDID In-Space Placebos: tau distribution (treated marked)"
        )


    # In-time placebos
    df_time = placebo_in_time(Y, meta, fit_sdid, training_years=training_years, effect_key="tau")

    time_unf = _rank_and_pvals(df_time, tau_CA)

    time_views = _filtered_views(df_time, pre_rmspe_CA)
    time_filt = {}
    for k, v in time_views.items():
        time_filt[k] = {
            "cutoff": v["cutoff"],
            **_rank_and_pvals(v["df"], tau_CA),
        }

    df_time_path = report_dir / "tables" / "sdid_placebo_time.csv"
    df_time.to_csv(df_time_path, index=False)

    time_summary = {
        "treated": {
            "tau": tau_CA,
            "pre_rmspe": pre_rmspe_CA,
            "post_rmspe": post_rmspe_CA,
            "rmspe_ratio": ratio_CA,
        },
        "training_years": int(training_years),
        "unfiltered": time_unf,
        "filtered": time_filt,
    }
    with open(report_dir / "tables" / "sdid_placebo_time_summary.json", "w") as f:
        json.dump(time_summary, f, indent=2)

    if plot:
        _maybe_save_tau_hist(
            df_time, tau_CA,
            report_dir / "figures" / "sdid_placebo_time_tau.png",
            title="SDID In-Time Placebos: tau distribution (treated marked)"
        )

    print("Wrote:")
    print(" -", df_space_path)
    print(" -", df_time_path)


if __name__ == "__main__":
    tyro.cli(main)
