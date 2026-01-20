import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import tyro
from estimators.sdid import fit_sdid 
from estimators.scm import fit_scm
from data.prop99 import load_json_parquet
import numpy as np
from sklearn.model_selection import ParameterGrid


def save_plot(Y: pd.DataFrame, meta: dict, sdid_res: dict, output_path: Path):
    
    y_treated = Y.iloc[-1].to_numpy()
    y_synth = sdid_res['y_synth'].to_numpy()
    gap = sdid_res['gap'].to_numpy()
    
    years = Y.columns.astype(int)
    t_idx = meta['treat_start_idx']

    lam = sdid_res['lambda']
    intercept = np.sum(lam * gap[:t_idx])
    
    y_synth_adj = y_synth + intercept
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(years, y_treated, label='California (Actual)', color='black', linewidth=2)
    ax.plot(years, y_synth_adj, label='Synthetic (SDID)', linestyle='--', color='blue', linewidth=2)
    
    ax.axvline(x=meta['treat_start_year'], color='gray', linestyle=':', label='Prop 99 Passed')
    ax.set_title(f"SDID: California vs. Synthetic Control (Tau={sdid_res['tau']:.2f})")
    ax.set_ylabel("Cigarette Sales (per capita)")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

def main(
    data_dir: str = "data/processed",
    zeta_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    eta_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
    zeta: float = 1.0,
    eta: float = 0.1,
    run_sweep: bool = True,
):
    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    Y, meta = load_json_parquet(data_dir)
# stability sweep for hyperparams
    if run_sweep:
        grid = ParameterGrid({"zeta": zeta_grid, "eta": eta_grid})
        rows = []

        for params in grid:
            z = float(params["zeta"])
            e = float(params["eta"])

            r = fit_sdid(Y, meta, z, e)

            omega = r["omega"].to_numpy()
            lam = r["lambda"]          # keep as Series for index alignment
            gap = r["gap"]             # Series over all times

            t0 = meta["treat_start_idx"]

            g_pre = gap.iloc[:t0]
            g_post = gap.iloc[t0:]

            # RMSPEs (root mean squared prediction error) on the GAP series
            pre_rmspe = float(np.sqrt(np.mean(g_pre.to_numpy() ** 2)))
            post_rmspe = float(np.sqrt(np.mean(g_post.to_numpy() ** 2)))
            rmspe_ratio = float(post_rmspe / (pre_rmspe + 1e-12))

            # SDID decomposition
            pre_gap = float((g_pre * lam).sum())      # lambda-weighted pre gap (baseline)
            post_gap = float(g_post.mean())
            tau_check = post_gap - pre_gap

            rows.append({
                "zeta": z,
                "eta": e,
                "tau": float(r["tau"]),
                "tau_check": float(tau_check),

                "pre_gap": pre_gap,
                "post_gap": post_gap,

                "pre_rmspe": pre_rmspe,
                "post_rmspe": post_rmspe,
                "rmspe_ratio": rmspe_ratio,

                "max_omega": float(np.max(omega)),
                "eff_donors": float(1.0 / np.sum(omega ** 2)),
                "max_lambda": float(lam.max()),
                "eff_years": float(1.0 / np.sum(lam.to_numpy() ** 2)),
            })

        sweep_df = pd.DataFrame(rows).sort_values(["zeta", "eta"]).reset_index(drop=True)

        sweep_path = report_dir / "tables" / "stabilitysweep.csv"
        sweep_df.to_csv(sweep_path, index=False)
# returning to standard run
    result = fit_sdid(Y, meta, zeta, eta)
    scm_result = fit_scm(Y, meta)

    comparison = {
        "scm_att": scm_result["att_avg"],
        "sdid_tau": result["tau"],
        "difference": result["tau"] - scm_result["att_avg"],
        "zeta": zeta,
        "eta": eta,
    }
    with open(report_dir / "tables/scm_vs_sdid.json", "w") as f:
        json.dump(comparison, f, indent=4)

    # 3) Save artifacts
    unit_weights_path = report_dir / "tables/sdid_weights_units.csv"
    result["omega"].sort_values(ascending=False).to_csv(unit_weights_path)

    time_weights_path = report_dir / "tables/sdid_weights_time.csv"
    result["lambda"].sort_values(ascending=False).to_csv(time_weights_path)

    plot_path = report_dir / "figures/sdid_path.png"
    save_plot(Y, meta, result, plot_path)

if __name__ == "__main__":
    tyro.cli(main)




