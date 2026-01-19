import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import tyro
from estimators.scm import fit_scm
from data.prop99 import load_json_parquet
from robustness.placebos import placebo_in_space, placebo_in_time


def save_plot(df: pd.DataFrame, actual_ratio: float, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_extreme = sum(df['rmspe_ratio'] >= actual_ratio)
    total_runs = len(df) + 1  # +1 includes the actual unit
    p_value = (n_extreme + 1) / total_runs

    ax.hist(df['rmspe_ratio'], bins=15, color='gray', alpha=0.6, label='Placebo Donors')
    
    ax.axvline(x=actual_ratio, color='red', linestyle='--', linewidth=2, label=f'California (Ratio={actual_ratio:.2f})')
    
    # 4. Style & Labels
    ax.set_title(f"In-Space Placebo Test (p-value: {p_value:.3f})")
    ax.set_xlabel("RMSPE Ratio (Post-Period Error / Pre-Period Error)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

def main(data_dir: str = "data/processed"):

    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "tables/scm_summary.json"
    
    with open(summary_path, "r") as f:
        baseline_stats = json.load(f)
    
    actual_ratio = baseline_stats['rmspe_ratio']


    Y, meta = load_json_parquet(data_dir)

    # 3. Run Model
    placebo_inspace_frame = placebo_in_space(Y, meta, fit_scm)

    training_years = 10 #determine run time placebo training years, index based
    placebo_intime_frame = placebo_in_time(Y, meta, fit_scm, training_years)
    
    # 4. Save Artifacts
    
    # Placebo in space summary
    inspace_path = report_dir / "tables/scm_placebo_space.csv"
    placebo_inspace_frame.sort_values("placebo_unit",ascending=False).to_csv(inspace_path, index=False)

    # B. Placebo In Time summary
    intime_path = report_dir / "tables/scm_placebo_time.csv"
    placebo_intime_frame.sort_values("treated_year", ascending=False).to_csv(intime_path, index=False)

    # C. Plot
    plot_path = report_dir / "figures/scm_placebo_space.png"
    save_plot(placebo_inspace_frame, actual_ratio, plot_path)

if __name__ == "__main__":
    tyro.cli(main)