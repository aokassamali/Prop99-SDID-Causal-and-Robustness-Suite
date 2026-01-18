import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import tyro
from estimators.scm import fit_scm


def load_json_parquet(data_dir: str) -> tuple[pd.DataFrame, dict]:
    project_root = Path(__file__).resolve().parents[1]
    folder_path = project_root / data_dir
    
    Y = pd.read_parquet(folder_path / "Y.parquet")
    
    with open(folder_path / "meta.json", "r") as f:
        meta = json.load(f)

    return Y, meta

def save_plot(results: dict, meta: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot Trends
    ax.plot(results['y_synth'], label='Synthetic Control', linestyle='--', color='red')
    ax.plot(results['y_treated'], label='California (Actual)', linewidth=2, color='black')
    
    # 2. Add Intervention Line
    ax.axvline(x=meta['treat_start_year'], color='gray', linestyle=':', label='Prop 99 Passed')
    
    # 3. Style
    ax.set_title("Prop 99: California vs. Synthetic Control (Baseline)")
    ax.set_ylabel("Cigarette Sales (per capita)")
    ax.set_xlabel("Year")
    ax.set_xlim(1970, 2000)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

def main(data_dir: str = "data/processed"):

    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / "reports"
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    Y, meta = load_json_parquet(data_dir)
    
    if 'treated_unit' not in meta: meta['treated_unit'] = "California"

    # 3. Run Model
    results = fit_scm(Y, meta)
    
    # 4. Save Artifacts
    
    # A. Weights Table (Who makes up Synthetic California?)
    weights_path = report_dir / "tables/scm_weights.csv"
    results['weights'].sort_values(ascending=False).to_csv(weights_path)

    # B. Summary Metrics (JSON)
    metrics_path = report_dir / "tables/scm_summary.json"
    summary = {
        "att_avg": results['att_avg'],
        **results['diagnostics']
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    # C. Plot
    plot_path = report_dir / "figures/scm_path.png"
    save_plot(results, meta, plot_path)

if __name__ == "__main__":
    tyro.cli(main)