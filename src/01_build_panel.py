from pathlib import Path
from data.prop99 import load_prop99_raw, make_outcome_matrix, validate_matrix
from dataclasses import dataclass
import pandas as pd
import hashlib
import yaml
import json
import tyro
import random
import numpy as np

@dataclass
class DatasetConfig:
    name: str
    url: str
    raw_path: str

@dataclass
class PanelConfig:
    unit_col: str
    time_col: str
    y_col: str
    treated_unit_id: int
    intervention_year: int
    treat_start_year: int

@dataclass
class ReproConfig:
    seed: int

@dataclass
class ProjectConfig:
    dataset: DatasetConfig
    panel: PanelConfig
    repro: ReproConfig

def load_data_config(config_name: str) -> ProjectConfig:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / config_name

    with open(config_path, "r") as f:
        raw_dict = yaml.safe_load(f)
        
    # Only load the 'data' section of your yaml
    return ProjectConfig(
        dataset=DatasetConfig(**raw_dict['dataset']),
        panel=PanelConfig(**raw_dict['panel']),
        repro=ReproConfig(**raw_dict['repro'])
    )

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Global Seed set to: {seed}")

def save_artifacts(Y: pd.DataFrame, meta: dict, output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    Y.to_parquet(path / "Y.parquet")
    
    meta["saved_at"] = str(path)
    
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

def main(config_name = "prop99.yaml"):

    cfg = load_data_config(config_name)
    seed_everything(cfg.repro.seed)
    
    df, raw_hash = load_prop99_raw(cfg.dataset.url, cfg.dataset.raw_path)
    
    Y = make_outcome_matrix(
        df, 
        unit_col=cfg.panel.unit_col, 
        time_col=cfg.panel.time_col, 
        y_col=cfg.panel.y_col, 
        treated_unit_id=cfg.panel.treated_unit_id
    )
    
    meta = validate_matrix(Y, cfg.panel.treated_unit_id, cfg.panel.treat_start_year)

    years = list(map(int, Y.columns))
    meta.update({
    "treated_unit_id": cfg.panel.treated_unit_id,
    "treat_start_year": cfg.panel.treat_start_year,
    "T0_year": years[meta["treat_start_idx"] - 1],
    "year_min": min(years), "year_max": max(years)
    })  

    meta["raw_sha256"] = raw_hash
    save_artifacts(Y, meta, "data/processed")

if __name__ == "__main__":
    tyro.cli(main)