import pandas as pd
from pathlib import Path
import hashlib

def load_prop99_raw(source, cache_path) -> tuple[pd.DataFrame, str]:
    path = Path(cache_path)
    
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(source).to_csv(path, index=False)
    
    raw_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    return pd.read_csv(path), raw_sha256

def make_outcome_matrix(df, unit_col: str, time_col: str, y_col: str, treated_unit_id: int) -> pd.DataFrame:
    Y = df.pivot(index = unit_col, columns = time_col, values = y_col)
    Y = Y.sort_index(axis=0).sort_index(axis=1)
    treated = Y.loc[treated_unit_id]
    controls = Y.drop(treated_unit_id)
    Y_sorted = pd.concat([controls, treated.to_frame().T])
    return Y_sorted

def validate_matrix(Y: pd.DataFrame, treated_unit_id: int, treat_start_time: int) -> dict:
    assert not Y.isnull().values.any()
    assert not Y.index.duplicated().any()
    assert Y.columns.is_monotonic_increasing
    assert Y.index[-1] == treated_unit_id
    assert treat_start_time in Y.columns
    assert treat_start_time < Y.columns.max()
    years = list(map(int, Y.columns))
    expected = list(range(min(years), max(years) + 1))
    assert years == expected
    
    treat_start_idx = list(Y.columns).index(treat_start_time)

    return {
        "treat_start_idx": treat_start_idx,
        "n_pre": treat_start_idx,
        "n_post": Y.shape[1] - treat_start_idx,
        "N": Y.shape[0],
        "T": Y.shape[1]
    }