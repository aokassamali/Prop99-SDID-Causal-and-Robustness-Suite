def fit_sdid(Y: pd.DataFrame, treated_unit_id: int, treat_start_year: int, zeta: float, eta: float) -> dict:

returns:

tau (float)

omega (pd.Series)

lambda (pd.Series)

diagnostics (dict)

def solve_simplex_ridge(A: np.ndarray, b: np.ndarray, l2: float) -> np.ndarray: