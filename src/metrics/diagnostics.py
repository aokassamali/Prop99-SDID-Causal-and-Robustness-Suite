import numpy as np

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return np.sqrt(np.mean((y_true - y_pred)**2))

def weight_stats(w: np.ndarray) -> dict:

    return {
        "sparsity": int(np.sum(w > 1e-3)),  # How many donors have non-zero weights
        "max_weight": float(np.max(w)),     # Did one donor dominate?
        "gini_coeff": float(np.sum(w**2))   # Herfindahl index (1.0 = 1 donor, low = uniform)
    }