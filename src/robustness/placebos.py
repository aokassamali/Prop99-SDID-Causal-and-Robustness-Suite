import pandas as pd

def placebo_in_space(Y: pd.DataFrame, meta: dict, fit_fn: callable) -> pd.DataFrame:
    
    rows = []
    Y_donors_only = Y.iloc[:-1]

    for u in Y_donors_only.index:  # donors only (treated is last)
        Yp = Y_donors_only.drop(index=u)
        Yp = pd.concat([Yp, Y_donors_only.loc[[u]]], axis=0)  # make u the "treated last"
        res = fit_fn(Yp, meta)
        rows.append({"placebo_unit": u, "att_avg": res["att_avg"], **res["diagnostics"]})
    return pd.DataFrame(rows)

def placebo_in_time(Y: pd.DataFrame, meta: dict, fit_fn: callable, training_years: int) -> pd.DataFrame:
    years = []
    
    Y_intime = Y.iloc[:, :meta['treat_start_idx'] - 1]
    max_idx = Y_intime.shape[1]
    candidate_indices = range(training_years, max_idx - 1)

    meta_intime = meta.copy()

    for u in candidate_indices:
        meta_intime["treat_start_idx"] = u
        fake_year = Y_intime.columns[u]
        res = fit_fn(Y_intime, meta_intime)
        years.append({"treated_year": fake_year, "att_avg": res["att_avg"], **res["diagnostics"]})
    return pd.DataFrame(years)