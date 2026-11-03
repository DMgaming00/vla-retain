"""The main test: does new-state policy divergence predict forgetting better than weight or feature drift?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

PREDICTORS = {
    "new_velocity_distance": "Velocity distance, new-task states",
    "new_action_chunk_distance": "Shared-noise action distance, new-task states",
    "new_action_mmd": "Action MMD, new-task states",
    "old_velocity_distance": "Velocity distance, old-task states",
    "old_action_chunk_distance": "Shared-noise action distance, old-task states",
    "old_action_mmd": "Action MMD, old-task states",
    "weight_drift": "Relative weight L2 (mean over groups)",
    "new_feature_drift": "1 - CKA, new-task states",
    "old_feature_drift": "1 - CKA, old-task states",
}
PRIMARY = "new_velocity_distance"
MERGE_KEYS = ["run_id", "method", "seed", "step", "alpha_spec"]


def join_tables(checkpoints: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty or drift.empty:
        return pd.DataFrame()
    return checkpoints.merge(drift, on=MERGE_KEYS, how="inner")


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def spearman_ci(frame: pd.DataFrame, predictor: str, target: str = "forgetting", n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman rho, with a percentile CI from a bootstrap over runs, since checkpoints of one run are not independent."""
    data = frame[[predictor, target, "run_id"]].dropna()
    rho = spearman(data[predictor], data[target])
    runs = data["run_id"].unique()
    groups = {r: g[[predictor, target]].to_numpy() for r, g in data.groupby("run_id")}
    rng = np.random.default_rng(seed)
    draws = []
    if len(runs) >= 3:
        for _ in range(n_boot):
            sample = np.concatenate([groups[r] for r in rng.choice(runs, size=len(runs))])
            value = spearman(sample[:, 0], sample[:, 1])
            if not np.isnan(value):
                draws.append(value)
    lo, hi = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) if draws else (float("nan"),) * 2
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi, "n": len(data), "n_runs": len(runs)}


def leave_one_method_out_r2(frame: pd.DataFrame, predictor: str, target: str = "forgetting") -> float:
    """Fit a line on all methods but one, predict the held-out method, and pool out-of-sample R^2."""
    data = frame[[predictor, target, "method"]].dropna()
    methods = data["method"].unique()
    if len(methods) < 3:
        return float("nan")
    predictions, truths = [], []
    for method in methods:
        train = data[data["method"] != method]
        test = data[data["method"] == method]
        if np.ptp(train[predictor].to_numpy()) == 0:
            return float("nan")
        slope, intercept = np.polyfit(train[predictor], train[target], 1)
        predictions.extend(slope * test[predictor] + intercept)
        truths.extend(test[target])
    truths, predictions = np.asarray(truths), np.asarray(predictions)
    total = np.sum((truths - truths.mean()) ** 2)
    return float(1 - np.sum((truths - predictions) ** 2) / total) if total > 0 else float("nan")


def partial_spearman(frame: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict:
    """Spearman partial correlation: rank everything, residualize on the ranked controls, correlate residuals."""
    data = frame[[x, y, *controls]].dropna()
    n, k = len(data), len(controls)
    if n < k + 4:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    ranks = data.rank().to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), ranks[:, 2:]])

    def residual(values: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ beta

    rx, ry = residual(ranks[:, 0]), residual(ranks[:, 1])
    if np.ptp(rx) < 1e-12 or np.ptp(ry) < 1e-12:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - k - 2
    t = rho * np.sqrt(dof / max(1 - rho**2, 1e-12))
    return {"rho": rho, "p": float(2 * sps.t.sf(abs(t), dof)), "n": n}


def correlation_report(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """All predictors against forgetting: rho with CI, and leave-one-method-out R^2."""
    rows = []
    for predictor, label in PREDICTORS.items():
        if predictor not in frame or frame[predictor].isna().all():
            continue
        row = {"predictor": predictor, "label": label, **spearman_ci(frame, predictor, n_boot=n_boot, seed=seed)}
        row["lomo_r2"] = leave_one_method_out_r2(frame, predictor)
        rows.append(row)
    return pd.DataFrame(rows)


def primary_partial(frame: pd.DataFrame) -> dict:
    return partial_spearman(frame, PRIMARY, "forgetting", ["weight_drift", "target_sr"])
