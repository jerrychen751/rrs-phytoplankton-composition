"""
Shared goodness-of-fit (GOF) metric computation.

Replicates the EXACT metric formulas used in the SDP model (model.py:265-289)
so that NN and PCR results are directly comparable. Several of these definitions
are non-standard — see inline comments for details.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def compute_gof(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """
    Compute goodness-of-fit metrics matching SDP's exact formulas.

    The SDP uses non-standard R² and RMSE definitions that we replicate here
    for a fair comparison. The key differences from standard metrics:

    - **R²**: Fits a linear regression from predicted → observed, then reports
      the R² of THAT fit. Standard R² compares predictions directly to observed.
      This version measures how well a linear correction of predictions explains
      the observed variance — it's more forgiving of systematic bias.

    - **RMSE**: Takes sqrt(root_mean_squared_error(obs, corrected_pred)),
      where corrected_pred comes from the linear regression above. Since
      root_mean_squared_error already returns sqrt(MSE), this computes
      MSE^(1/4). This is a bug in the original SDP code, but we replicate
      it for consistency.

    Args:
        predicted: Model predictions, shape (n_samples,).
        observed: Ground truth values, shape (n_samples,).

    Returns:
        Dict with keys: R2, RMSE, MAE, pct_bias, median_pct_error, mean_pct_error.
    """
    pred = predicted.ravel()
    obs = observed.ravel()

    # --- R² and RMSE via linear regression: predicted → observed ---
    # This fits observed = slope * predicted + intercept, then computes
    # how much variance in observed is explained by this corrected model.
    reg = LinearRegression()
    reg.fit(pred.reshape(-1, 1), obs)

    r2 = reg.score(pred.reshape(-1, 1), obs)

    # Linearly-corrected predictions
    corrected = reg.predict(pred.reshape(-1, 1))

    # SDP takes sqrt of RMSE — effectively MSE^(1/4), not true RMSE.
    # We replicate this quirk so metrics are comparable.
    rmse = np.sqrt(root_mean_squared_error(obs, corrected))

    # --- MAE: straightforward mean absolute error ---
    mae = np.mean(np.abs(pred - obs))

    # --- Percent metrics with zero-floor at 1e-4 ---
    # Replaces exact zeros with 1e-4 to avoid division by zero.
    # Non-zero values are left unchanged.
    obs_safe = np.where(obs == 0, 1e-4, obs)

    pct_bias = np.mean(((pred - obs_safe) / obs_safe) * 100)
    pct_errors = np.abs(((pred - obs_safe) / obs_safe) * 100)
    median_pct_error = np.median(pct_errors)
    mean_pct_error = np.mean(pct_errors)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "pct_bias": pct_bias,
        "median_pct_error": median_pct_error,
        "mean_pct_error": mean_pct_error,
    }
