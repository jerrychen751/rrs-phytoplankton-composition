"""
Scikit-learn model training and prediction for the method comparison.

Seven methods suited for small-sample, high-dimensional spectral regression
(299–300 features, ~268 training samples):

Linear methods:
1. SDP  (Spectral Derivative Pigments) — Kramer et al. PCR baseline.
2. PLS  (Partial Least Squares) — supervised dimensionality reduction.
3. ElasticNet — L1+L2 regularized linear regression.
4. Ridge — L2-regularized linear regression on all features.

Nonlinear methods:
5. HistGBT (Histogram-based Gradient Boosted Trees) — tree-based.
6. KernelRidge — Ridge with RBF kernel (squared loss).
7. SVR — Support Vector Regression with RBF kernel (epsilon-insensitive loss).

SVR vs KernelRidge: both use RBF kernels, but SVR uses epsilon-insensitive
loss (ignores errors < epsilon, penalizes larger errors linearly) while
KernelRidge uses squared loss. SVR is more robust to outliers — important
for pigments with extreme concentration ranges (e.g., DVchla, Perid).
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.sdp_pigments.core.model import rrsModelTrain


def train_sklearn_model(
    X: np.ndarray,
    y: np.ndarray,
    method_name: str,
    config: dict,
) -> dict:
    """
    Train a single sklearn model with CV hyperparameter selection.

    Standardizes X internally (except HistGBT, which doesn't need it since
    tree splits are invariant to monotonic transforms of features). The
    fitted scaler is saved so we can apply the same transform at prediction
    time — a critical detail that's easy to forget and causes silent bugs
    if the test data has a different scale than what the model was trained on.

    Args:
        X: Training features, shape (n_samples, n_features).
        y: Training targets, shape (n_samples,).
        method_name: One of "SDP", "PLS", "ElasticNet", "HistGBT", "Ridge", "KernelRidge", "SVR".
        config: Full CONFIG dict — pulls params from config["sklearn"].

    Returns:
        Dict with keys:
          - "model": the fitted sklearn estimator (Pipeline for PCR)
          - "scaler": fitted StandardScaler (or None for HistGBT)
          - "method": the method name string
    """
    sklearn_cfg = config["sklearn"]

    if method_name == "SDP":
        # SDP handles its own standardization internally (per-permutation)
        model_info = _fit_sdp(X, y, config)
        return model_info

    if method_name == "HistGBT":
        # Trees don't need standardization — splits are rank-based
        model = _fit_histgbt(X, y, sklearn_cfg)
        return {"model": model, "scaler": None, "method": method_name}

    # Standardize for linear methods: zero mean, unit variance per feature.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if method_name == "PLS":
        model = _fit_pls(X_std, y, sklearn_cfg, config)
    elif method_name == "ElasticNet":
        model = _fit_elasticnet(X_std, y, sklearn_cfg)
    elif method_name == "Ridge":
        model = _fit_ridge(X_std, y, sklearn_cfg)
    elif method_name == "KernelRidge":
        model = _fit_kernel_ridge(X_std, y, sklearn_cfg)
    elif method_name == "SVR":
        model = _fit_svr(X_std, y, sklearn_cfg)
    else:
        raise ValueError(f"Unknown sklearn method: {method_name}")

    return {"model": model, "scaler": scaler, "method": method_name}


def predict_sklearn_model(
    X_new: np.ndarray,
    model_info: dict,
) -> np.ndarray:
    """
    Predict with a trained sklearn model.

    Applies the saved scaler (if any) and clips negative predictions to 0
    (pigment concentrations can't be negative).

    Args:
        X_new: New feature matrix, shape (n_samples, n_features).
        model_info: Dict from train_sklearn_model().

    Returns:
        Predictions, shape (n_samples,), clipped to >= 0.
    """
    # SDP stores coefficients directly — prediction is X @ a_coefs + c_coefs
    if model_info["method"] == "SDP":
        a_coefs = model_info["a_coefs"]  # (n_features, n_permutations)
        c_coefs = model_info["c_coefs"]  # (n_permutations,)
        # Each permutation produces an independent prediction; take median
        all_preds = X_new @ a_coefs + c_coefs  # (n_samples, n_permutations)
        pred = np.median(all_preds, axis=1)
        return np.clip(pred, 0, None)

    model = model_info["model"]
    scaler = model_info["scaler"]

    if scaler is not None:
        X_input = scaler.transform(X_new)
    else:
        X_input = X_new

    pred = model.predict(X_input)

    # PLSRegression.predict() returns shape (n, 1) — squeeze to 1-D
    pred = np.asarray(pred).ravel()

    return np.clip(pred, 0, None)


# ---------------------------------------------------------------------------
# Internal fitting helpers
# ---------------------------------------------------------------------------


def _fit_sdp(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> dict:
    """
    Fit the actual Kramer SDP model (rrsModelTrain).

    This is the real SDP algorithm, not an sklearn approximation. It runs:
    1. 100 random 75/25 train/val permutations
    2. Within each permutation, k-fold CV (k=5) selects the best number
       of PCA components by MAE
    3. The k fold coefficients are averaged, then unstandardized back to
       the original spectral domain
    4. Each permutation's mean coefficients are validated against the
       held-out 25%

    The output is 100 sets of (a_coefs, c_coefs) — one per permutation.
    At prediction time, all 100 are applied and the median is taken.

    Note: with only ~29 training samples in LOOCV, the internal 75/25
    split gives ~22 training / ~7 validation per permutation, and the
    5-fold CV within that uses ~17 training / ~5 validation. This is
    very small, but it's the same algorithm that Kramer uses — we're
    testing the method as-is.

    Args:
        X: Raw (unstandardized) training features, shape (n_samples, 299).
        y: Training targets, shape (n_samples,).
        config: Full CONFIG dict.

    Returns:
        Dict with keys:
          - "a_coefs": unstandardized coefficients, shape (299, n_permutations)
          - "c_coefs": unstandardized intercepts, shape (n_permutations,)
          - "scaler": None (SDP handles standardization internally)
          - "method": "SDP"
    """
    sdp_cfg = config.get("sklearn", {}).get("sdp", {})
    n_permutations = sdp_cfg.get("n_permutations", 100)
    max_pcs = sdp_cfg.get("max_pcs", 30)
    k = sdp_cfg.get("k_folds", 5)

    # Cap max_pcs to what the data can support.
    # rrsModelTrain requires: max_pcs <= 0.75 * (1 - 1/k) * n_samples
    n = X.shape[0]
    max_feasible = int(0.75 * (1 - 1 / k) * n)
    max_pcs = min(max_pcs, max_feasible)

    coefficients, intercepts, summary_gofs, _ = rrsModelTrain(
        RrsD=X,
        hplc_i=y,
        pft_index="pigment",
        n_permutations=n_permutations,
        max_pcs=max_pcs,
        k=k,
        mdl_pick_metric="MAE",
    )

    r2 = summary_gofs["Mean_R2"].values[0]
    rmse = summary_gofs["Mean_RMSE"].values[0]
    print(f"      SDP: {n_permutations} permutations, max_pcs={max_pcs}, "
          f"internal R²={r2:.3f}, RMSE={rmse:.4f}")

    return {
        "a_coefs": coefficients,   # (299, n_permutations)
        "c_coefs": intercepts,     # (n_permutations,)
        "scaler": None,
        "method": "SDP",
    }


def _fit_pls(
    X_std: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
    config: dict,
) -> PLSRegression:
    """
    Fit PLS with cross-validated component selection.

    PLS (Partial Least Squares) regression finds latent "components" — linear
    combinations of the X features — that maximize the covariance between X
    and y simultaneously. Compare this to PCA, which finds components that
    maximize variance in X alone (unsupervised). Because PLS is supervised,
    its components tend to be more predictive.

    The number of components (n_components) is the key hyperparameter. We
    select via inner CV with negative MAE as the scoring metric.

    Args:
        X_std: Standardized training features.
        y: Training targets.
        sklearn_cfg: Config dict with "pls" sub-dict.
        config: Full CONFIG dict (for inner_folds).

    Returns:
        Fitted PLSRegression with the best n_components.
    """
    max_comps = sklearn_cfg["pls"]["max_components"]
    inner_cv = config.get("cv", {}).get("inner_folds", 5)

    # Cap components to what's available in the smallest inner CV fold.
    n_train = X_std.shape[0]
    n_inner_train = n_train - int(np.ceil(n_train / inner_cv))
    max_comps = min(max_comps, n_inner_train, X_std.shape[1])

    best_score = -np.inf
    best_n = 2

    for n in range(2, max_comps + 1):
        pls = PLSRegression(n_components=n, scale=False)
        scores = cross_val_score(
            pls, X_std, y, cv=inner_cv, scoring="neg_mean_absolute_error",
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_n = n

    # Refit on all training data with the best component count
    best_pls = PLSRegression(n_components=best_n, scale=False)
    best_pls.fit(X_std, y)
    print(f"      PLS: selected {best_n} components (CV MAE={-best_score:.4f})")

    return best_pls


def _fit_elasticnet(
    X_std: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
) -> ElasticNetCV:
    """
    Fit ElasticNet with built-in CV for alpha and l1_ratio selection.

    ElasticNet combines L1 (lasso) and L2 (ridge) penalties:
        loss = MSE + alpha * (l1_ratio * ||w||_1 + (1-l1_ratio) * ||w||_2^2)

    - l1_ratio=1 is pure Lasso (maximally sparse)
    - l1_ratio=0 is pure Ridge (no sparsity, shrinks correlated features together)

    ElasticNetCV uses coordinate descent with warm starting along a
    regularization path — starts with a large alpha and decreases while using
    the previous solution as the starting point.

    Args:
        X_std: Standardized training features.
        y: Training targets.
        sklearn_cfg: Config dict with "elasticnet" sub-dict.

    Returns:
        Fitted ElasticNetCV.
    """
    en_cfg = sklearn_cfg["elasticnet"]

    model = ElasticNetCV(
        l1_ratio=en_cfg["l1_ratios"],
        alphas=en_cfg["n_alphas"],
        cv=en_cfg["cv"],
        max_iter=en_cfg["max_iter"],
        random_state=42,
    )
    model.fit(X_std, y)

    n_nonzero = int(np.sum(model.coef_ != 0))
    print(
        f"      ElasticNet: alpha={model.alpha_:.4f}, "
        f"l1_ratio={model.l1_ratio_:.2f}, "
        f"{n_nonzero}/{len(model.coef_)} non-zero coefficients"
    )

    return model


def _fit_ridge(
    X_std: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
) -> RidgeCV:
    """
    Fit Ridge regression with built-in CV for alpha (regularization strength).

    Ridge adds an L2 penalty to ordinary least squares:
        loss = ||y - Xw||² + alpha * ||w||²

    Unlike PCR, Ridge keeps ALL 299 features — it doesn't discard any PCs.
    Instead, it shrinks all coefficients toward zero, with the amount of
    shrinkage controlled by alpha. In the PCA basis, Ridge shrinks each PC's
    coefficient by a factor of s_k² / (s_k² + alpha), where s_k is the k-th
    singular value. High-variance PCs (large s_k) are shrunk less; low-variance
    PCs (small s_k) are shrunk more. This is softer than PCR's hard cutoff
    which keeps the top k PCs and completely discards the rest.

    If Ridge beats PCR, it means there's useful predictive signal in the
    lower-ranked PCs that PCR throws away. If PCR beats Ridge, it means the
    hard cutoff is a better noise filter than soft shrinkage.

    RidgeCV uses efficient Leave-One-Out CV (via the SVD shortcut) to select
    alpha — this costs no more than a single Ridge fit, making it effectively
    free. The alphas parameter is a grid of candidate values spanning several
    orders of magnitude.

    Args:
        X_std: Standardized training features.
        y: Training targets.
        sklearn_cfg: Config dict with "ridge" sub-dict.

    Returns:
        Fitted RidgeCV.
    """
    ridge_cfg = sklearn_cfg.get("ridge", {})

    # Default alpha grid: log-spaced from 0.01 to 10000.
    # The optimal alpha depends on the signal-to-noise ratio — small alpha
    # (weak regularization) for clean data, large alpha for noisy data.
    # With 299 features and ~30 samples, we expect a fairly large alpha.
    alphas = ridge_cfg.get("alphas") or np.logspace(-2, 4, 50)

    # scoring="neg_mean_absolute_error" matches the metric used by PCR/PLS
    # for component selection, ensuring a fair comparison.
    model = RidgeCV(
        alphas=alphas,
        scoring="neg_mean_absolute_error",
    )
    model.fit(X_std, y)

    print(f"      Ridge: alpha={model.alpha_:.4f}")

    return model


def _fit_kernel_ridge(
    X_std: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
) -> KernelRidge:
    """
    Fit Kernel Ridge Regression with RBF kernel via inner cross-validation.

    KRR solves:  alpha_vec = (K + lambda * I)^{-1} y
    where K[i,j] = exp(-gamma * ||x_i - x_j||²) is the RBF kernel matrix.

    Conceptually, the RBF kernel maps each 299-dim spectrum into an
    infinite-dimensional feature space where the regression becomes linear.
    But we never compute those features explicitly — we only need the n×n
    kernel matrix (n=29 for LOOCV), which is tiny. This is the "kernel trick".

    Two hyperparameters to tune:
    - alpha (regularization): how much to penalize complex solutions.
      Analogous to Ridge's alpha — prevents overfitting when n << p.
    - gamma (kernel width): controls how far each training point's influence
      extends. Small gamma = each point influences distant neighbors (smooth
      predictions), large gamma = each point only affects very similar spectra
      (spiky predictions, prone to overfitting).

    We use grid search with inner CV because KernelRidge doesn't have a
    built-in CV shortcut like RidgeCV. The grid is small (5×5 = 25 combos)
    so this is fast at n=30.

    Args:
        X_std: Standardized training features.
        y: Training targets.
        sklearn_cfg: Config dict with "kernel_ridge" sub-dict.

    Returns:
        Fitted KernelRidge with best (alpha, gamma) from CV.
    """
    kr_cfg = sklearn_cfg.get("kernel_ridge", {})

    # Default grids — log-spaced to cover orders of magnitude.
    alphas = kr_cfg.get("alphas") or np.logspace(-3, 3, 7)
    gammas = kr_cfg.get("gammas") or np.logspace(-5, -1, 5)
    inner_cv = kr_cfg.get("cv", 5)

    best_score = -np.inf
    best_alpha = alphas[0]
    best_gamma = gammas[0]

    for alpha in alphas:
        for gamma in gammas:
            model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            scores = cross_val_score(
                model, X_std, y, cv=inner_cv, scoring="neg_mean_absolute_error",
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                best_gamma = gamma

    best_model = KernelRidge(alpha=best_alpha, kernel="rbf", gamma=best_gamma)
    best_model.fit(X_std, y)
    print(
        f"      KernelRidge: alpha={best_alpha:.4f}, gamma={best_gamma:.2e} "
        f"(CV MAE={-best_score:.4f})"
    )

    return best_model


def _fit_histgbt(
    X: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
) -> HistGradientBoostingRegressor:
    """
    Fit HistGradientBoosting with early stopping.

    Histogram-based GBT bins each feature into 256 buckets before finding
    splits, which makes it O(n_features * 256) per split instead of
    O(n_features * n_samples * log(n_samples)).

    Key regularization for small datasets:
    - max_depth=3: shallow trees prevent overfitting
    - learning_rate=0.05: slow learning = more robust ensemble
    - max_features=0.5: column subsampling decorrelates trees
    - min_samples_leaf=10: prevents memorizing small groups
    - early stopping via validation_fraction

    Args:
        X: Raw (unstandardized) training features.
        y: Training targets.
        sklearn_cfg: Config dict with "histgbt" sub-dict.

    Returns:
        Fitted HistGradientBoostingRegressor.
    """
    gbt_cfg = sklearn_cfg["histgbt"]

    model = HistGradientBoostingRegressor(
        max_iter=gbt_cfg["max_iter"],
        max_depth=gbt_cfg["max_depth"],
        min_samples_leaf=gbt_cfg["min_samples_leaf"],
        learning_rate=gbt_cfg["learning_rate"],
        max_features=gbt_cfg["max_features"],
        validation_fraction=gbt_cfg["validation_fraction"],
        n_iter_no_change=gbt_cfg["n_iter_no_change"],
        random_state=42,
    )
    model.fit(X, y)

    # n_iter_ is the actual number of boosting rounds before early stopping
    print(f"      HistGBT: {model.n_iter_} iterations (max {gbt_cfg['max_iter']})")

    return model


def _fit_svr(
    X_std: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
) -> SVR:
    """
    Fit Support Vector Regression with RBF kernel via grid search CV.

    SVR minimizes:
        0.5 * ||w||² + C * sum(max(0, |y_i - f(x_i)| - epsilon))

    The epsilon-insensitive loss ignores prediction errors smaller than
    epsilon (the "tube" around the regression line). Points inside the tube
    contribute zero loss; points outside contribute linearly proportional to
    their distance from the tube boundary. This makes SVR more robust to
    outliers than squared-loss methods like KernelRidge.

    Three hyperparameters:
    - C (regularization): trade-off between model complexity and fitting
      error. Large C = less regularization, fits training data more tightly.
    - gamma (kernel width): same as KernelRidge — controls locality.
    - epsilon (tube width): how much error is tolerated before penalizing.
      We fix this at 0.1 (a reasonable default) and tune only C and gamma.

    Args:
        X_std: Standardized training features.
        y: Training targets.
        sklearn_cfg: Config dict with "svr" sub-dict.

    Returns:
        Fitted SVR with best (C, gamma) from GridSearchCV.
    """
    svr_cfg = sklearn_cfg.get("svr", {})

    Cs = svr_cfg.get("Cs") or np.logspace(-1, 3, 5)
    gammas = svr_cfg.get("gammas") or np.logspace(-5, -1, 5)
    epsilon = svr_cfg.get("epsilon", 0.1)
    inner_cv = svr_cfg.get("cv", 5)

    param_grid = {"C": Cs, "gamma": gammas}
    gs = GridSearchCV(
        SVR(kernel="rbf", epsilon=epsilon),
        param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        refit=True,
    )
    gs.fit(X_std, y)

    best = gs.best_params_
    print(
        f"      SVR: C={best['C']:.4f}, gamma={best['gamma']:.2e}, "
        f"epsilon={epsilon} (CV MAE={-gs.best_score_:.4f})"
    )

    return gs.best_estimator_
