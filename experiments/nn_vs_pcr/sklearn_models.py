"""
Scikit-learn model training and prediction for NN vs linear regression comparison.

Four methods suited for small-sample, high-dimensional spectral regression
(299 features, ~29 training samples in LOOCV):

1. PCR  (Principal Component Regression) — PCA dimensionality reduction followed
         by ordinary least-squares linear regression. This is the sklearn
         equivalent of SDP's core algorithm (PCA + LinearRegression). Inner CV
         selects the number of PCA components (2..20).

2. PLS  (Partial Least Squares) — supervised dimensionality reduction that
         finds latent components maximizing covariance between X and y,
         unlike PCA which only maximizes variance in X. The chemometrics
         gold standard for spectral data.

3. ElasticNet — L1+L2 regularized linear regression. L1 (lasso penalty)
         drives irrelevant wavelength coefficients to zero (automatic feature
         selection), while L2 (ridge penalty) handles groups of correlated
         adjacent wavelengths without arbitrarily picking one.

4. HistGBT (Histogram-based Gradient Boosted Trees) — non-parametric
         nonlinear method. sklearn's HistGradientBoostingRegressor bins
         continuous features into 256 histogram buckets before splitting,
         making it much faster than traditional GBT.

No ensembles needed: PCR, PLS, and ElasticNet are deterministic (same data →
same model). HistGBT has randomness via max_features subsampling, but we fix
random_state=42 for reproducibility.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        method_name: One of "PCR", "PLS", "ElasticNet", "HistGBT".
        config: Full CONFIG dict — pulls params from config["sklearn"].

    Returns:
        Dict with keys:
          - "model": the fitted sklearn estimator (Pipeline for PCR)
          - "scaler": fitted StandardScaler (or None for HistGBT)
          - "method": the method name string
    """
    sklearn_cfg = config["sklearn"]

    if method_name == "HistGBT":
        # Trees don't need standardization — splits are rank-based
        model = _fit_histgbt(X, y, sklearn_cfg)
        return {"model": model, "scaler": None, "method": method_name}

    # Standardize for linear methods: zero mean, unit variance per feature.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if method_name == "PCR":
        model = _fit_pcr(X_std, y, sklearn_cfg, config)
    elif method_name == "PLS":
        model = _fit_pls(X_std, y, sklearn_cfg, config)
    elif method_name == "ElasticNet":
        model = _fit_elasticnet(X_std, y, sklearn_cfg)
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


def _fit_pcr(
    X_std: np.ndarray,
    y: np.ndarray,
    sklearn_cfg: dict,
    config: dict,
) -> Pipeline:
    """
    Fit PCR (PCA + Linear Regression) with cross-validated component selection.

    PCR is conceptually what SDP does: reduce the 299-dimensional spectrum
    to a few principal components, then regress pigment concentration on those
    PCs. The key hyperparameter is n_components — too few means underfitting
    (missing spectral information), too many means overfitting (fitting noise
    in the trailing PCs).

    We wrap PCA + LinearRegression in a sklearn Pipeline, which chains
    transformers and an estimator into a single object. The Pipeline's
    .fit() calls PCA.fit_transform() then LinearRegression.fit(), and
    .predict() calls PCA.transform() then LinearRegression.predict().
    This is cleaner than manually passing data between steps and ensures
    the same PCA projection is applied at train and test time.

    Component selection uses inner cross-validation (default 5-fold) on the
    training data, scored by negative MAE (sklearn convention: higher = better).

    Args:
        X_std: Standardized training features.
        y: Training targets.
        sklearn_cfg: Config dict with "pcr" sub-dict.
        config: Full CONFIG dict (for inner_folds).

    Returns:
        Fitted Pipeline([PCA(best_n), LinearRegression()]).
    """
    max_comps = sklearn_cfg["pcr"]["max_components"]
    inner_cv = config.get("cv", {}).get("inner_folds", 5)

    # Cap components to what's available in the smallest inner CV fold.
    # With k-fold CV, the largest fold has ceil(n/k) samples, so the
    # smallest training partition has n - ceil(n/k) samples. PCA requires
    # n_components <= n_samples, so we cap here to avoid failures inside
    # cross_val_score.
    n_train = X_std.shape[0]
    n_inner_train = n_train - int(np.ceil(n_train / inner_cv))
    max_comps = min(max_comps, n_inner_train, X_std.shape[1])

    best_score = -np.inf
    best_n = 2

    for n in range(2, max_comps + 1):
        pipe = Pipeline([
            ("pca", PCA(n_components=n)),
            ("lr", LinearRegression()),
        ])
        # cross_val_score with neg_mean_absolute_error: higher (less negative) = better
        scores = cross_val_score(
            pipe, X_std, y, cv=inner_cv, scoring="neg_mean_absolute_error",
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_n = n

    # Refit on all training data with the best component count
    best_pipe = Pipeline([
        ("pca", PCA(n_components=best_n)),
        ("lr", LinearRegression()),
    ])
    best_pipe.fit(X_std, y)
    print(f"      PCR: selected {best_n} components (CV MAE={-best_score:.4f})")

    return best_pipe


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
