"""
NN training loop with identical evaluation protocol to SDP.

Key design decisions:
- Split synchronization: generates the exact same 75/25 train/val splits
  as SDP by replicating its random state evolution.
- Standardization: uses training-set mean/std for both train and val
  (standard ML practice; slightly different from SDP which standardizes
  val with its own stats — a minor data leakage in the original code).
- Early stopping: sub-splits training data 80/20 for a held-out early
  stopping set, analogous to SDP's k-fold model selection.
- GOF metrics: computed with the shared evaluation.py to match SDP exactly.

Also provides ensemble training/prediction for independent validation:
- train_nn_ensemble(): trains N models on the FULL dataset (with 80/20
  sub-splits for early stopping only) for use on external test data.
- predict_nn_ensemble(): applies a saved ensemble to new data, returning
  the median prediction across ensemble members.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from experiments.nn_vs_pcr.nn_models import build_model
from experiments.nn_vs_pcr.evaluation import compute_gof


def generate_sdp_splits(
    n_samples: int,
    n_permutations: int,
    train_fraction: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/val splits identical to those used by SDP's rrsModelTrain.

    The SDP seeds np.random to 100 at the start, then for each permutation calls:
      1. np.random.permutation(n_samples) — takes first 75% as training indices
      2. np.random.permutation(n_train) — for k-fold CV indices (not used by NN,
         but we must consume this call to keep the random state in sync)

    Since rrsModelTrain reseeds to 100 on every call, all 13 pigments see the
    same split sequence. We replicate that here.

    Args:
        n_samples: Total number of samples (e.g., 145).
        n_permutations: Number of permutations (e.g., 100).
        train_fraction: Fraction for training (e.g., 0.75).
        seed: Random seed (e.g., 100).

    Returns:
        List of (train_indices, val_indices) tuples, one per permutation.
    """
    np.random.seed(seed)
    splits = []
    n_train = int(n_samples * train_fraction)

    for _ in range(n_permutations):
        # Call 1: 75/25 split (SDP model.py line 81)
        perm = np.random.permutation(n_samples)
        train_idx = perm[:n_train]

        # Validation indices: all samples NOT in training set, original order
        # np.delete preserves original ordering of remaining elements
        val_idx = np.delete(np.arange(n_samples), train_idx)

        # Call 2: k-fold permutation (SDP model.py line 96)
        # We don't use these indices, but we MUST consume this call to keep
        # numpy's global random state aligned with SDP's state evolution
        _ = np.random.permutation(n_train)

        splits.append((train_idx, val_idx))

    return splits


def _train_single_permutation(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    arch_name: str,
    config: dict,
    perm_idx: int,
    pca_components_override: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Train and evaluate a single NN for one permutation.

    Steps:
      1. Split data using pre-generated indices
      2. Standardize features with training-set statistics
      3. (PCAMLP only) Apply SVD to reduce to top-k PCs
      4. Sub-split training 80/20 for early stopping
      5. Train with Adam, MSE loss, dropout, weight decay
      6. Early stop when val loss plateaus
      7. Predict on held-out 25%, clip negatives, compute GOF

    Args:
        X: Full feature matrix, shape (n_samples, n_features).
        y: Full target vector, shape (n_samples,).
        train_idx: Training sample indices for this permutation.
        val_idx: Validation sample indices for this permutation.
        arch_name: Architecture name ("SpectralCNN", "TightPCAMLP").
        config: Full CONFIG dict.
        perm_idx: Permutation index (used for reproducible sub-split seeding).

    Returns:
        Tuple of (gof_dict, predictions, observed) for this permutation.
    """
    nn_cfg = config["nn"]

    # --- 1. Split ---
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # --- 2. Standardize using training-set statistics ---
    # axis=0 computes stats across samples, giving per-feature mean/std.
    # Adding epsilon (1e-10) prevents division by zero for constant features.
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-10
    X_train_std = (X_train - train_mean) / train_std
    X_val_std = (X_val - train_mean) / train_std

    # --- 3. PCA reduction for TightPCAMLP ---
    if arch_name == "TightPCAMLP":
        # SVD on standardized training data (same approach as SDP model.py:136)
        # numpy's SVD: X = U @ diag(S) @ VT
        # The rows of VT are the principal component directions
        U, S, VT = np.linalg.svd(X_train_std, full_matrices=False)
        n_pcs = pca_components_override if pca_components_override is not None else nn_cfg["tight_pcamlp_pca_components"]
        # Project both sets onto top-k PCs by multiplying by VT^T (the loadings)
        pca_basis = VT[:n_pcs]  # shape: (n_pcs, n_features)
        X_train_input = X_train_std @ pca_basis.T  # shape: (n_train, n_pcs)
        X_val_input = X_val_std @ pca_basis.T      # shape: (n_val, n_pcs)
        input_dim = n_pcs
    else:
        X_train_input = X_train_std
        X_val_input = X_val_std
        input_dim = X_train_std.shape[1]

    # --- 4. Sub-split training data 80/20 for early stopping ---
    # Use a separate RNG (default_rng) to avoid polluting numpy's global state.
    # Each permutation gets a deterministic but different sub-split.
    rng = np.random.default_rng(42 + perm_idx)
    n_es = len(X_train_input)
    es_perm = rng.permutation(n_es)
    n_es_train = int(n_es * 0.8)
    es_train_idx = es_perm[:n_es_train]
    es_val_idx = es_perm[n_es_train:]

    X_es_train = X_train_input[es_train_idx]
    y_es_train = y_train[es_train_idx]
    X_es_val = X_train_input[es_val_idx]
    y_es_val = y_train[es_val_idx]

    # --- 5. Build model and training infrastructure ---
    # Use a fixed seed per permutation for weight initialization reproducibility
    torch.manual_seed(42 + perm_idx)
    model = build_model(arch_name, input_dim, config)

    # Per-architecture hyperparameter overrides:
    # TightPCAMLP uses stronger weight decay (5e-3 vs 1e-3) because its
    # ~305 params are still ~2x the sample count, and more patience (40 vs 30)
    # because the stronger regularization slows convergence.
    if arch_name == "TightPCAMLP":
        weight_decay = nn_cfg.get("tight_pcamlp_weight_decay", nn_cfg["weight_decay"])
        patience = nn_cfg.get("tight_pcamlp_patience", nn_cfg["patience"])
    else:
        weight_decay = nn_cfg["weight_decay"]
        patience = nn_cfg["patience"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=nn_cfg["lr"],
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    # Convert to tensors. float32 is standard for PyTorch training —
    # double (float64) would work but is slower on GPU and unnecessary here.
    X_es_train_t = torch.tensor(X_es_train, dtype=torch.float32)
    y_es_train_t = torch.tensor(y_es_train, dtype=torch.float32)
    X_es_val_t = torch.tensor(X_es_val, dtype=torch.float32)
    y_es_val_t = torch.tensor(y_es_val, dtype=torch.float32)

    # DataLoader handles batching and shuffling. With ~86 training samples
    # and batch_size=32, we get ~3 mini-batches per epoch.
    train_dataset = TensorDataset(X_es_train_t, y_es_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=nn_cfg["batch_size"],
        shuffle=True,
    )

    # --- 6. Training loop with early stopping ---
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(nn_cfg["max_epochs"]):
        # Training phase: model.train() enables dropout
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        # Validation phase: model.eval() disables dropout
        model.eval()
        with torch.no_grad():
            val_pred = model(X_es_val_t)
            val_loss = criterion(val_pred, y_es_val_t).item()

        # Early stopping: save best model, stop if no improvement for `patience` epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # state_dict() returns a copy of all learnable parameters
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 7. Predict on held-out validation set ---
    model.eval()
    X_val_t = torch.tensor(X_val_input, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_val_t).numpy()

    # Clip negative predictions to 0 (pigment concentrations can't be negative)
    # Same constraint as SDP (model.py:258-259 for pft_index='pigment')
    predictions = np.clip(predictions, 0, None)

    # --- 8. Compute GOF ---
    gof = compute_gof(predictions, y_val)

    return gof, predictions, y_val


def train_nn_pigment(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    arch_name: str,
    config: dict,
    pigment_name: str,
    pca_components_override: int | None = None,
) -> dict:
    """
    Train and evaluate one NN architecture for one pigment across all permutations.

    This is the NN equivalent of one call to rrsModelTrain — it runs through
    all 100 permutations and collects per-permutation GOF metrics.

    Args:
        X: Full feature matrix (2nd derivative), shape (n_samples, 299).
        y: Pigment concentrations, shape (n_samples,).
        splits: Pre-generated (train_idx, val_idx) list from generate_sdp_splits.
        arch_name: Which architecture to use.
        config: Full CONFIG dict.
        pigment_name: For logging/progress output.

    Returns:
        Dict with:
          - 'summary': dict of mean/std for each metric
          - 'all_gofs': dict of per-permutation metric arrays
          - 'example_preds': (predictions, observed) from first permutation
    """
    n_perms = len(splits)
    metric_names = ["R2", "RMSE", "MAE", "pct_bias", "median_pct_error", "mean_pct_error"]
    all_metrics = {m: np.zeros(n_perms) for m in metric_names}
    example_preds = None

    for i, (train_idx, val_idx) in enumerate(splits):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    {arch_name} | {pigment_name} | permutation {i + 1}/{n_perms}")

        gof, preds, obs = _train_single_permutation(
            X, y, train_idx, val_idx, arch_name, config, perm_idx=i,
            pca_components_override=pca_components_override,
        )

        for m in metric_names:
            all_metrics[m][i] = gof[m]

        # Save first permutation's predictions for scatter plots
        if i == 0:
            example_preds = (preds, obs)

    # Summarize across permutations
    summary = {}
    for m in metric_names:
        summary[f"Mean_{m}"] = np.mean(all_metrics[m])
        summary[f"SD_{m}"] = np.std(all_metrics[m])

    return {
        "summary": summary,
        "all_gofs": all_metrics,
        "example_preds": example_preds,
    }


# ---------------------------------------------------------------------------
# Ensemble training / prediction for independent (external) validation
# ---------------------------------------------------------------------------


def train_nn_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    arch_name: str,
    config: dict,
    n_ensemble: int = 100,
    seed_base: int = 42,
) -> list[dict]:
    """
    Train an ensemble of models on the FULL dataset for external prediction.

    Unlike the cross-validation functions above (which hold out 25% for
    evaluation on each permutation), this trains every ensemble member on
    ALL available samples. The only split is an internal 80/20 sub-split
    of the full data for early stopping — this prevents overfitting during
    training but doesn't waste any data on a held-out test set, because the
    test set is a completely separate external dataset (e.g., PAX Shearwater).

    Each ensemble member gets a different random seed, producing:
    - Different weight initialization (torch.manual_seed)
    - Different 80/20 sub-split (np.random.default_rng)
    This diversity is key to ensemble robustness — the members disagree on
    noisy regions but agree on well-supported predictions, and the median
    smooths out individual model quirks.

    Args:
        X: Full feature matrix, shape (n_samples, n_features).
        y: Full target vector, shape (n_samples,).
        arch_name: Architecture name ("SpectralCNN", "TightPCAMLP").
        config: Full CONFIG dict.
        n_ensemble: Number of ensemble members to train.
        seed_base: Base seed — member i uses seed_base + i.

    Returns:
        List of dicts, one per ensemble member, each containing:
          - state_dict: trained model weights
          - train_mean: per-feature mean used for standardization
          - train_std: per-feature std used for standardization
          - input_dim: model input dimensionality
          - pca_basis: PCA projection matrix (PCAMLP only, else None)
    """
    nn_cfg = config["nn"]
    trained_models: list[dict] = []

    for i in range(n_ensemble):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    {arch_name} | ensemble member {i + 1}/{n_ensemble}")

        # --- Deterministic sub-split for early stopping ---
        # default_rng is a modern NumPy RNG that doesn't touch the global
        # np.random state. Each member gets a unique but reproducible split.
        rng = np.random.default_rng(seed_base + i)
        n = len(X)
        perm = rng.permutation(n)
        n_train = int(n * 0.8)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # --- Standardize using training-set statistics ---
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0) + 1e-10
        X_train_std = (X_train - train_mean) / train_std
        X_val_std = (X_val - train_mean) / train_std

        # --- PCA reduction for TightPCAMLP ---
        pca_basis = None
        if arch_name == "TightPCAMLP":
            U, S, VT = np.linalg.svd(X_train_std, full_matrices=False)
            n_pcs = nn_cfg["tight_pcamlp_pca_components"]
            pca_basis = VT[:n_pcs]  # (n_pcs, n_features)
            X_train_input = X_train_std @ pca_basis.T
            X_val_input = X_val_std @ pca_basis.T
            input_dim = n_pcs
        else:
            X_train_input = X_train_std
            X_val_input = X_val_std
            input_dim = X_train_std.shape[1]

        # --- Build and train ---
        torch.manual_seed(seed_base + i)
        model = build_model(arch_name, input_dim, config)

        # Per-architecture weight decay override (same logic as _train_single_permutation)
        if arch_name == "TightPCAMLP":
            weight_decay = nn_cfg.get("tight_pcamlp_weight_decay", nn_cfg["weight_decay"])
            patience = nn_cfg.get("tight_pcamlp_patience", nn_cfg["patience"])
        else:
            weight_decay = nn_cfg["weight_decay"]
            patience = nn_cfg["patience"]

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=nn_cfg["lr"],
            weight_decay=weight_decay,
        )
        criterion = nn.MSELoss()

        X_train_t = torch.tensor(X_train_input, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_input, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=nn_cfg["batch_size"],
            shuffle=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(nn_cfg["max_epochs"]):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        trained_models.append({
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "input_dim": input_dim,
            "pca_basis": pca_basis,
        })

    return trained_models


def predict_nn_ensemble(
    X_new: np.ndarray,
    trained_models: list[dict],
    arch_name: str,
    config: dict,
) -> np.ndarray:
    """
    Apply a trained ensemble to new (external) data and return median predictions.

    For each ensemble member, the function:
    1. Standardizes X_new with that member's saved training-set statistics
    2. (PCAMLP only) Projects onto that member's saved PCA basis
    3. Runs a forward pass through the model

    The final prediction is the **median** across all ensemble members. Median
    is more robust than mean to outlier predictions from individual models —
    this matches SDP's approach of taking the median over 100 permutations
    (prediction.py:125).

    Args:
        X_new: New feature matrix, shape (n_new, n_features).
        trained_models: List of model dicts from train_nn_ensemble().
        arch_name: Architecture name (must match training).
        config: Full CONFIG dict.

    Returns:
        Median predictions, shape (n_new,), clipped to >= 0.
    """
    all_preds = []

    for model_dict in trained_models:
        # Standardize with this member's saved statistics
        X_std = (X_new - model_dict["train_mean"]) / model_dict["train_std"]

        # PCA projection if applicable
        if model_dict["pca_basis"] is not None:
            X_input = X_std @ model_dict["pca_basis"].T
        else:
            X_input = X_std

        # Rebuild model architecture and load saved weights
        model = build_model(arch_name, model_dict["input_dim"], config)
        model.load_state_dict(model_dict["state_dict"])
        model.eval()

        X_t = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X_t).numpy()

        all_preds.append(pred)

    # Stack into (n_ensemble, n_new), take median across ensemble axis
    stacked = np.stack(all_preds, axis=0)
    median_pred = np.median(stacked, axis=0)

    # Clip negatives — pigment concentrations can't be negative
    return np.clip(median_pred, 0, None)
