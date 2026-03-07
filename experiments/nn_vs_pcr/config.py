"""
Experiment configuration for PAX coastal internal cross-validation.

Compares 6 regression methods (4 sklearn + 2 NN) on ~30 PAX Shearwater
matchups using Leave-One-Out Cross-Validation (LOOCV). All hyperparameters
are centralized here so you can tune without hunting through code.
"""

CONFIG = {
    "data": {
        # PAX Shearwater in-situ HPLC and satellite data paths (relative to project root)
        "hplc_path": "experiments/pax_shearwater_validation/inputs/validation/in_situ_hplc/PACE-PAX_SHEARWATER_HPLC_pigments.csv",
        "rrs_dir": "experiments/pax_shearwater_validation/inputs/rrs",
        "sst_dir": "experiments/pax_shearwater_validation/inputs/sst",
        "sss_dir": "experiments/pax_shearwater_validation/inputs/sss",
        # The 13 pigments modeled by SDP (from training.py :: pigs2mdl)
        "pigments": [
            "Tchla", "Zea", "DVchla", "ButFuco", "HexFuco", "Allo",
            "MVchlb", "Neo", "Viola", "Fuco", "Chlc12", "Chlc3", "Perid",
        ],
        # Max days to search forward/backward if same-day Rrs pixel is cloudy.
        # 0 = exact date only; 2 = try +/-2 days.
        "temporal_window_days": 2,
        # Spectral preprocessing parameters (must match model training):
        # - interp_nm: interpolate L3's non-uniform wavelengths to a 1 nm grid
        # - smooth_nm: 5 nm moving-mean to suppress noise before 2nd derivative
        # - edge_trim_nm: discard 4 nm on each edge to avoid smoothing artifacts
        # - final_range_nm: the SDP model expects Rrs on [400, 700] nm
        "spectral": {
            "interp_nm": 1,
            "smooth_nm": 5,
            "edge_trim_nm": 4,
            "final_range_nm": [400, 700],
        },
    },
    "cv": {
        # LOOCV: train on N-1 samples, test on 1, repeat N times.
        # Deterministic and maximally data-efficient for small datasets.
        "strategy": "LOOCV",
        # Inner CV on the N-1 training samples selects hyperparameters
        # (n_components for PCR/PLS, alpha/l1_ratio for ElasticNet).
        "inner_folds": 5,
    },
    "nn": {
        "architectures": ["SpectralCNN", "TightPCAMLP"],
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "lr": 1e-3,
        "max_epochs": 500,
        "patience": 30,
        # Smaller batch size for 29 training samples (vs 32 for 145 Kramer samples).
        # With 29 samples and 80/20 early-stopping split, training set is ~23 samples.
        # batch_size=16 gives ~1.5 mini-batches per epoch — enough stochasticity
        # for gradient noise (which acts as implicit regularization) while not
        # so small that gradient estimates are too noisy.
        "batch_size": 16,
        # Per-fold ensemble: 10 members (vs 100 for the Kramer full-dataset ensemble).
        # With LOOCV, total model fits = 30 folds × 10 members × 2 architectures
        # × 13 pigments = 7,800. Keeping ensemble small bounds runtime.
        "n_ensemble": 10,
        # SpectralCNN: 1D conv on raw 299-dim 2nd-derivative features.
        "spectral_cnn_dropout": 0.3,
        # TightPCAMLP: PCA + small MLP.
        "tight_pcamlp_pca_components": 15,
        "tight_pcamlp_hidden_dim": 16,
        "tight_pcamlp_dropout": 0.2,
        "tight_pcamlp_weight_decay": 5e-3,
        "tight_pcamlp_patience": 40,
    },
    "sklearn": {
        "methods": ["PCR", "PLS", "ElasticNet", "HistGBT"],
        # PCR (Principal Component Regression): the sklearn equivalent of what
        # SDP does internally — PCA dimensionality reduction followed by ordinary
        # least-squares linear regression. This gives a direct linear baseline
        # for the NN comparison. Inner CV selects n_components (2..20).
        "pcr": {"max_components": 20},
        # PLS (Partial Least Squares): supervised dimensionality reduction.
        # max_components capped at 20 because N-1 = 29 training samples,
        # and components beyond ~20 capture noise rather than signal.
        "pls": {"max_components": 20},
        # ElasticNet: L1+L2 regularized linear regression.
        "elasticnet": {
            "l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            "n_alphas": 50,
            "cv": 5,
            "max_iter": 10000,
        },
        # HistGradientBoosting: gradient-boosted decision trees.
        # min_samples_leaf bumped to 10 (from 10 — same) to prevent overfitting
        # on the tiny 29-sample training sets.
        "histgbt": {
            "max_iter": 200,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "learning_rate": 0.05,
            "max_features": 0.5,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
        },
    },
}
