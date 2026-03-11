"""
Experiment configuration for the method comparison.

Compares 7 regression methods (4 linear + 3 nonlinear) for predicting
phytoplankton pigment concentrations from hyperspectral Rrs.

SOURCES: all in-situ HPLC + satellite data sources for the multi-source
k-fold experiment. Each entry maps to one geographic dataset.

CONFIG: hyperparameters shared by both scripts:
  - pax_coastal_cv.py  — LOOCV on PAX Shearwater only (n≈30)
  - multi_source_cv.py — 5-fold CV across all sources  (n≈268)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Data sources
# Each source entry describes:
#   name          — short identifier used in output labels
#   hplc_files    — HPLC file paths (relative to project root)
#   hplc_format   — "csv" (PAX, already in SDP column names) or "seabass"
#   surface_depth_m — max depth (m) to include as surface; ignored for "csv"
#   rrs_dir       — directory of PACE L3 0.1-deg Rrs .nc files
#   sst_dir       — directory of AQUA MODIS daily SST .nc files
#   sss_dir       — directory of SMAP 8-day SSS .nc4 files
#   download      — bbox and date_range for download_l3_rrs.py
#                   bbox = [lon_min, lat_min, lon_max, lat_max]
#                   date_range already includes the ±temporal_window_days buffer
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "kramer",
        # Kramer et al. 2021 (PANGAEA doi:10.1594/PANGAEA.937536): 145 open-ocean
        # samples, 2004–2018, with in-situ hyperspectral Rrs (400–700 nm, 1 nm)
        # co-located with HPLC pigments and in-situ Sal/Temp.
        # The Rrs, SST, and SSS are all embedded in the CSV itself, so no
        # satellite matchup or auxiliary file downloads are needed.
        "hplc_files": ["models/sdp_pigments/training/Kramer-etal_2021.csv"],
        "hplc_format": "kramer",
        "surface_depth_m": None,
        "rrs_dir": None,    # unused: Rrs is in the HPLC CSV
        "l2_rrs_dir": None, # unused: in-situ Rrs, no satellite matchup
        "sst_dir": None,    # unused: SST/SSS are Temp/Sal columns in the CSV
        "sss_dir": None,
        "download": None,   # nothing to download
    },
    {
        "name": "pax_shearwater",
        "hplc_files": [
            "experiments/pax_shearwater_validation/inputs/validation/in_situ_hplc/PACE-PAX_SHEARWATER_HPLC_pigments.csv",
        ],
        "hplc_format": "csv",
        "surface_depth_m": None,   # CSV is already surface-only
        "rrs_dir": "experiments/pax_shearwater_validation/inputs/rrs",
        "l2_rrs_dir": "experiments/method_comparison/inputs/pax_shearwater/l2_rrs",
        "sst_dir": "experiments/pax_shearwater_validation/inputs/sst",
        "sss_dir": "experiments/pax_shearwater_validation/inputs/sss",
        "download": {
            # Already downloaded; download_l3_rrs.py skips this source.
            "bbox": [-122.5, 33.5, -118.5, 38.5],
            "date_range": ["2024-09-04", "2024-09-17"],
        },
    },
    {
        "name": "pvst_sopace_km2419",
        # Pacific transoceanic cruise, Nov–Dec 2024.
        "hplc_files": [
            "experiments/pvst_sopace_validation/inputs/in_situ_hplc/940f89322d_PVST-SOPACE_KM2419-HPLC_20241130_R1.sb",
        ],
        "hplc_format": "seabass",
        "surface_depth_m": 10.0,
        "rrs_dir": "experiments/method_comparison/inputs/pvst_sopace_km2419/rrs",
        "l2_rrs_dir": "experiments/method_comparison/inputs/pvst_sopace_km2419/l2_rrs",
        "sst_dir": "experiments/method_comparison/inputs/pvst_sopace_km2419/sst",
        "sss_dir": "experiments/method_comparison/inputs/pvst_sopace_km2419/sss",
        "download": {
            "bbox": [-158.5, -15.5, -131.5, 21.5],
            "date_range": ["2024-11-28", "2024-12-17"],
        },
    },
    {
        "name": "pvst_sopace_tn444",
        # Bay of Bengal cruise, May–Jun 2025.
        "hplc_files": [
            "experiments/pvst_sopace_validation/inputs/in_situ_hplc/10d10ff7e5_PVST_SOPACE_TN444-HPLC_20251014.sb",
        ],
        "hplc_format": "seabass",
        "surface_depth_m": 10.0,
        "rrs_dir": "experiments/method_comparison/inputs/pvst_sopace_tn444/rrs",
        "l2_rrs_dir": "experiments/method_comparison/inputs/pvst_sopace_tn444/l2_rrs",
        "sst_dir": "experiments/method_comparison/inputs/pvst_sopace_tn444/sst",
        "sss_dir": "experiments/method_comparison/inputs/pvst_sopace_tn444/sss",
        "download": {
            "bbox": [85.5, 9.5, 89.0, 14.5],
            "date_range": ["2025-05-04", "2025-06-17"],
        },
    },
    {
        "name": "pvst_sopace_tn440",
        # Western Pacific / Micronesia cruise, Dec 2024–Jan 2025.
        "hplc_files": [
            "experiments/pvst_sopace_validation/inputs/in_situ_hplc/32c3f67fb5_PVST_SOPACE_TN440-HPLC_20251103.sb",
        ],
        "hplc_format": "seabass",
        "surface_depth_m": 10.0,
        "rrs_dir": "experiments/method_comparison/inputs/pvst_sopace_tn440/rrs",
        "l2_rrs_dir": "experiments/method_comparison/inputs/pvst_sopace_tn440/l2_rrs",
        "sst_dir": "experiments/method_comparison/inputs/pvst_sopace_tn440/sst",
        "sss_dir": "experiments/method_comparison/inputs/pvst_sopace_tn440/sss",
        "download": {
            "bbox": [147.5, 7.0, 149.5, 15.5],
            "date_range": ["2024-12-28", "2025-01-10"],
        },
    },
    {
        "name": "pvst_bats",
        # Bermuda Atlantic Time-series Study (BATS) station, Jan–Aug 2025.
        # Each file is one cruise; all are at the fixed BATS location.
        "hplc_files": [
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PACE_PVST_AE2513_HPLC_20250603_20250603_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_BATS_PLANKTON_AE2515_HPLC_20250713_20250713_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_BATS_PLANKTON_AE2517_HPLC_20250812_20250812_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AE2503_HPLC_20250304_20250304_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AE2506_HPLC_20250404_20250404_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AE2510_HPLC_20250504_20250504_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AE2512_HPLC_20250521_20250521_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AR8602_HPLC_20250117_20250117_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AR8603_HPLC_20250129_20250129_R1.sb",
            "experiments/pvst_bats_validation/inputs/in_situ_hplc/PVST_PLANTKON_BERMUDA_AR8604_HPLC_20250211_20250211_R1.sb",
        ],
        "hplc_format": "seabass",
        "surface_depth_m": 10.0,
        "rrs_dir": "experiments/method_comparison/inputs/pvst_bats/rrs",
        "l2_rrs_dir": "experiments/method_comparison/inputs/pvst_bats/l2_rrs",
        "sst_dir": "experiments/method_comparison/inputs/pvst_bats/sst",
        "sss_dir": "experiments/method_comparison/inputs/pvst_bats/sss",
        "download": {
            "bbox": [-65.5, 30.5, -63.0, 32.5],
            "date_range": ["2025-01-15", "2025-08-14"],
        },
    },
    {
        "name": "nes_lter",
        # NE US Shelf cross-shelf transect (Martha's Vineyard → shelf break).
        # 6 cruises, Feb 2024 – Apr 2025.
        "hplc_files": [
            "experiments/nes_lter_validation/inputs/in_situ_hplc/NES-LTER_AE2426_HPLC_20241106_1628_R1.sb",
            "experiments/nes_lter_validation/inputs/in_situ_hplc/NES-LTER_AR88_HPLC_20250424_1510_R1.sb",
            "experiments/nes_lter_validation/inputs/in_situ_hplc/NES-LTER_EN712_HPLC_20240210_0431_R1.sb",
            "experiments/nes_lter_validation/inputs/in_situ_hplc/NES-LTER_EN715_HPLC_20240503_1739_R1.sb",
            "experiments/nes_lter_validation/inputs/in_situ_hplc/NES-LTER_EN720_HPLC_20240906_1855_R1.sb",
            "experiments/nes_lter_validation/inputs/in_situ_hplc/NES-LTER_EN727_HPLC_20250124_1904_R1.sb",
        ],
        "hplc_format": "seabass",
        "surface_depth_m": 10.0,
        "rrs_dir": "experiments/method_comparison/inputs/nes_lter/rrs",
        "l2_rrs_dir": "experiments/method_comparison/inputs/nes_lter/l2_rrs",
        "sst_dir": "experiments/method_comparison/inputs/nes_lter/sst",
        "sss_dir": "experiments/method_comparison/inputs/nes_lter/sss",
        "download": {
            "bbox": [-71.5, 39.0, -70.0, 41.5],
            "date_range": ["2024-02-07", "2025-05-01"],
        },
    },
]


# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
CONFIG = {
    "data": {
        # PAX-only paths (for pax_coastal_cv.py backward compat)
        "hplc_path": "experiments/pax_shearwater_validation/inputs/validation/in_situ_hplc/PACE-PAX_SHEARWATER_HPLC_pigments.csv",
        "rrs_dir": "experiments/pax_shearwater_validation/inputs/rrs",
        "sst_dir": "experiments/pax_shearwater_validation/inputs/sst",
        "sss_dir": "experiments/pax_shearwater_validation/inputs/sss",
        "pigments": [
            "Tchla", "Zea", "DVchla", "ButFuco", "HexFuco", "Allo",
            "MVchlb", "Neo", "Viola", "Fuco", "Chlc12", "Chlc3", "Perid",
        ],
        "temporal_window_days": 2,
        "spectral": {
            "interp_nm": 1,
            "smooth_nm": 5,
            "edge_trim_nm": 4,
            "final_range_nm": [400, 700],
        },
    },
    "multi_source": {
        # Settings for multi_source_cv.py.
        "surface_depth_m": 10.0,
        # Wider temporal search window than PAX (more clouds in NE shelf winter).
        "temporal_window_days": 3,
        "pigments": [
            "Tchla", "Zea", "DVchla", "ButFuco", "HexFuco", "Allo",
            "MVchlb", "Neo", "Viola", "Fuco", "Chlc12", "Chlc3", "Perid",
        ],
        "spectral": {
            "interp_nm": 1,
            "smooth_nm": 5,
            "edge_trim_nm": 4,
            "final_range_nm": [400, 700],
        },
        # L2 swath-level matchup settings (used when --rrs-level L2).
        # L2 files are ~1 km per pixel vs ~11 km for L3, so the spatial and
        # temporal match windows are tighter.
        "l2": {
            "time_window_hours": 12,
            "max_distance_km": 5.0,
            # l2_flags bits that disqualify a pixel (OR mask).  Standard ocean-
            # color exclusions: ATMFAIL(0), LAND(1), HILT(3), STRAYLIGHT(4),
            # CLDICE(8), COCCOLITH(9), HISOLZEN(14), LOWLW(16),
            # NAVFAIL(25), FILTER(26).
            "qc_exclude_bits": [0, 1, 3, 4, 8, 9, 14, 16, 25, 26],
            "min_finite_fraction": 0.95,
            "max_granules_per_obs": 5,
        },
    },
    "cv": {
        # LOOCV — used by pax_coastal_cv.py
        "strategy": "LOOCV",
        "inner_folds": 5,
        # k-fold — used by multi_source_cv.py
        "kfold_k": 5,
        "kfold_seed": 42,
    },
    "sklearn": {
        # 4 linear: SDP (PCR), PLS, ElasticNet, Ridge
        # 3 nonlinear: KernelRidge (RBF), HistGBT (trees), SVR (RBF, epsilon-insensitive)
        "methods": ["SDP", "PLS", "ElasticNet", "HistGBT", "Ridge", "KernelRidge", "SVR"],
        "sdp": {
            "n_permutations": 100,
            # Raised from 30 → 50 to capture more variance components with
            # the larger training set (~120 samples vs 29 in PAX LOOCV).
            # Still safely below n_train - 1 in both cases.
            "max_pcs": 50,
            "k_folds": 5,
        },
        "pls": {
            # Raised from 20 → 30 for the same reason.
            "max_components": 30,
        },
        "elasticnet": {
            "l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            "n_alphas": 50,
            "cv": 5,
            "max_iter": 10000,
        },
        "ridge": {
            "alphas": None,   # None = logspace(-2, 4, 50)
        },
        "kernel_ridge": {
            "alphas": None,   # None = logspace(-3, 3, 7)
            "gammas": None,   # None = logspace(-5, -1, 5)
            "cv": 5,
        },
        "histgbt": {
            "max_iter": 200,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "learning_rate": 0.05,
            "max_features": 0.5,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
        },
        "svr": {
            # SVR with RBF kernel — uses epsilon-insensitive loss (ignores
            # errors < epsilon), making it more robust to extreme outliers than
            # KernelRidge (which uses squared loss). A common choice in remote
            # sensing regression problems.
            "Cs": None,       # None = logspace(-1, 3, 5)
            "gammas": None,   # None = logspace(-5, -1, 5)
            "epsilon": 0.1,
            "cv": 5,
        },
    },
    "chl_conditioning": {
        # Append log10(HPLC Tchla) as an additional feature for non-Tchla pigments.
        # Motivation: total chlorophyll is a strong proxy for trophic state and
        # community structure (Hirata et al. 2011, Sun & Ward 2021). Knowing Tchla
        # helps disambiguate similar Rrs spectra from different community compositions.
        # At deployment time, satellite chlor_a (PACE OCI L3) replaces HPLC Tchla.
        "enabled": True,
        "target_pigment": "Tchla",
    },
    "strict_qc": {
        # Filters applied when --strict-qc is passed to multi_source_cv.py.
        # Goal: remove samples with atmospheric correction failures, poor
        # bio-optical model fits, or implausible pigment values.
        "negative_rrs_reject": True,
        "max_rrs_sr": 0.05,
        "spectral_shape_check": True,
        "gsm_mapd_threshold": 33.0,
        "gsm_mapd_wl_range": [400, 600],  # wavelength range for MAPD (avoid noisy red/NIR)
        "hplc_max_concentration": {
            "Tchla": 100.0,
            "default": 10.0,
        },
        "temporal_window_days": 1,
        # Spatial matchup quality (satellite-matched data only, not Kramer).
        # max_pixel_dist_km: reject if the in-situ station is too far from the
        #   nearest L3 grid center. At 0.1°, half the diagonal is ~7.8 km; 5.5 km
        #   keeps stations within roughly half a pixel of center.
        # spatial_cv: extract a window×window box of L3 pixels around the station
        #   and compute the coefficient of variation (CV) of native-band Rrs across
        #   valid pixels. High CV means the pixel sits in a front, cloud edge, or
        #   optically complex zone where a single pixel is unrepresentative.
        #   Based on Bailey & Werdell (2006) satellite validation protocol.
        "max_pixel_dist_km": 5.5,
        "spatial_cv_window": 3,    # 3×3 pixel box
        "spatial_cv_wl_range": [440, 560],  # wavelength range for CV (strong signal)
        "max_spatial_cv": 0.15,    # reject if median CV > 15%
    },
    "regime_routing": {
        # Tchla threshold (mg/m³) separating oligotrophic from productive regime.
        # Applied to HPLC Tchla at training time and to PACE OCI L3 chlor_a at
        # deployment time. 0.5 mg/m³ is a standard open-ocean vs coastal divide.
        "tchla_threshold": 0.5,
    },
}
