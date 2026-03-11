"""Experiment configuration (Python).

PVST BATS Plankton validation: 10 monthly cruises to BATS station
(Bermuda Atlantic Time-series Study), Jan–Aug 2025.
Depth profiles at ~31.7°N, 64.1°W; satellite validation uses surface samples.
"""

CONFIG = {
    "schema_version": 1,
    "experiment": {
        "name": "pvst_bats_validation",
        "description": "PVST BATS Plankton HPLC (10 cruises, Jan–Aug 2025) SeaBASS bundle",
        "region": "BATS (Bermuda)",
        "bbox": {
            "west": -65,
            "south": 31,
            "east": -63,
            "north": 32,
        },
        "time_range": {
            "start": "2025-01-17",
            "end": "2025-08-12",
        },
    },
    "validation": {
        "hplc": {
            # Directory containing one or more SeaBASS HPLC files.
            "path": "inputs/in_situ_hplc",
        },
        "matching": {
            "time_window_hours": 3,
            "max_distance_km": 5,
            "search_padding_deg": None,
            "max_granules_per_obs": 3,
        },
        "qc": {
            "l2_flags": {
                "exclude_bits": [0, 1, 3, 4, 8, 9, 14, 16, 25, 26],
            },
            "pixel_spectral_validity": {
                "min_finite_fraction": 0.95,
            },
        },
        "spectral": {
            "interp_nm": 1,
            "smooth_nm": 5,
            "edge_trim_nm": 4,
            "final_range_nm": [400, 700],
        },
    },
    "pace": {
        "l2_aop_short_names": [
            "PACE_OCI_L2_AOP",
            "PACE_OCI_L2_AOP_NRT",
        ],
    },
    "sdp": {
        # None = all 13 pigments
        "pigments": None,
    },
    "io": {
        "downloads_dir": "~/Downloads/rrs-SDP-pigments",
        "input_dir": "inputs",
        "pace_l2_rrs": {
            "output_subdir": "pace_l2_rrs",
            "output_filename": "pace_l2_rrs_matchups.nc",
            "keep_raw_granules": False,
            "raw_granules_subdir": "raw_granules",
        },
        "output": {
            "dir": "outputs",
            "filename": "sdp_results.nc",
        },
    },
}
