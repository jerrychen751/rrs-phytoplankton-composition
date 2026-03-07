"""Experiment configuration (Python).

This replaces the legacy `config.yaml` for the experiment.
"""

CONFIG = {
    "schema_version": 1,
    "experiment": {
        "name": "pvst_sopace_validation",
        "description": "PVST SOPACE HPLC (KM2419, TN440, TN444) SeaBASS bundle",
        "region": "PVST SOPACE",
        "bbox": {
            "west": -180,
            "south": -15,
            "east": 180,
            "north": 21,
        },
        "time_range": {
            "start": "2024-11-30",
            "end": "2025-06-15",
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
                # See `docs/notes.md`.
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
        # Paths are resolved relative to this config file's directory when relative.
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
