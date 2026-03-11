"""Experiment configuration (Python).

NES-LTER validation: 6 cruises along the NES-LTER cross-shelf transect
(Martha's Vineyard to the shelf break, ~39.8–41.3°N, 70.5–70.9°W).
Feb 2024 – Apr 2025.  Satellite validation uses surface samples.
"""

CONFIG = {
    "schema_version": 1,
    "experiment": {
        "name": "nes_lter_validation",
        "description": "NES-LTER HPLC (6 cruises, Feb 2024 – Apr 2025) SeaBASS bundle",
        "region": "NES-LTER (NE US Shelf)",
        "bbox": {
            "west": -71,
            "south": 39.5,
            "east": -70,
            "north": 41.5,
        },
        "time_range": {
            "start": "2024-02-09",
            "end": "2025-04-29",
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
