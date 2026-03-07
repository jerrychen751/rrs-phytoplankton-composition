"""Template experiment configuration (Python).

Copy this file to `experiments/<experiment_name>/config.py` and edit values.

Experiment-centric folder layout (relative to this config file):
  inputs/{rrs,sss,sst}/
  outputs/
"""

CONFIG = {
    "schema_version": 1,
    "experiment": {
        # Required: filesystem-friendly experiment identifier
        "name": "experiment_name",
        "description": "Brief description of the experiment purpose and data",
        "region": "Region name (e.g., North Atlantic, Mediterranean Sea)",
        "bbox": {
            "west": -30,
            "south": 30,
            "east": -10,
            "north": 50,
        },
        "time_range": {
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
        },
    },
    "validation": {
        "hplc": {
            # Path to SeaBASS-derived in-situ dataset file (CSV or .sb), or a directory
            # containing multiple .sb/.csv files to be concatenated.
            "path": "inputs/validation/in_situ_hplc/<hplc_dataset>.csv",
            "columns": {
                "lat": "lat",
                "lon": "lon",
                "date": "date",
                "time": "time",
                "station": "station",
            },
            "datetime": {
                "date_format": "%Y%m%d",
                "time_format": "%H:%M:%S",
                "timezone": "UTC",
            },
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
        # Required: input data directory (relative to this config file or absolute path)
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
