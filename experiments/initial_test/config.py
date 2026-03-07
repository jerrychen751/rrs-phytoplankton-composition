"""Experiment configuration (Python).

This replaces the legacy `config.yaml` for the experiment.
"""

CONFIG = {
    "schema_version": 1,
    "experiment": {
        "name": "initial_test",
        "description": "North Atlantic, September 2025 - Initial test run",
        "region": "North Atlantic",
        "bbox": {
            "west": -30,
            "south": 30,
            "east": -10,
            "north": 50,
        },
        "time_range": {
            "start": "2025-09-01",
            "end": "2025-09-10",
        },
    },
    "sdp": {
        # None = all 13 pigments
        "pigments": None,
    },
    "io": {
        # Paths are resolved relative to this config file's directory when relative.
        "input_dir": "inputs",
        "output": {
            "dir": "outputs",
            "filename": "sdp_results.nc",
        },
    },
}

