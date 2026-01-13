# Experiments

This repo uses an experiment-centric folder layout:

- `experiments/<experiment_name>/config.yaml` – experiment configuration (YAML)
- `experiments/<experiment_name>/inputs/` – external inputs consumed by scripts
- `experiments/<experiment_name>/outputs/` – generated artifacts (safe to delete + regenerate)
- `experiments/<experiment_name>/analysis/` – notebooks, plots, and one-off analysis scripts for the experiment

## Path resolution

When a config file is loaded from disk, relative paths in the config (for example
`io.input_dir: "inputs"`) are resolved relative to the directory containing that
`config.yaml`.

This keeps configs self-contained and makes experiment folders portable.

## Common subfolders

There is no strict schema enforced on `inputs/`, but the code expects a few
conventions:

- `inputs/{rrs,sss,sst}/` – mapped (gridded) inputs for the legacy "L3 mapped Rrs" workflow
- `inputs/validation/hplc/` – in-situ HPLC pigment data used as ground truth for matchups

The `scripts/data/download_pace_rrs.py` workflow uses `validation.hplc.path` from
the config and writes its PACE L2 matchup outputs under `io.downloads_dir` (by
default `~/Downloads/rrs-SDP-pigments/`), not under `experiments/<name>/outputs/`.
