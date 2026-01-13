# PVST SOPACE HPLC (SeaBASS)

This folder is intended to contain the in-situ HPLC pigment SeaBASS files used
as ground truth for the `pvst_sopace_validation` experiment.

Notes:
- The repo `.gitignore` ignores `*.sb` under `experiments/`, so these files are
  expected to exist locally but are not committed.
- `experiments/pvst_sopace_validation/config.yaml` points `validation.hplc.path`
  at this directory, and `scripts/data/download_pace_rrs.py` will concatenate all
  `.sb` files found here (adding a `source_file` column for traceability).
