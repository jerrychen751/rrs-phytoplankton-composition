# SDP pigments model assets

This folder groups the shared (cross-experiment) assets for the SDP pigment
prediction model.

## `training/`

Inputs used to train the SDP model and compute Rrs residuals:

- `training/Kramer-etal_2021.{tab,csv}` – the training dataset in tab/CSV forms
- `training/Kramer-etal_2021_column_headers.json` – extracted column metadata
- `training/residual_rrs_coefficients/` – reference coefficient tables used by the
  residual computation in `models/sdp_pigments/core/physics.py`

## `coefficients/`

Trained model coefficient ensembles (A and C terms) written by
`models/sdp_pigments/train.py` and consumed by `models/sdp_pigments/core/prediction.py`.
