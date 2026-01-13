# rrs-SDP-pigments

Python implementation of the Kramer Rrs SDP method for predicting phytoplankton pigment concentrations from hyperspectral remote sensing reflectance.

## Credits

**Original MATLAB implementation**: https://github.com/sashajane19/Rrs_pigments

**Model training code**: `core/model.py` adapted from Dylan Catlett: https://github.com/dcat4/bioOptix_and_PFTs

## Papers

Kramer, S.J., D.A. Siegel, S. Maritorena, D. Catlett (2022). Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on global scales. Remote Sensing of Environment, 270, 1-14, https://doi.org/10.1016/j.rse.2021.112879.

Kramer, S.J., S. Maritorena, I. Cetinić, P.J. Werdell, D.A. Siegel (2024). Phytoplankton communities quantified from hyperspectral ocean reflectance correspond to pigment-based communities. Optics Express, 32(20), 1-16. https://doi.org/10.1364/OE.529906.

## Usage

Train model: `python scripts/training/train_model.py`

Run predictions: `python main.py` (prompts for config file path)

See `config/template.yaml` for the experiment configuration format (nested schema with `experiment`, `sdp`, and `io` sections).
