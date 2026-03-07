# rrs-SDP-pigments

Python implementation of the Kramer Rrs SDP method for predicting phytoplankton pigment concentrations from hyperspectral remote sensing reflectance.

## Credits

**Original MATLAB implementation**: https://github.com/sashajane19/Rrs_pigments

**Model training code**: `models/sdp_pigments/core/model.py` adapted from Dylan Catlett: https://github.com/dcat4/bioOptix_and_PFTs

## Papers

Kramer, S.J., D.A. Siegel, S. Maritorena, D. Catlett (2022). Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on global scales. Remote Sensing of Environment, 270, 1-14, https://doi.org/10.1016/j.rse.2021.112879.

Kramer, S.J., S. Maritorena, I. Cetinić, P.J. Werdell, D.A. Siegel (2024). Phytoplankton communities quantified from hyperspectral ocean reflectance correspond to pigment-based communities. Optics Express, 32(20), 1-16. https://doi.org/10.1364/OE.529906.

## How the Model Works

The SDP (Spectral Derivative Pigments) method predicts 13 phytoplankton pigment concentrations from hyperspectral above-surface remote sensing reflectance (Rrs). It requires Rrs on a **400–700 nm grid at exactly 1 nm resolution** (301 wavelengths), plus sea surface temperature (SST) and sea surface salinity (SSS) as ancillary inputs.

The key idea: raw Rrs is dominated by water absorption and scattering, which masks the subtle spectral features caused by individual pigments. The model removes the water signal through a physics-based inversion, then takes the **second derivative** of the residual spectrum to isolate the sharp absorption features that are diagnostic of specific pigments.

### Preprocessing Pipeline (PACE L2 Rrs → Model Input)

The following steps transform raw PACE OCI Level-2 Rrs into the fixed spectral grid the model expects. These steps are **mandatory** — the model coefficients were trained on data preprocessed with exactly these parameters, so changing any of them breaks the mathematical relationship between the spectra and predicted pigment concentrations.

#### Step 1: Quality Control

Pixels are filtered using two QC gates:

1. **L2 flags masking** — Pixels are rejected if any of the following flags are set:
   ATMFAIL (bit 0), LAND (1), HIGLINT (3), HILT (4), STRAYLIGHT (8), CLDICE (9), LOWLW (14), NAVWARN (16), NAVFAIL (25), FILTER (26).
   These remove atmospheric correction failures, land, sunglint, saturation, stray light, cloud/ice contamination, low water-leaving radiance, and navigation problems.

2. **Per-pixel spectral validity** — At least 95% of the native wavelength bands must have finite (non-NaN) values. This catches partially corrupt pixels before interpolation.

#### Step 2: Interpolation to 1 nm Grid

PACE OCI native bands are not spaced at exactly 1 nm. Linear interpolation (`np.interp`) maps the native Rrs onto a uniform 1 nm grid. The interpolation is done on an **extended** range (396–704 nm) rather than the final 400–700 nm — this provides padding so that edge wavelengths have valid neighbors for the smoothing step that follows. NaN values in the native spectrum are dropped before interpolation, so internal gaps between finite bands are filled by interpolating across them. However, there is no extrapolation: target wavelengths outside the range of the finite source data remain NaN.

#### Step 3: Smoothing (5 nm Moving Mean)

A centered 5 nm moving mean is applied to the interpolated spectrum. For each wavelength λ, the output is the average of {λ−2, λ−1, λ, λ+1, λ+2}. This suppresses sensor noise, which is critical because the model later takes second derivatives — **derivatives amplify noise**, so smoothing here prevents noise from dominating the pigment signal downstream.

The smoothing is NaN-aware: a full 5-element window of finite values is required. If any element in the window is NaN, that wavelength's smoothed value is NaN (rather than silently averaging fewer points).

#### Step 4: Edge Trimming

The first and last 4 nm are trimmed from the smoothed spectrum. This removes the edges where the moving mean window was incomplete or influenced by the boundary of the interpolated data. After trimming, the spectrum covers exactly **400–700 nm at 1 nm resolution** (301 values).

#### Step 5: Final Validation

The preprocessed spectrum must be exactly 301 wavelengths with all values finite. Any spectrum with remaining NaN values is rejected — the model cannot tolerate gaps because `np.diff(..., 2)` (used later for the second derivative) would propagate NaN through the entire derivative array.

### Model Pipeline (Preprocessed Rrs → Pigment Concentrations)

Once Rrs is on the 400–700 nm @ 1 nm grid, the SDP model runs the following steps:

#### Step 6: Convert Above-Surface to Below-Surface Reflectance

The Gordon et al. (1988) reflectance model operates on **below-surface** remote sensing reflectance (rrs = L_u(0⁻)/E_d(0⁻)), not the above-surface Rrs measured by the satellite. The conversion uses Lee et al. (2002):

```
rrs = Rrs / (0.52 + 1.7 × Rrs)
```

This accounts for the air-water interface transmission and reflection.

#### Step 7: GSM Semi-Analytical Inversion

The Garver-Siegel-Maritorena (GSM) model decomposes below-surface rrs into inherent optical properties (IOPs):

- **Seawater absorption (a_sw)** — Computed from SST and SSS via the refractive index model of Quan & Fry (1994). This is why SST and SSS are required inputs.
- **Phytoplankton absorption (a_ph = A × chl^B)** — Uses empirical shape coefficients A and B from Kramer et al. The chlorophyll concentration (chl) is solved for by the optimization.
- **CDOM + detrital absorption (a_cdm)** — Exponential decay with wavelength; the slope depends on the Rrs(490)/Rrs(555) ratio.
- **Seawater backscattering (bb_sw)** — From Zhang et al. (2009), depends on SST and SSS.
- **Particle backscattering (bb_p)** — Power law with wavelength; the exponent depends on the Rrs(440)/Rrs(555) ratio.

The three free parameters (chl, CDOM magnitude, bb_p magnitude) are estimated by minimizing the difference between the observed rrs and the Gordon et al. forward model using Nelder-Mead optimization (`scipy.optimize.fmin`).

#### Step 8: Compute Rrs Residuals

Using the inverted IOPs, a modeled Rrs is reconstructed. The **residual** is:

```
RrsD = measured_Rrs − modeled_Rrs
```

This removes the broad spectral signature of water, chlorophyll, CDOM, and particle scattering. What remains are the spectral features that the simple GSM model cannot explain — and those features carry information about the specific mix of pigments present.

#### Step 9: Second Derivative

The second derivative of the Rrs residuals with respect to wavelength is computed:

```
d²RrsD/dλ²  (via np.diff(RrsD, 2))
```

This produces 299 values (first derivative) → 297 values (second derivative), covering approximately 402–698 nm. The second derivative isolates **curvature** — narrow absorption peaks and troughs caused by individual pigments — while removing any residual linear trends or offsets.

#### Step 10: Ensemble Prediction

For each pigment, 100 pre-trained linear models (from cross-validated permutations of the training data) are applied:

```
predictions = d²RrsD @ a_coefs + c_coefs    (shape: n_samples × 100)
```

The **median** across the 100 ensemble members is taken as the final prediction. This ensemble approach provides robustness to variability in the training data splits. Negative predictions are clipped to zero (pigment concentrations cannot be negative).

### Available Pigments

The model predicts 13 HPLC pigment concentrations (µg/L):

| Model Name | Display Name | Pigment |
|------------|-------------|---------|
| Tchla | T chla | Total chlorophyll *a* |
| Fuco | Fuco | Fucoxanthin |
| HexFuco | HexFuco | 19'-Hexanoyloxyfucoxanthin |
| ButFuco | ButFuco | 19'-Butanoyloxyfucoxanthin |
| Zea | Zea | Zeaxanthin |
| DVchla | DV chla | Divinyl chlorophyll *a* |
| MVchlb | MV chlb | Monovinyl chlorophyll *b* |
| Chlc12 | chl c1+c2 | Chlorophyll *c*1+*c*2 |
| Chlc3 | chl c3 | Chlorophyll *c*3 |
| Perid | Perid | Peridinin |
| Allo | Allo | Alloxanthin |
| Neo | Neo | Neoxanthin |
| Viola | Viola | Violaxanthin |

### Required Preprocessing Config

These parameters are **hardcoded constraints** validated at runtime (`main.py`):

| Parameter | Value | Reason |
|-----------|-------|--------|
| `interp_nm` | 1 | Model coefficients are defined on a 1 nm grid |
| `smooth_nm` | 5 | Second derivatives are noise-sensitive; 5 nm is empirically optimized |
| `edge_trim_nm` | 4 | Removes smoothing artifacts at spectrum boundaries |
| `final_range_nm` | [400, 700] | Wavelength range the model was trained on |

## Usage

Train model: `python3 models/sdp_pigments/train.py`

Run predictions: `python3 main.py` (prompts for config file path)

Experiment configs live under `experiments/<experiment_name>/config.py`.
Inputs/outputs are intended to be experiment-local under `experiments/<experiment_name>/{inputs,outputs}/`.
Shared model assets live under `models/sdp_pigments/`.

See `experiments/_template/config.py` for the experiment configuration format (nested schema with `experiment`, `sdp`, and `io` sections).
