## Input data
- 145 rows of observations
    - Total after quality control + concurrency check.
    - HPLC + Rrs were within 2 hours at the same geographic location, both quality-controlled in terms of removing observations with high noise-to-signal ratios

Smoothing and interpolation procedures for Rrs
1. 5 nm moving mean (bandpass filter) --> each wavelength's reflectance was replaced by average of values in a 5 nm window centered on that point
    - First as last 4 nm of all spectra were removed
2. Then individual Rrs spectrum was visually inspected for variance in wavelength ranges where there shouldn't be; those observations were removed.

### Using with PACE Data
