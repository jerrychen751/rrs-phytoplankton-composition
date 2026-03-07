## Input data (Model Training Set)

*   **145 rows of observations**
    *   Total count after quality control and concurrency checks.
    *   **Concurrency:** HPLC and Rrs measurements were taken within 2 hours of each other at the same geographic location.
    *   **QC:** Both datasets were quality-controlled, specifically removing observations with high noise-to-signal ratios in the red/NIR bands.

## Rrs smoothing and interpolation

1.  **Bandpass Filter:** A 5 nm moving mean bandpass filter was applied; each wavelength's reflectance was replaced by the average of values in a 5 nm window centered on that point.
2.  **Trimming:** The first and last 4 nm of all spectra were removed following the smoothing procedure.
3.  **Visual Inspection:** Individual Rrs spectra were visually inspected for high variance in wavelength ranges where variance is physically unlikely; those observations were removed.

## PACE-PAX Shearwater validation

### Data info

*   **Cruise:** PACE-PAX Shearwater
*   **Date range:** September 6, 2024 - September 26, 2024
*   **Spatial range:**
    *   Longitude: -120.0538 deg to -119.1739 deg
    *   Latitude: 33.6046 deg to 34.3736 deg

### Matchup criteria

*   **Time Window:** +/- 3 hours time difference
*   **Spatial Sampling:** Nearest pixel spectrum (no neighborhood aggregation)
*   **QC:** L2 flags masking (see _PACE L2/L3 QC flags_ below)

### PACE OCI L2 Rrs downloads

Refer to docs/rskit\_api.md. Use rskit to download and then locally subset.

### AQUA MODIS SST

*   **Source:** earthaccess package
*   **Status:** Standard product, AQUA MODIS sensor.
*   **Product:** SST (11 micrometer), 4 km, daytime/daily.

### SSS 8-Day Running Mean

*   **Source:** harmony-py package

### Validation procedure

*   Parity plot
*   Bland-Altman plot
*   Small summary statistics table

## PACE L2/L3 QC flags

| Bit | Flag Name | Description | L2 Mask Default | L3 Mask Default |
| --- | --- | --- | --- | --- |
| 00 | ATMFAIL | Atmospheric correction failure | ON | N/A |
| 01 | LAND | Pixel is over land | ON | ON |
| 02 | PRODWARN | One or more product algorithms generated a warning | N/A | N/A |
| 03 | HIGLINT | Sunglint: reflectance exceeds threshold | ON | N/A |
| 04 | HILT | Observed radiance very high or saturated | ON | ON |
| 05 | HISATZEN | Sensor view zenith angle exceeds threshold | ON | N/A |
| 06 | COASTZ | Pixel is in shallow water | N/A | N/A |
| 07 | CLDSHDW | Pixel is in cloud shadow | N/A | N/A |
| 08 | STRAYLIGHT | Probable stray light contamination | ON | ON |
| 09 | CLDICE | Probable cloud or ice contamination | ON | ON |
| 10 | COCCOLITH | Coccolithophores detected | ON | N/A |
| 11 | TURBIDW | Turbid water detected | N/A | N/A |
| 12 | HISOLZEN | Solar zenith exceeds threshold | ON | N/A |
| 13 | spare | Reserved for future use | N/A | N/A |
| 14 | LOWLW | Very low water-leaving radiance | ON | N/A |
| 15 | CHLFAIL | Chlorophyll algorithm failure | ON | N/A |
| 16 | NAVWARN | Navigation quality is suspect | ON | N/A |
| 17 | ABSAER | Absorbing aerosols determined | N/A | N/A |
| 18 | spare | Reserved for future use | N/A | N/A |
| 19 | MAXAERITER | Maximum iterations reached for NIR iteration | ON | N/A |
| 20 | MODGLINT | Moderate sun glint contamination | N/A | N/A |
| 21 | CHLWARN | Chlorophyll out-of-bounds | ON | N/A |
| 22 | ATMWARN | Atmospheric correction is suspect | ON | N/A |
| 23 | spare | Reserved for future use | N/A | N/A |
| 24 | SEAICE | Probable sea ice contamination | N/A | N/A |
| 25 | NAVFAIL | Navigation failure | ON | N/A |
| 26 | FILTER | Pixel rejected by user-defined filter OR insufficient data for smoothing filter | N/A | N/A |
| 27 | spare | Reserved for future use | N/A | N/A |
| 28 | BOWTIEDEL | Deleted off-nadir pixels | N/A | N/A |
| 29 | HIPOL | High degree of polarization determined | N/A | N/A |
| 30 | PRODFAIL | Failure in any product | N/A | N/A |
| 31 | spare | Reserved for future use | N/A | N/A |

### Bits to exclude during validation

These flags are masked (pixels removed) during the quality control process:

| Flag Name | Bit Number | Description |
| --- | --- | --- |
| **ATMFAIL** | 0 | Atmospheric correction failure |
| **LAND** | 1 | Pixel is over land |
| **HIGLINT** | 3 | Sunglint reflectance exceeds threshold |
| **HILT** | 4 | Radiance saturated |
| **STRAYLIGHT** | 8 | Stray light contamination |
| **CLDICE** | 9 | Cloud or ice contamination |
| **LOWLW** | 14 | Very low water-leaving radiance |
| **NAVWARN** | 16 | Navigation quality suspect |
| **NAVFAIL** | 25 | Navigation failure |
| **FILTER** | 26 | Pixel rejected by user filter/insufficient data |
