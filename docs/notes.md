## Input data

*   145 rows of observations
    *   Total after quality control + concurrency check.
    *   HPLC + Rrs were within 2 hours at the same geographic location, both quality-controlled in terms of removing observations with high noise-to-signal ratios

## Rrs smoothing and interpolation

1.  5 nm moving mean (bandpass filter) --> each wavelength's reflectance was replaced by average of values in a 5 nm window centered on that point
    *   First as last 4 nm of all spectra were removed
2.  Then individual Rrs spectrum was visually inspected for variance in wavelength ranges where there shouldn't be; those observations were removed.

## PACE-PAX Shearwater validation

### Data info

HPLC from PACE-PAX Shearwater cruise

Date range: September 6, 2024 - September 26, 2024

Spatial range:

*   Longitude: -120.0538 deg to -119.1739 deg
*   Latitude: 33.6046 deg to 34.3736 deg

### Matchup criteria

*   +/- 3 hours time difference
*   3x3 neighborhood median spectrum (>= 6 valid pixels)
*   L2 flags masking + spectral homogeneity QC (see PACE L2/L3 QC flags below)

### PACE OCI L2 Rrs matchups (current workflow)

Use `scripts/data/download_pace_rrs.py` with the experiment config:

```
python scripts/data/download_pace_rrs.py --config experiments/pax_shearwater_validation/config.yaml
```

*   Downloads only the granules needed to match the in-situ station times/locations.
*   Uses scanline time from `scan_line_attributes` for temporal matching (prefers `time` if present; otherwise derives from `year`/`day`/`msec`).
*   Outputs a compact NetCDF matchup dataset under:  
    `~/Downloads/rrs-SDP-pigments/pax_shearwater_validation/pace_l2_rrs/`

### PACE OCI RRS (historical L3 mapped approach; deprecated)

https://harmony.earthdata.nasa.gov/C3620140444-OB_CLOUD/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(33:35)&subset=lon(-121:-119)&subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")&format=application%2Fx-netcdf4

provisional product status, PACE OCI sensor, rrs product, daily, 0.1 degree resolution, start date is sept 6 2024 to sept 26 2024. lon from -121 to -119. lat from 33 to 35. type is mapped.

can't use browser to download; need to use script (otherwise only get 5 wavelengths)

### AQUA MODIS SST

https://harmony.earthdata.nasa.gov/C1615905770-OB_DAAC/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(33:35)&subset=lon(-121:-119)&subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")&format=application%2Fx-netcdf4

standard product status, AQUA MODIS sensor, sst (11 micrometer) product, 4 km, daytime / daily, start date is sept 6 2024 to sept 26 2024. lon from -121 to -119. lat from 33 to 35. type is mapped.

### SSS 8-Day Running Mean

https://harmony.earthdata.nasa.gov/C2208422957-POCLOUD/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(33:35)&subset=lon(-121:-119)&subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")&format=application%2Fx-netcdf4

### Validation procedure

Parity plot  
Bland-Altman plot  
Small summary statistics table

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

To exclude during Tchla validation:

| Flag Name | Bit Number |
| --- | --- |
| LAND | 1 |
| HIGLINT | 3 |
| HILT | 4 |
| STRAYLIGHT | 8 |
| CLDICE | 9 |
| ATMFAIL | 0 |
| LOWLW | 14 |
| FILTER | 26 |
| NAVFAIL | 25 |
| NAVWARN | 16 |