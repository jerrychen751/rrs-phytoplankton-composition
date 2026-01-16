# Data Info
HPLC from PACE-PAX Shearwater cruise

Date range: September 6, 2024 - September 26, 2024

Spatial range:
- Longitude: -120.0538° to -119.1739°
- Latitude: 33.6046° to 34.3736°

For a point to get matched up between in-situ and PACE:
+/- 3 hours time difference
+3x3 neighborhood median spectrum (>= 6 valid pixels)
L2 flags masking + spectral homogeneity QC (see `experiments/pax_shearwater_validation/analysis/pace_qc/notes.md`)

PACE OCI L2 Rrs matchups (current workflow)
    Use `scripts/data/download_pace_rrs.py` with the experiment config:

        python scripts/data/download_pace_rrs.py --config experiments/pax_shearwater_validation/config.yaml

    - Downloads only the granules needed to match the in-situ station times/locations.
    - Uses scanline time from `scan_line_attributes` for temporal matching (prefers `time` if present; otherwise derives from `year`/`day`/`msec`).
    - Outputs a compact NetCDF matchup dataset under:
        ~/Downloads/rrs-SDP-pigments/pax_shearwater_validation/pace_l2_rrs/

PACE OCI RRS (historical L3 mapped approach; deprecated)
    https://harmony.earthdata.nasa.gov/C3620140444-OB_CLOUD/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(33:35)&subset=lon(-121:-119)&subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")&format=application%2Fx-netcdf4

    provisional product status, PACE OCI sensor, rrs product, daily, 0.1 degree resolution, start date is sept 6 2024 to sept 26 2024. lon from -121 to -119. lat from 33 to 35. type is mapped.

    can't use browser to download; need to use script (otherwise only get 5 wavelengths)

AQUA MODIS SST
    https://harmony.earthdata.nasa.gov/C1615905770-OB_DAAC/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(33:35)&subset=lon(-121:-119)&subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")&format=application%2Fx-netcdf4

    standard product status, AQUA MODIS sensor, sst (11 micrometer) product, 4 km, daytime / daily, start date is sept 6 2024 to sept 26 2024. lon from -121 to -119. lat from 33 to 35. type is mapped.

        
SSS 8-Day Running Mean
    https://harmony.earthdata.nasa.gov/C2208422957-POCLOUD/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(33:35)&subset=lon(-121:-119)&subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")&format=application%2Fx-netcdf4


# Validation Procedure
Parity plot
Bland-Altman plot
Small summary statistics table
