# In-Situ HPLC Pigment Data

In-situ HPLC pigment concentration measurements for SDP model validation from the PACE-PAX Shearwater cruise (September 2024).

## Files

- **PACE-PAX_SHEARWATER_HPLC_NASA_R1.sb** - Raw SeaBASS format (36 stations, 108 samples, Sept 6-26, 2024, ~34°N, 119-120°W)
- **PACE-PAX_SHEARWATER_HPLC_NASA_R1.csv** - Full dataset (all columns)
- **PACE-PAX_SHEARWATER_HPLC_pigments.csv** - Pigment concentrations only (Tot_Chl_a, But-fuco, Hex-fuco, Allo, Fuco, Perid, Zea, DV_Chl_a, MV_Chl_b, Chl_c1c2, Chl_c3, Neo, Viola, Tchl) plus metadata (station, date, time, lon, lat)

## Pigments

All 13 SDP model pigments are available: Tot_Chl_a, Zea, DV_Chl_a, But-fuco, Hex-fuco, Allo, MV_Chl_b, Neo, Viola, Fuco, Chl_c1c2, Chl_c3, Perid. Additional pigments (Tchl, alpha-beta-Car, Diadino, Diato) are also included.

## Data Quality

- Missing values: NaN (originally -9999 or -8888)
- Below detection limit: NaN (originally -8888)
- Replicates: Typically 3 per station
- Units: mg/m³

## Validation Workflow

1. Load HPLC data
2. Extract PACE satellite RRS at matching locations/times
3. Run SDP model on PACE RRS data
4. Match predictions to HPLC by station/date/location
5. Calculate metrics (R², RMSE, bias)

## Source

PACE-PAX_SHEARWATER cruise, NASA GSFC (Crystal Thomas), Agilent RR1200, DOI: 10.5067/SeaBASS/PACE-PAX/DATA001
