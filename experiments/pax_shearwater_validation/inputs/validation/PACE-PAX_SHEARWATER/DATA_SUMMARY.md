# PACE-PAX_SHEARWATER Data Summary

## Overview

This directory contains in-situ validation data collected during the PACE-PAX Shearwater cruise (September 2024) for validating PACE satellite data and the SDP pigment prediction model.

## Key Data Files

### 1. HPLC Pigment Data

**Location:** `archive/PACE-PAX_SHEARWATER_HPLC_NASA_R1.sb`

- **Format:** SeaBASS (.sb) file
- **Content:** HPLC-measured pigment concentrations for validation
- **Fields:** All 13 pigments (Tchla, Zea, DVchla, ButFuco, HexFuco, Allo, MVchlb, Neo, Viola, Fuco, Chlc12, Chlc3, Perid) plus additional pigments
- **Spatial Coverage:** 33.56°N to 34.37°N, 120.05°W to 118.06°W
- **Temporal Coverage:** September 6-26, 2024
- **Stations:** Multiple stations with replicates
- **Missing values:** -9999 (missing), -8888 (below detection limit)

### 2. In-Situ Optical Data

**Location:** `documents/docs_ACSprofile20250916/noscatcor/`

- **Files:** 36 SeaBASS files (PACE-PAX_RV_Shearwater_acs017_stXX_noscatcor.sb)
- **Format:** SeaBASS (.sb) files
- **Content:**
  - **cgp** fields: Colored dissolved organic matter + particle absorption (1/m) at wavelengths 399.8-744.6 nm
  - **agp** fields: Absorption (1/m) at wavelengths 400.7-743.4 nm
  - Depth profiles with temperature, salinity, conductivity
- **Note:** These are absorption/backscattering profiles, NOT RRS data directly
- **Stations:** st02 through st40 (some stations missing)

### 3. Other Available Data

- **ACS profiles:** Absorption/backscattering coefficient profiles
- **CDOM data:** Colored dissolved organic matter absorption
- **DOC data:** Dissolved organic carbon
- **POC data:** Particulate organic carbon
- **HyperBB data:** Hyperspectral backscattering
- **Radiometry data:** Above-water radiometry (AWR) - may contain RRS data

## Important Notes

### RRS Data Location

The ACS profile files contain absorption/backscattering data but **NOT** remote sensing reflectance (RRS) directly. For in-situ RRS comparison, you may need to:

1. Look for above-water radiometry (AWR) files in the documents folder
2. Check if RRS can be derived from the absorption/backscattering data
3. Look for files processed by HyperCP (mentioned in README.md) that would contain L2 RRS data

### Data Matching Strategy

To compare PACE satellite RRS with in-situ data:

1. **Spatial matching:** Match PACE pixels to in-situ locations (lat/lon from HPLC file)
2. **Temporal matching:** Match PACE overpass times to in-situ measurement times
3. **Spectral matching:** Interpolate in-situ hyperspectral RRS to PACE OCI wavelengths (if needed)

### Model Validation Strategy

To compare model predictions with HPLC:

1. Extract PACE RRS at in-situ locations/times
2. Run SDP model on PACE RRS data
3. Match model predictions to HPLC measurements by:
   - Station number
   - Date/time
   - Location (lat/lon)
4. Calculate validation metrics (R², RMSE, bias, etc.)

## File Structure

```
PACE-PAX_SHEARWATER/
├── archive/
│   └── PACE-PAX_SHEARWATER_HPLC_NASA_R1.sb  # HPLC pigment data
├── documents/
│   ├── docs_ACSprofile20250916/
│   │   └── noscatcor/  # 36 ACS profile files
│   ├── docs_hyperBB20250428/  # Backscattering data
│   ├── docs_CDOM20250730/  # CDOM absorption
│   ├── docs_DOC20250627/  # DOC data
│   ├── docs_POC20250830/  # POC data
│   └── [other optical data folders]
└── documents.tgz  # Compressed archive
```

## Next Steps

1. **Locate in-situ RRS data:** Check for AWR (above-water radiometry) files or HyperCP-processed L2 files
2. **Load HPLC data:** Parse SeaBASS file to extract pigment concentrations
3. **Load PACE satellite data:** Extract RRS at matching locations/times
4. **Run comparisons:**
   - PACE RRS vs. in-situ RRS
   - Model predictions vs. HPLC measurements
