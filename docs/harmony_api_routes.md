**Does not work for PACE / AQUA MODIS in terms of subsetting data beforehand.**

URL STRUCTURE BREAKDOWN:
https://harmony.earthdata.nasa.gov/{Concept-ID}/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?{Parameters}

1. Concept-ID (Base Path)
   - Format: Alphanumeric string starting with 'C' (e.g., C3620140444-OB_CLOUD)
   - Purpose: Unique identifier for the specific NASA dataset/collection you are querying.

2. subset=lat(min:max)
   - Format: subset=lat(float:float)
   - Example: subset=lat(33:35)
   - Note: Defines the spatial bounding box latitude range in decimal degrees.

3. subset=lon(min:max)
   - Format: subset=lon(float:float)
   - Example: subset=lon(-121:-119)
   - Note: Defines the spatial bounding box longitude range in decimal degrees.

4. subset=time("start":"end")
   - Format: subset=time("YYYY-MM-DDTHH:MM:SSZ":"YYYY-MM-DDTHH:MM:SSZ")
   - Example: subset=time("2024-09-06T00:00:00Z":"2024-09-26T23:59:59Z")
   - Note: Uses ISO 8601 format enclosed in double quotes. The 'Z' indicates UTC time.

5. format
   - Format: MIME type (often URL-encoded)
   - Example: format=application%2Fx-netcdf4
   - Common Values: 
     * application/x-netcdf4 (for NetCDF)
     * image/tiff (for GeoTIFF)
     * image/png (for preview images)

PACE OCI RRS
https://harmony.earthdata.nasa.gov/C3620140444-OB_CLOUD/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(min_lat:max_lat)&subset=lon(min_lon:max_lon)&subset=time("start_time":"end_time")&format=application%2Fx-netcdf4

AQUA MODIS SST
https://harmony.earthdata.nasa.gov/C1615905770-OB_DAAC/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(min_lat:max_lat)&subset=lon(min_lon:max_lon)&subset=time("start_time":"end_time")&format=application%2Fx-netcdf4

SSS 8-Day Running Mean
https://harmony.earthdata.nasa.gov/C2208422957-POCLOUD/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(min_lat:max_lat)&subset=lon(min_lon:max_lon)&subset=time("start_time":"end_time")&format=application%2Fx-netcdf4