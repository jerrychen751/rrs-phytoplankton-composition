# API Usage: NASA Earthdata

This guide shows how to use RS-Kit with the NASA Earthdata plugin for discovery, query construction, and downloads. It focuses on the `nasa_earthdata` source and its public API surface.

----

## Quick start

```python
import rskit as rs
from rskit.plugins import NasaEarthdata

# 1) Store credentials in your OS keyring
rs.auth.add_credential(
    "nasa_earthdata",
    username="YOUR_EARTHDATA_USERNAME",
    password="YOUR_EARTHDATA_PASSWORD",
    token="YOUR_EARTHDATA_BEARER_TOKEN",
)

# 2) Resolve a collection concept ID (required for downloads)
plugin = NasaEarthdata()
collection_id = plugin.get_collection_concept_id(
    doi="YOUR_COLLECTION_DOI",
)

# 3) Build a query with spatial + temporal extents
query = (
    rs.query("nasa_earthdata")
    .region(bbox=(-160, 10, -150, 20))
    .time("2024-01-01", "2024-01-31")
    .with_params(
        collection_concept_id=collection_id,
        cloud_cover=(0, 20),
        max_granules=3,
        sort_key="-start_date",
    )
)

# 4) Download subsetted data (Harmony first, local fallback)
files = plugin.download_subsetted_data(
    query,
    destination="~/data/nasa",
    mask_out_of_bounds=True,
)
```

----

## Credentials

NASA Earthdata credentials are required to use the plugin. RS-Kit stores them in your OS keyring.

Required fields:
- `username`
- `password`
- `token`

Helpful APIs:
- `rs.auth.get_credential_schema("nasa_earthdata")` to inspect required fields
- `rs.auth.add_credential(...)` to store credentials
- `rs.auth.get_credentials("nasa_earthdata")` to confirm what is stored

----

## Discover collections and variables

Use the plugin for discovery before you download. Most workflows start by resolving a collection concept ID.

```python
from rskit.plugins import NasaEarthdata

nasa = NasaEarthdata()

# Resolve a collection concept ID
collection_id = nasa.get_collection_concept_id(
    short_name="COLLECTION_SHORT_NAME",
    version="COLLECTION_VERSION",
)

# Inspect collection metadata
info = nasa.get_collection_info(collection_id)

# Inspect variables available for the collection
variables = nasa.get_collection_variables(collection_id)

# Check for a specific variable name
has_var = nasa.contains_variable(collection_id, "ssha")
```

If the identifiers you provide are ambiguous, `get_collection_concept_id` raises a `ValueError` and asks you to refine the DOI/short name/version.

----

## Build a query

NASA Earthdata queries require three things:
- `Query.region(...)` with either `lon`/`lat` ranges or a `bbox`.
- `Query.time(...)` with a start and end time.
- `collection_concept_id` passed via `Query.with_params(...)`.

Example with explicit `lon`/`lat` ranges:

```python
query = (
    rs.query("nasa_earthdata")
    .region(lon=(-157.690, -132.354), lat=(14.248, 20.657))
    .time("2024-11-30", "2024-12-15")
    .with_params(collection_concept_id=collection_id)
)
```

Optional query params recognized by the NASA Earthdata plugin:
- `cloud_cover`: tuple of `(min_percent, max_percent)`.
- `sort_key`: CMR sort key (example: `"-start_date"`).
- `max_granules`: maximum number of granules to return/download.

Notes:
- `variables` and `drop_nan_lines` are included in the schema, but the current implementation does not apply them.

----

## Download data

### Download full granules (no subsetting)

```python
files = plugin.download_data(
    query,
    destination="~/data/nasa",
    limit=10,
    skip_existing=True,
)
```

Behavior:
- Downloads full granules that intersect the query spatial/temporal extents.
- Default download location is `~/Downloads/rskit-nasa_earthdata` if `destination` is not provided.
- Returns a list of file paths (as strings).

### Download subsetted data

```python
files = plugin.download_subsetted_data(
    query,
    destination="~/data/nasa",
    limit=5,
    skip_existing=True,
    mask_out_of_bounds=False,
)
```

Behavior:
- Uses NASA Harmony for server-side spatial/temporal subsetting when supported.
- Falls back to client-side subsetting if Harmony is unavailable or fails.
- `mask_out_of_bounds=True` preserves the full grid while replacing out-of-bounds cells with fill values.

You can check Harmony support for a collection:

```python
caps = plugin.list_harmony_capabilities(collection_id)
uses_harmony = plugin.supports_harmony(collection_id)
```

----

## Troubleshooting

Common errors and what they mean:
- "No credentials found for 'nasa_earthdata'": credentials were not added via `rs.auth.add_credential(...)`.
- "NASA Earthdata queries require `collection_concept_id`": include `collection_concept_id` in `Query.with_params(...)`.
- "Use .region()" or "Use .time()": you must set spatial and temporal extents before downloading.
