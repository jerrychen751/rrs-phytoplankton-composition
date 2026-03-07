#!/usr/bin/env python3
"""
Temporary helper: download SST + SSS inputs for PVST SOPACE validation.

Goal
----
Download gridded granules.

- For SST: download full granules, then perform **local bbox subsetting** with
  xarray immediately after each file is downloaded.
- For SSS: download via Harmony with bbox/time constraints (8-day granules only)
  and keep the downloaded output as-is (no local subsetting).

Download methods:
- SST: `earthaccess` (CMR search + download) full granules
- SSS: NASA Harmony python library (`harmony`) (8-day granules only; no local subsetting)

This intentionally does NOT use:
- RS-Kit for downloads or subsetting

Why
----
`main.py` runs a PACE OCI L2 (swath) matchup pipeline, but it still needs gridded
SST/SSS products to sample values at each matchup point.

The local subsetting logic was written by inspecting the structure of:
`experiments/pvst_sopace_validation/inputs/sss/SMAP_L3_SSS_20250524_8DAYS_V5.0.nc`
which has dims `(latitude, longitude, time)` and uses descending latitude.

Run (recommended)
----
  conda activate swot-pace
  python3 scripts/data/tmp_download_pvst_sst_sss.py --region hplc_2025_10_14 --start 2025-05-06 --end 2025-06-15

Authentication:
- This script expects `EARTHDATA_TOKEN` to be defined in `.env` at the repo root
  (`/Users/jerry/coding/rrs-SDP-pigments/.env`). It loads that file at runtime
  if the token is not already present in your process environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PVST_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "pvst_sopace_validation"
DOTENV_PATH = PROJECT_ROOT / ".env"

# Region info to subset (west, south, east, north)
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "hplc_2024_11_30": (-157.6904, -14.2475, -132.3535, 20.657),
    "hplc_2025_10_14": (85.9995703, 9.9134, 88.5087, 13.94745),
    "hplc_2025_11_03": (148.208, 7.651, 149.03, 15.0),
}

# Concept IDs from `docs/notes.md`
SST_COLLECTION_CONCEPT_ID = "C1615905770-OB_DAAC"
SSS_COLLECTION_CONCEPT_ID = "C2208422957-POCLOUD"

SST_DEST = PVST_EXPERIMENT_DIR / "inputs" / "sst"
SSS_DEST = PVST_EXPERIMENT_DIR / "inputs" / "sss"


def _load_dotenv(path: Path) -> None:
    """
    Minimal `.env` loader (avoids adding a python-dotenv dependency).

    - Does not overwrite already-defined environment variables.
    - Supports simple `KEY=VALUE` lines and optional leading `export `.
    - Strips matching single/double quotes around values.
    """
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def _get_earthdata_token() -> str:
    _load_dotenv(DOTENV_PATH)
    token = os.getenv("EARTHDATA_TOKEN") or os.getenv("EARTHACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            f"Missing Earthdata bearer token. Expected EARTHDATA_TOKEN in the environment or in {DOTENV_PATH}.\n"
            "Set EARTHDATA_TOKEN (preferred) or EARTHACCESS_TOKEN.\n"
            "Example:\n"
            "  export EARTHDATA_TOKEN='...'\n"
        )
    return token


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _infer_lat_lon_names(ds: object) -> tuple[str, str]:
    """
    Infer latitude/longitude coordinate names from common gridded NetCDF layouts.
    """
    coords = getattr(ds, "coords", {})
    variables = getattr(ds, "variables", {})

    lat_candidates = ("latitude", "lat")
    lon_candidates = ("longitude", "lon")

    lat_name = next((n for n in lat_candidates if n in coords or n in variables), None)
    lon_name = next((n for n in lon_candidates if n in coords or n in variables), None)
    if lat_name is None or lon_name is None:
        raise KeyError(
            "Unable to infer lat/lon coordinate names. "
            f"coords={list(coords)} vars={list(variables)[:10]}"
        )
    return lat_name, lon_name


def _subset_dataset_to_bbox(*, ds: object, bbox: tuple[float, float, float, float]) -> object:
    """
    Subset a gridded dataset to bbox=(west, south, east, north) using `.sel(...)`.

    Handles descending latitude coordinates (SMAP SSS uses descending latitude).
    """
    import numpy as np

    lat_name, lon_name = _infer_lat_lon_names(ds)
    west, south, east, north = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    lat = getattr(ds, "__getitem__")(lat_name).values
    lon = getattr(ds, "__getitem__")(lon_name).values

    lat_asc = bool(lat[0] < lat[-1])
    lon_asc = bool(lon[0] < lon[-1])

    lon_min = float(np.nanmin(lon))
    if lon_min >= 0.0 and west < 0.0:
        # Dataset uses [0, 360] but bbox uses [-180, 180]
        west = (west + 360.0) % 360.0
        east = (east + 360.0) % 360.0

    lat_slice = slice(south, north) if lat_asc else slice(north, south)
    lon_slice = slice(west, east) if lon_asc else slice(east, west)

    # None of the provided PVST bboxes cross the dateline, so a simple lon slice is fine.
    return getattr(ds, "sel")({lat_name: lat_slice, lon_name: lon_slice})


def _subset_file(*, input_path: Path, bbox: tuple[float, float, float, float], output_path: Path) -> None:
    """
    Read input_path, subset to bbox, and write output_path atomically.
    """
    import xarray as xr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".part")
    tmp.unlink(missing_ok=True)

    with xr.open_dataset(input_path) as ds:
        subset = _subset_dataset_to_bbox(ds=ds, bbox=bbox)
        subset.to_netcdf(tmp)

    tmp.replace(output_path)


def _download_sst_via_earthaccess(
    *,
    collection_concept_id: str,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    destination: Path,
) -> list[Path]:
    """
    Download SST granules via `earthaccess`, then subset in-place immediately.

    We still run the local subsetting code (rather than server-side subsetting)
    so the output structure matches what the downstream pipeline expects.
    """
    import earthaccess
    import inspect

    _load_dotenv(DOTENV_PATH)
    earthaccess.login(strategy="environment")

    destination.mkdir(parents=True, exist_ok=True)
    #
    # NOTE: `earthaccess.search_data(...)` requires a *collection filter* when
    # spatial parameters are provided. Our "collection_concept_id" is a CMR
    # **collection** concept-id (e.g., "C1615905770-OB_DAAC"), which most
    # earthaccess versions accept via the `concept_id` argument.
    #
    # Some earthaccess versions accept arbitrary kwargs and silently ignore
    # unknown names, so we defensively choose the right parameter name by
    # inspecting the function signature.
    sig = inspect.signature(earthaccess.search_data)
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if "concept_id" in sig.parameters or has_kwargs:
        concept_kw = "concept_id"
    elif "collection_concept_id" in sig.parameters:
        concept_kw = "collection_concept_id"
    else:
        raise RuntimeError(
            "Unable to determine how to pass a collection filter into `earthaccess.search_data(...)`.\n"
            "This call must include a collection constraint when using `bounding_box=...`.\n"
            "Fix by upgrading earthaccess or updating this script to use `short_name=...` / "
            "`entry_title=...` for the SST collection.\n"
            f"Attempted SST collection concept-id: {collection_concept_id}"
        )

    granules = earthaccess.search_data(
        **{
            concept_kw: collection_concept_id,
            "bounding_box": bbox,
            "temporal": (str(start), str(end)),
            "count": -1,
        }
    )
    if not granules:
        print(f"No SST granules found for collection {collection_concept_id} in {start}..{end}")
        return []

    # PVST workflow wants *only* 8-day composites at 4 km resolution.
    #
    # CMR search returns a mix of products (DAY, 8D, R32, MO, climatologies, ...)
    # and often both 4km and 9km links. Avoid downloading unwanted granules by
    # extracting and filtering the data links ourselves.
    want_rx = re.compile(r"^AQUA_MODIS\.\d{8}_\d{8}\.L3m\.8D\.SST\.sst\.4km\.nc$")

    wanted_urls: set[str] = set()
    for granule in granules:
        try:
            urls = granule.data_links()
        except Exception:
            urls = []
        for url in urls:
            filename = str(url).split("?", 1)[0].rsplit("/", 1)[-1]
            if want_rx.match(filename):
                wanted_urls.add(str(url))

    if not wanted_urls:
        print(
            "No SST download links matched the 8D 4km pattern.\n"
            f"Expected filenames like: AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.8D.SST.sst.4km.nc\n"
            f"Collection concept-id: {collection_concept_id}"
        )
        return []

    wanted_list = sorted(wanted_urls)
    print(f"Found {len(wanted_list)} SST files matching 8D 4km; downloading...")

    downloaded: list[Path] = []
    paths = earthaccess.download(wanted_list, local_path=str(destination), threads=1)
    for path in paths:
        file_path = Path(path)
        downloaded.append(file_path)
        if file_path.suffix.lower() not in {".nc", ".nc4"}:
            print(f"  [skip] Unsupported SST file extension (expected .nc/.nc4): {file_path.name}")
            continue

        print(f"  Subsetting SST in-place -> {file_path.name}")
        _subset_file(input_path=file_path, bbox=bbox, output_path=file_path)

    return downloaded


def _download_sss_via_harmony(
    *,
    collection_concept_id: str,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    destination: Path,
) -> list[Path]:
    """
    Download SSS using NASA's Harmony python client and return the downloaded file paths.

    For SSS we intentionally do **not** locally subset after download. Harmony
    already supports server-side spatial/temporal constraints, and we want to
    preserve the on-disk output exactly as Harmony produces it.

    This request is also restricted to 8-day products via `granule_name=["*8DAYS*"]`.
    """
    import datetime as dtmod

    import harmony

    token = _get_earthdata_token()
    destination.mkdir(parents=True, exist_ok=True)

    client = harmony.Client(token=token)
    request = harmony.Request(
        harmony.Collection(collection_concept_id),
        spatial=harmony.BBox(*bbox),
        temporal={
            "start": dtmod.datetime.combine(start, dtmod.time.min, tzinfo=dtmod.timezone.utc),
            "end": dtmod.datetime.combine(end, dtmod.time.max, tzinfo=dtmod.timezone.utc),
        },
        granule_name=["*8DAYS*"],
    )

    job_id = client.submit(request)
    futures = list(client.download_all(job_id, directory=str(destination), overwrite=False))

    downloaded: list[Path] = []
    for future in concurrent.futures.as_completed(futures):
        path = Path(future.result())
        downloaded.append(path)
        if path.suffix.lower() not in {".nc", ".nc4"}:
            print(f"  [skip] Unsupported SSS file extension (expected .nc/.nc4): {path.name}")
            continue
    return downloaded


def _list_netcdf_files(directory: Path) -> list[Path]:
    """
    Non-recursive collection of NetCDF files in a directory.
    """
    files: set[Path] = set()
    for pattern in ("*.nc", "*.nc4"):
        files.update({p.resolve() for p in directory.glob(pattern) if p.is_file()})
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=sorted(REGIONS), default="hplc_2025_10_14")
    parser.add_argument("--start", default="2025-05-06", help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default="2025-06-15", help="YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    region_key = str(args.region)
    bbox = REGIONS[region_key]
    start_date = _parse_date(str(args.start))
    end_date = _parse_date(str(args.end))
    if start_date > end_date:
        raise ValueError(f"Invalid date range: start={start_date} must be <= end={end_date}")

    print(f"Region: {region_key}")
    print(f"Subset bbox (west, south, east, north): {bbox}")
    print(f"Time window: {start_date} .. {end_date}")

    _load_dotenv(DOTENV_PATH)

    # --- SST
    sst_paths = _download_sst_via_earthaccess(
        collection_concept_id=SST_COLLECTION_CONCEPT_ID,
        bbox=bbox,
        start=start_date,
        end=end_date,
        destination=SST_DEST,
    )
    _ = sst_paths

    # --- SSS
    # Download via Harmony (library is available in `swot-pace`).
    sss_paths = _download_sss_via_harmony(
        collection_concept_id=SSS_COLLECTION_CONCEPT_ID,
        bbox=bbox,
        start=start_date,
        end=end_date,
        destination=SSS_DEST,
    )
    _ = sss_paths

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
