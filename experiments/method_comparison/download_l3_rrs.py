#!/usr/bin/env python3
"""
Download PACE OCI L3 0.1-degree daily Rrs files via OPeNDAP for all
non-PAX data sources in the multi-source CV experiment.

Run this on a machine with internet access (locally or on the HPC login node)
BEFORE running multi_source_cv.py. The compute node may not have internet.

Usage:
    python experiments/method_comparison/download_l3_rrs.py

Credentials:
    NASA Earthdata login must be stored in ~/.netrc:
        machine urs.earthdata.nasa.gov login <user> password <pass>
    See: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget

How it works:
    OPeNDAP (OPen-source Project for a Network Data Access Protocol) lets us
    open a remote NetCDF file URL with xarray and immediately apply .sel() to
    request only the spatial subset we want. The server returns just the
    subsetted bytes — no need to download the full ~400MB global file.

    L3 latitude runs 90 → -90 (descending), so spatial subsets need
    slice(lat_max, lat_min) rather than the usual slice(min, max).

Skips:
    - pax_shearwater: already has L3 Rrs files in its own directory.
    - Any date whose output file already exists.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.method_comparison.config import SOURCES

# OB.DAAC OPeNDAP root for PACE L3 mapped (L3SMI) products.
OPENDAP_BASE = "https://oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/L3SMI"

# Sources whose rrs_dir is already populated and should not be re-downloaded.
SKIP_SOURCES = {"pax_shearwater"}


def build_opendap_url(date: dt.date) -> str:
    """
    Build the OPeNDAP URL for a PACE OCI L3 0.1-deg daily Rrs file.

    OB.DAAC directory tree: /{YYYY}/{MMDD}/filename.nc
    The filename follows the convention:
      PACE_OCI.{YYYYMMDD}.L3m.DAY.RRS.V3_1.Rrs.0p1deg.nc
    """
    mmdd = date.strftime("%m%d")
    yyyymmdd = date.strftime("%Y%m%d")
    return (
        f"{OPENDAP_BASE}/{date.year}/{mmdd}/"
        f"PACE_OCI.{yyyymmdd}.L3m.DAY.RRS.V3_1.Rrs.0p1deg.nc"
    )


def subset_and_save(
    ds: xr.Dataset,
    out_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """
    Spatially subset an open PACE L3 dataset and write to disk.

    L3 lat coordinate is descending (90 → -90), so we slice from max to min.
    Writes a temporary file then atomically renames to avoid partial downloads.
    zlib compression at level 4 gives ~3x size reduction on NaN-heavy arrays.
    """
    # Descending lat: slice(high, low)
    subset = ds.sel(
        lat=slice(lat_range[1], lat_range[0]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    # "palette" is a visualization artifact in some L3 files — not useful data.
    if "palette" in subset:
        subset = subset.drop_vars("palette")
    subset.load()   # Pull subset data from remote server into memory

    tmp = out_path.with_suffix(".tmp.nc")
    encoding = {v: {"zlib": True, "complevel": 4} for v in subset.data_vars}
    subset.to_netcdf(tmp, encoding=encoding)
    # fsync forces dirty pages from the Lustre client cache to the OSTs before
    # the rename makes the file visible. Without this, HDF5 metadata checksums
    # can mismatch on subsequent reads when the cache hasn't fully flushed.
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.rename(out_path)   # Atomic: readers never see a partial file


def download_source(source_cfg: dict) -> None:
    """
    Download L3 Rrs files for one data source.

    Iterates day by day over the source's configured date_range, building the
    OPeNDAP URL for each day, opening via xr.open_dataset (lazy remote read),
    subsetting spatially, and saving locally. Days with no server-side file
    (e.g., near-real-time gaps) raise OSError and are logged as "No data".
    """
    name = source_cfg["name"]
    dl = source_cfg["download"]
    bbox = dl["bbox"]          # [lon_min, lat_min, lon_max, lat_max]
    lon_range = (bbox[0], bbox[2])
    lat_range = (bbox[1], bbox[3])

    start = dt.datetime.strptime(dl["date_range"][0], "%Y-%m-%d").date()
    end   = dt.datetime.strptime(dl["date_range"][1], "%Y-%m-%d").date()

    out_dir = PROJECT_ROOT / source_cfg["rrs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Source: {name}")
    print(f"  lon {lon_range[0]} to {lon_range[1]}, lat {lat_range[0]} to {lat_range[1]}")
    print(f"  {start} → {end}  ({(end - start).days + 1} days)")
    print(f"  output: {out_dir}")
    print(f"{'='*60}")

    current = start
    saved = skipped = missing = errors = 0

    while current <= end:
        fname = f"PACE_OCI.{current.strftime('%Y%m%d')}.L3m.DAY.RRS.V3_1.Rrs.0p1deg.nc"
        out_path = out_dir / fname

        if out_path.exists():
            skipped += 1
            current += dt.timedelta(days=1)
            continue

        url = build_opendap_url(current)
        try:
            # xr.open_dataset with OPeNDAP URL is lazy — no data transferred
            # until .load() is called inside subset_and_save.
            with xr.open_dataset(url, engine="netcdf4") as ds:
                subset_and_save(ds, out_path, lon_range, lat_range)
            size_mb = out_path.stat().st_size / (1024 ** 2)
            saved += 1
            print(f"  Saved {fname} ({size_mb:.1f} MB)")
        except OSError:
            # Server returns 404 or similar when no granule exists for that day.
            missing += 1
            print(f"  No data: {current}")
        except Exception as exc:
            errors += 1
            print(f"  Error on {current}: {exc}")

        current += dt.timedelta(days=1)

    print(
        f"  Done: {saved} saved, {skipped} already existed, "
        f"{missing} missing on server, {errors} errors"
    )


def main() -> None:
    for source_cfg in SOURCES:
        if source_cfg["name"] in SKIP_SOURCES:
            print(f"\nSkipping {source_cfg['name']} (L3 Rrs already present)")
            continue
        if source_cfg.get("download") is None:
            print(f"\nSkipping {source_cfg['name']} (no download config)")
            continue
        download_source(source_cfg)

    print("\nAll sources complete.")
    print("Next step: run download_sst_sss.py, then submit run_multi_source_cv.sbatch")


if __name__ == "__main__":
    main()
