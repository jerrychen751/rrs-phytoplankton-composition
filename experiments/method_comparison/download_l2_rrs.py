#!/usr/bin/env python3
"""
Download PACE OCI L2 AOP granules via CMR for all non-Kramer data sources
in the multi-source CV experiment.

Run this on a machine with internet access (HPC login node) BEFORE submitting
the L2 CV job.  Compute nodes typically lack internet.

Usage:
    python experiments/method_comparison/download_l2_rrs.py

How it works:
    For each source with an l2_rrs_dir, we:
      1. Load HPLC to get unique observation dates and the source's bbox.
      2. Search CMR for PACE L2 AOP granules covering the bbox on each date.
      3. Deduplicate granule URLs (many stations share the same overpass).
      4. Download each unique granule with Bearer token auth + fsync + atomic
         rename (Lustre-safe).
      5. Skip files that already exist on disk.

    Estimated: ~80–125 unique granules, ~25–60 GB total.

Skips:
    - Kramer: in-situ Rrs, no satellite matchup needed.
    - Any source whose l2_rrs_dir is None.
    - Granules already downloaded.

Credentials:
    NASA Earthdata login must be stored in ~/.netrc:
        machine urs.earthdata.nasa.gov login <user> password <pass>
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.method_comparison.config import SOURCES
from utils.cmr_download import (
    search_cmr_bbox,
    get_bearer_token,
    download_granule_fsync,
)
from utils.seabass_loader import load_seabass_file

import requests


def get_unique_dates(source_cfg: dict) -> list[dt.date]:
    """
    Extract unique observation dates from a source's HPLC files.

    Parses the date column (YYYYMMDD integer format) from each HPLC file
    and returns sorted unique dates.  For CSV sources (PAX), reads directly;
    for SeaBASS sources, uses the SeaBASS loader.
    """
    dates: set[dt.date] = set()

    for path_str in source_cfg["hplc_files"]:
        path = PROJECT_ROOT / path_str
        if source_cfg["hplc_format"] == "csv":
            df = pd.read_csv(path)
        else:
            df, _ = load_seabass_file(path)

        if "date" not in df.columns:
            print(f"    WARNING: no 'date' column in {path.name}")
            continue

        for val in df["date"].dropna().unique():
            try:
                d = dt.datetime.strptime(str(int(val)), "%Y%m%d").date()
                dates.add(d)
            except (ValueError, OverflowError):
                continue

    return sorted(dates)


def download_source(source_cfg: dict, session: requests.Session) -> None:
    """
    Download L2 granules for one data source.

    Searches CMR for each unique observation date using the source's bbox,
    deduplicates across dates, and downloads each unique granule.
    """
    name = source_cfg["name"]
    l2_rrs_dir_str = source_cfg.get("l2_rrs_dir")
    if not l2_rrs_dir_str:
        print(f"\nSkipping {name} (no l2_rrs_dir)")
        return

    download_cfg = source_cfg.get("download")
    if not download_cfg:
        print(f"\nSkipping {name} (no download config)")
        return

    bbox = tuple(download_cfg["bbox"])  # [lon_min, lat_min, lon_max, lat_max]
    out_dir = PROJECT_ROOT / l2_rrs_dir_str
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Source: {name}")
    print(f"  bbox: {bbox}")
    print(f"  output: {out_dir}")

    # Get unique observation dates from HPLC
    obs_dates = get_unique_dates(source_cfg)
    print(f"  {len(obs_dates)} unique observation dates")

    if not obs_dates:
        print("  No dates found — skipping")
        return

    # Search CMR for all dates, deduplicate granule URLs
    all_granules: dict[str, dict] = {}  # url -> {title, url}
    for obs_date in obs_dates:
        granules = search_cmr_bbox(bbox, obs_date)
        for gran in granules:
            if gran["url"] not in all_granules:
                all_granules[gran["url"]] = gran

    print(f"  {len(all_granules)} unique L2 granules found across all dates")

    if not all_granules:
        print("  No granules found via CMR — source may predate PACE first light")
        return

    # Download with a 2-second delay between files to avoid server rate-limiting.
    # NASA OB.DAAC throttles rapid sequential downloads; a short pause keeps
    # us under the threshold while still finishing in a reasonable time.
    INTER_DOWNLOAD_DELAY_S = 2.0

    saved = skipped = failed = 0
    n_total = len(all_granules)
    for idx, (url, gran) in enumerate(all_granules.items(), 1):
        fname = url.rsplit("/", 1)[-1]
        dest = out_dir / fname

        if dest.exists():
            skipped += 1
            continue

        print(f"  [{idx}/{n_total}] Downloading {fname}...", end=" ", flush=True)
        if download_granule_fsync(url, dest, session):
            size_mb = dest.stat().st_size / (1024**2)
            print(f"OK ({size_mb:.0f} MB)")
            saved += 1
        else:
            print("FAILED (after retries)")
            failed += 1

        time.sleep(INTER_DOWNLOAD_DELAY_S)

    print(f"  Done: {saved} saved, {skipped} already existed, {failed} failed")


def main() -> None:
    # Authenticate once, reuse the session for all downloads
    print("Obtaining Earthdata Bearer token...")
    token = get_bearer_token()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    for source_cfg in SOURCES:
        if source_cfg["hplc_format"] == "kramer":
            print(f"\nSkipping {source_cfg['name']} (in-situ Rrs, no satellite matchup)")
            continue
        download_source(source_cfg, session)

    print("\nAll sources complete.")
    print("Next step: submit run_multi_source_cv_l2.sbatch")


if __name__ == "__main__":
    main()
