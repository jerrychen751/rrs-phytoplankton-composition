"""
Download PACE OCI Level 2 BGC (Biogeochemical) data from NASA CMR API.

This module downloads PACE OCI Level 2 BGC products.
It targets the 'PACE_OCI_L2_BGC' collection and can optionally extract the 'chlor_a' variable.

Files are saved directly in the specified output directory.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
import xarray as xr


# Authentication
#
# This downloader expects an Earthdata access token in the environment.
# Avoid hard-coding secrets in the repo.
def _get_token() -> Optional[str]:
    # Prefer EARTHDATA_TOKEN (matches earthaccess convention), but support the
    # repo legacy env var name as well.
    return os.getenv("EARTHDATA_TOKEN") or os.getenv("EARTHACCESS_TOKEN")

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config_from_file, get_bbox, get_time_range, get_input_dir

# CMR API endpoints
CMR_GRANULE_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
CMR_COLLECTION_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"

# PACE OCI Level 2 BGC dataset identifiers
PACE_COLLECTIONS = {
    "L2_BGC": {
        "short_names": ["PACE_OCI_L2_BGC"],
    },
}
DEFAULT_COLLECTION_KEY = "L2_BGC"


def get_session() -> requests.Session:
    """Create an authenticated requests session."""
    session = requests.Session()
    token = _get_token()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def search_collections(session: requests.Session, short_name: str) -> Optional[dict]:
    """Search for a collection by short name."""
    params = {
        "short_name": short_name,
        "page_size": 1
    }
    try:
        response = session.get(CMR_COLLECTION_SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json()
        entries = results.get('feed', {}).get('entry', [])
        if entries:
            return entries[0]
    except Exception as e:
        print(f"Error searching for collection {short_name}: {e}")
    return None


def find_pace_collection(
    session: requests.Session,
    collection_key: str = DEFAULT_COLLECTION_KEY,
) -> Optional[str]:
    """Find the correct PACE OCI collection."""
    collection_cfg = PACE_COLLECTIONS.get(collection_key)
    if not collection_cfg:
        print(f"Unknown collection key '{collection_key}'. Available keys: {list(PACE_COLLECTIONS)}")
        return None

    concept_id = collection_cfg.get("concept_id")
    if concept_id:
        print(f"Using predefined concept ID for {collection_key}: {concept_id}")
        return concept_id

    print(f"Searching for PACE OCI collection ({collection_key})...")
    for short_name in collection_cfg.get("short_names", []):
        collection = search_collections(session, short_name)
        if collection:
            collection_id = collection.get('id')
            print(f"Found collection: {short_name} (ID: {collection_id})")
            return collection_id
    print(f"Unable to locate collection for key '{collection_key}'.")
    return None


def search_granules(
    tspan: Tuple[str, str],
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    collection_concept_id: str,
    clouds: Optional[Tuple[int, int]] = None,
    sort_key: Optional[str] = None,
) -> dict:
    """
    Search for specific granules within a collection using NASA's Common Metadata Repository (CMR) API given a set of conditions.
    """
    # Validate lon_range
    if not (
        isinstance(lon_range, (tuple, list))
        and len(lon_range) == 2
        and all(isinstance(x, (float, int)) for x in lon_range)
    ):
        raise ValueError(
            "lon_range must be a tuple or list of two floats or ints: (lon_min, lon_max)."
        )

    # Validate lat_range
    if not (
        isinstance(lat_range, (tuple, list))
        and len(lat_range) == 2
        and all(isinstance(x, (float, int)) for x in lat_range)
    ):
        raise ValueError(
            "lat_range must be a tuple or list of two floats or ints: (lat_min, lat_max)."
        )

    # Validate tspan
    if not (
        isinstance(tspan, (tuple, list))
        and len(tspan) == 2
        and all(isinstance(d, str) for d in tspan)
    ):
        raise ValueError(
            "tspan must be a tuple or list of two strings: (start_date, end_date) in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:mm:ssZ' format."
        )

    if re.match(r"^\d{4}-\d{2}-\d{2}$", tspan[0]) and re.match(r"^\d{4}-\d{2}-\d{2}$", tspan[1]):
        tspan = (f"{tspan[0]}T00:00:00Z", f"{tspan[1]}T23:59:59Z")

    # Make the request
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"

    # Construct bbox from lon_range and lat_range
    bbox = (lon_range[0], lat_range[0], lon_range[1], lat_range[1])
    
    # Format temporal as comma-separated string for CMR API
    temporal_str = f"{tspan[0]},{tspan[1]}"
    
    params = {
        "collection_concept_id": collection_concept_id,
        "temporal": temporal_str,
        "bounding_box": ",".join(str(float(x)) for x in bbox),
        "page_size": 2000,
    }

    if clouds and len(clouds) == 2 and all(isinstance(x, (float, int)) for x in clouds):
        params["cloud_cover"] = f"{clouds[0]},{clouds[1]}"
    
    if sort_key:
        params["sort_key"] = sort_key
    
    all_entries = []
    headers = {}
    token = _get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    while True:
        try:
            response = requests.get(cmr_url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            results = response.json()
            
            entries = results.get("feed", {}).get("entry", [])
            all_entries.extend(entries)

            # Check for pagination token
            search_after = response.headers.get("CMR-Search-After")
            
            if search_after and entries:
                headers["CMR-Search-After"] = search_after
            else:
                break
        except requests.RequestException as req_err:
            raise requests.RequestException(f"HTTP request to CMR failed: {req_err}") from req_err
        except json.JSONDecodeError as json_err:
            raise json.JSONDecodeError(f"Failed to decode JSON response: {json_err}", doc=response.text, pos=0) from json_err

    return {"feed": {"entry": all_entries}}


def get_download_urls(granule: dict) -> List[str]:
    """
    Extracts direct download URLs for granule data files from NASA CMR search results.
    """
    download_urls = []
    links = granule.get("links", [])
    
    # First, try strict criteria: data link with "download" in title
    for link in links:
        rel = link.get("rel", "")
        title = link.get("title", "").lower()
        href = link.get("href", "")
        
        # Look for data download links (not OPeNDAP or other service links)
        if rel == "http://esipfed.org/ns/fedsearch/1.1/data#" and "download" in title:
            if href and "/opendap/" not in href.lower():
                download_urls.append(href)
    
    # Fallback: if no strict matches, look for any data link that's not OPeNDAP
    if not download_urls:
        for link in links:
            rel = link.get("rel", "")
            href = link.get("href", "")
            
            # Check for data link types
            is_data_link = (
                rel == "http://esipfed.org/ns/fedsearch/1.1/data#" or
                rel == "https://api.earthdata.nasa.gov/edsc/1.1/data#" or
                "data" in rel.lower()
            )
            
            if is_data_link and href:
                # Exclude OPeNDAP URLs which require different authentication
                if "/opendap/" not in href.lower() and (href.endswith('.nc') or href.endswith('.nc4')):
                    download_urls.append(href)
    
    download_urls.sort()
    return download_urls


def download_file(
    session: requests.Session,
    url: str,
    output_path: str,
    chunk_size: int = 8192
) -> bool:
    """
    Download a file from URL to output path.
    """
    try:
        # Check if file already exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            # Check file size via HEAD request
            head_response = session.head(url, timeout=30)
            if head_response.status_code == 200:
                remote_size = int(head_response.headers.get('Content-Length', 0))
                if file_size == remote_size and file_size > 0:
                    print(f"  File already exists: {os.path.basename(output_path)}")
                    return True
        
        # Download file
        response = session.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Downloading: {percent:.1f}%", end='', flush=True)
        
        print(f"\n  Downloaded: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"\n  Error downloading {url}: {e}")
        return False


def extract_chlor_a(
    source_path: Path,
    output_dir: Path,
    keep_original: bool = True,
    compression_level: int = 4,
) -> Optional[str]:
    """
    Extract chlor_a variable from the downloaded granule.
    
    Returns the path to the extracted file.
    """
    try:
        # Open dataset
        # L2 files often have groups. 'chlor_a' is usually in 'geophysical_data' group or top level.
        # We'll try opening normally first.
        try:
            ds = xr.open_dataset(source_path)
            # Check if chlor_a is in data_vars
            if 'chlor_a' not in ds.data_vars:
                # Try groups
                ds.close()
                ds = xr.open_dataset(source_path, group='geophysical_data')
                if 'chlor_a' not in ds.data_vars:
                    print(f"  'chlor_a' not found in {source_path.name}")
                    ds.close()
                    return None
                
                # Also need navigation data
                ds_nav = xr.open_dataset(source_path, group='navigation_data')
                ds = ds.merge(ds_nav)
                ds_nav.close()
        except Exception as e:
            print(f"  Error opening {source_path.name}: {e}")
            return None

        # Create subset with just chlor_a and coordinates
        ds_subset = ds[['chlor_a']]
        
        # Save
        base_name = source_path.stem
        
        out_name = f"{base_name}_chlor_a.nc"
        out_path = output_dir / out_name
        
        encoding = {'chlor_a': {'zlib': True, 'complevel': compression_level}}
        ds_subset.to_netcdf(out_path, encoding=encoding)
        print(f"  Extracted chlor_a -> {out_path}")
        
        ds.close()
        
        if not keep_original and source_path.exists():
            try:
                source_path.unlink()
                print(f"  Removed original file {source_path.name}")
            except OSError as exc:
                print(f"  Could not remove original file {source_path.name}: {exc}")
                
        return str(out_path)

    except Exception as e:
        print(f"  Error processing {source_path.name}: {e}")
        return None


def download_pace_bgc(
    lon_range: Tuple[float, float] = (-121, -119),
    lat_range: Tuple[float, float] = (33, 35),
    time_range: Tuple[str, str] = ("2024-09-06", "2024-09-26"),
    output_dir: str = "experiments/seabass_validation/inputs/pace_bgc",
    collection_key: str = DEFAULT_COLLECTION_KEY,
    extract_chla: bool = False,
) -> List[str]:
    """
    Download PACE OCI Level 2 BGC data.
    """
    print(f"Downloading PACE OCI Level 2 BGC data")
    
    session = get_session()
    
    # Find the collection
    collection_id = find_pace_collection(session, collection_key=collection_key)
    
    # Search for granules
    if collection_id:
        search_results = search_granules(
            time_range,
            lon_range,
            lat_range,
            collection_id,
        )
        granules = search_results.get("feed", {}).get("entry", [])
    else:
        # Fallback: search by dataset name
        print("Searching by dataset name...")
        all_granules = []
        short_names = PACE_COLLECTIONS.get(collection_key, {}).get("short_names", [])
        for short_name in short_names:
            params = {
                "short_name": short_name,
                "bounding_box": f"{lon_range[0]},{lat_range[0]},{lon_range[1]},{lat_range[1]}",
                "temporal": f"{time_range[0]}T00:00:00Z,{time_range[1]}T23:59:59Z",
                "page_size": 100
            }
            try:
                headers = {}
                token = _get_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                response = requests.get(CMR_GRANULE_SEARCH_URL, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                results = response.json()
                entries = results.get('feed', {}).get('entry', [])
                if entries:
                    all_granules.extend(entries)
                    print(f"Found {len(entries)} granules for {short_name}")
            except Exception as e:
                print(f"Error searching {short_name}: {e}")
        granules = all_granules
    
    if not granules:
        print("No granules found matching the criteria.")
        return []
    
    # Download files
    downloaded_files: List[str] = []
    output_path = Path(output_dir).resolve()
    
    print(f"\nDownloading {len(granules)} granules to {output_path}...")
    
    for i, granule in enumerate(granules, 1):
        granule_id = granule.get('producer_granule_id', '') or granule.get('title', '')
        
        print(f"\n[{i}/{len(granules)}] Processing: {granule_id}")
        
        urls = get_download_urls(granule)
        if not urls:
            print(f"  No download URLs found")
            continue
        
        # For L2, we don't filter by "0.1deg" or "4km". We just take the .nc file.
        # Usually there is one main data file.
        target_url = None
        for url in urls:
            if url.endswith('.nc'):
                target_url = url
                break
        
        if not target_url and urls:
            target_url = urls[0]
            
        if not target_url:
            print("  No suitable download URL found.")
            continue
            
        filename = os.path.basename(target_url.split('?')[0])
        file_path = output_path / filename
        
        if download_file(session, target_url, str(file_path)):
            downloaded_files.append(str(file_path))
            
            if extract_chla:
                extract_chlor_a(file_path, output_path, keep_original=True)
        else:
            print(f"  Skipping because download failed for {filename}")
        
        time.sleep(0.5)
    
    print(f"\nDownload complete! Saved {len(downloaded_files)} files.")
    return downloaded_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PACE OCI L2 BGC granules.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML (uses experiment.bbox, experiment.time_range, io.input_dir)",
    )
    parser.add_argument(
        "--lon-range",
        metavar=("LON_MIN", "LON_MAX"),
        type=float,
        nargs=2,
        default=None,
        help="Override longitude range",
    )
    parser.add_argument(
        "--lat-range",
        metavar=("LAT_MIN", "LAT_MAX"),
        type=float,
        nargs=2,
        default=None,
        help="Override latitude range",
    )
    parser.add_argument(
        "--time-range",
        metavar=("START", "END"),
        type=str,
        nargs=2,
        default=None,
        help="Override time range (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (defaults to <io.input_dir>/pace_bgc when using --config)",
    )
    parser.add_argument(
        "--collection-key",
        type=str,
        default=DEFAULT_COLLECTION_KEY,
        help=f"Collection key (one of: {', '.join(PACE_COLLECTIONS)})",
    )
    parser.add_argument(
        "--extract-chla",
        action="store_true",
        help="Extract and save chlor_a subset per file",
    )
    args = parser.parse_args()

    if not _get_token():
        raise RuntimeError(
            "Missing Earthdata token. Set EARTHDATA_TOKEN (preferred) or EARTHACCESS_TOKEN.\n"
            "Example:\n"
            "  export EARTHDATA_TOKEN='...'\n"
        )

    lon_range: Tuple[float, float] = (-121.0, -119.0)
    lat_range: Tuple[float, float] = (33.0, 35.0)
    time_range: Tuple[str, str] = ("2024-09-06", "2024-09-26")
    output_dir = PROJECT_ROOT / "experiments" / "seabass_validation" / "inputs" / "pace_bgc"

    if args.config:
        config_file = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config)
        cfg = load_config_from_file(config_file)
        west, south, east, north = get_bbox(cfg)
        cfg_time_range = get_time_range(cfg)
        if cfg_time_range is None:
            raise ValueError(
                "Config is missing 'experiment.time_range'. This is required for downloads."
            )
        lon_range = (float(west), float(east))
        lat_range = (float(south), float(north))
        time_range = (str(cfg_time_range[0]), str(cfg_time_range[1]))
        input_dir = get_input_dir(cfg, PROJECT_ROOT, config_dir=config_file.parent)
        output_dir = input_dir / "pace_bgc"

    if args.lon_range is not None:
        lon_range = (float(args.lon_range[0]), float(args.lon_range[1]))
    if args.lat_range is not None:
        lat_range = (float(args.lat_range[0]), float(args.lat_range[1]))
    if args.time_range is not None:
        time_range = (args.time_range[0], args.time_range[1])
    if args.output_dir is not None:
        output_dir = args.output_dir

    print("=" * 60)
    print("PACE OCI Level 2 BGC Data Downloader")
    print("=" * 60)
    print(f"Lon range: {lon_range}")
    print(f"Lat range: {lat_range}")
    print(f"Time range: {time_range}")
    print(f"Output dir: {output_dir.resolve()}")

    downloaded_files = download_pace_bgc(
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        output_dir=str(output_dir),
        extract_chla=args.extract_chla,
        collection_key=args.collection_key,
    )

    if downloaded_files:
        print(f"\nSuccessfully downloaded {len(downloaded_files)} files.")
