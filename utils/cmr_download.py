"""NASA Earthdata CMR search and authenticated download utilities.

Provides:
  - search_cmr()          — find PACE L2 AOP granules near a lat/lon on a date
  - get_bearer_token()    — obtain a URS OAuth Bearer token from ~/.netrc
  - download_granule()    — streaming download with Bearer auth
  - download_granule_fsync() — same, but with os.fsync + atomic rename for
                               Lustre-safe writes (avoids HDF5 checksum corruption)

These were originally private helpers in validate_kramer_sdp.py and are now
shared by both that script and download_l2_rrs.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

# ── CMR endpoints ────────────────────────────────────────────────────────

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
L2_SHORT_NAMES = ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"]


# ── CMR search ───────────────────────────────────────────────────────────

def search_cmr(
    lat: float,
    lon: float,
    date: "datetime.date",
    *,
    padding_deg: float = 1.0,
    short_names: list[str] | None = None,
    page_size: int = 10,
) -> list[dict]:
    """
    Search CMR for PACE L2 AOP granules covering a point on a given date.

    Args:
        lat: Latitude of the target point (degrees N).
        lon: Longitude of the target point (degrees E).
        date: The observation date to search.
        padding_deg: Bounding-box padding around the point (degrees).
            A 1-degree pad ensures we capture any swath whose footprint
            overlaps the point, even if the granule center is slightly off.
        short_names: CMR collection short names to search.  Defaults to
            both science-quality and near-real-time PACE L2 AOP products.
        page_size: Max results per short_name query.

    Returns:
        List of dicts with 'title' and 'url' keys. The URL points to the
        direct-download .nc file on OB.DAAC or the cloud archive.
    """
    if short_names is None:
        short_names = list(L2_SHORT_NAMES)

    bbox = f"{lon - padding_deg},{lat - padding_deg},{lon + padding_deg},{lat + padding_deg}"
    start = f"{date.isoformat()}T00:00:00Z"
    end = f"{date.isoformat()}T23:59:59Z"

    results: list[dict] = []
    for short_name in short_names:
        params = {
            "short_name": short_name,
            "bounding_box": bbox,
            "temporal": f"{start},{end}",
            "page_size": page_size,
            "sort_key": "-start_date",
        }
        try:
            resp = requests.get(CMR_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for entry in data.get("feed", {}).get("entry", []):
                download_url = None
                for link in entry.get("links", []):
                    href = link.get("href", "")
                    if (href.endswith(".nc")
                            and ("oceandata" in href or "obdaac" in href)
                            and "opendap" not in href.lower()):
                        download_url = href
                        break

                if download_url:
                    results.append({
                        "title": entry.get("title", ""),
                        "url": download_url,
                    })
        except Exception as e:
            print(f"      CMR search error ({short_name}): {e}")

    return results


def search_cmr_bbox(
    bbox: tuple[float, float, float, float],
    date: "datetime.date",
    *,
    short_names: list[str] | None = None,
    page_size: int = 200,
) -> list[dict]:
    """
    Search CMR for PACE L2 AOP granules overlapping a bounding box on a date.

    This is a broader variant of search_cmr() that uses a pre-computed bbox
    rather than a single point with padding.  Useful for downloading all
    granules that cover a geographic region on a given day.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max).
        date: The observation date to search.
        short_names: CMR collection short names.
        page_size: Max results per short_name query.

    Returns:
        List of dicts with 'title' and 'url' keys, deduplicated by URL.
    """
    if short_names is None:
        short_names = list(L2_SHORT_NAMES)

    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    start = f"{date.isoformat()}T00:00:00Z"
    end = f"{date.isoformat()}T23:59:59Z"

    seen_urls: set[str] = set()
    results: list[dict] = []

    for short_name in short_names:
        params = {
            "short_name": short_name,
            "bounding_box": bbox_str,
            "temporal": f"{start},{end}",
            "page_size": page_size,
            "sort_key": "-start_date",
        }
        try:
            resp = requests.get(CMR_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for entry in data.get("feed", {}).get("entry", []):
                download_url = None
                for link in entry.get("links", []):
                    href = link.get("href", "")
                    if (href.endswith(".nc")
                            and ("oceandata" in href or "obdaac" in href)
                            and "opendap" not in href.lower()):
                        download_url = href
                        break

                if download_url and download_url not in seen_urls:
                    seen_urls.add(download_url)
                    results.append({
                        "title": entry.get("title", ""),
                        "url": download_url,
                    })
        except Exception as e:
            print(f"      CMR search error ({short_name}): {e}")

    return results


# ── Authentication ────────────────────────────────────────────────────────

def get_bearer_token() -> str:
    """
    Obtain a URS Bearer token using ~/.netrc credentials.

    The NASA cloud archive (TEA) requires OAuth-based authentication.
    Basic auth via .netrc only works for the traditional OB.DAAC and
    OPeNDAP endpoints.  For the cloud archive, we generate a Bearer
    token from the URS API, which is then sent in the Authorization
    header on every download request.

    Raises:
        RuntimeError: If no .netrc entry exists for urs.earthdata.nasa.gov.
    """
    import base64
    import netrc as _netrc

    nrc = _netrc.netrc()
    auth = nrc.authenticators("urs.earthdata.nasa.gov")
    if not auth:
        raise RuntimeError(
            "No .netrc entry for urs.earthdata.nasa.gov. "
            "See: https://wiki.earthdata.nasa.gov/display/EL/"
            "How+To+Access+Data+With+cURL+And+Wget"
        )
    user, _, password = auth
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}"}

    # Check for existing token first
    resp = requests.get(
        "https://urs.earthdata.nasa.gov/api/users/tokens",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    tokens = resp.json()
    if tokens:
        return tokens[0]["access_token"]

    # Create a new token if none exist
    resp = requests.post(
        "https://urs.earthdata.nasa.gov/api/users/token",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Download ──────────────────────────────────────────────────────────────

def download_granule(
    url: str,
    dest: Path,
    session: requests.Session,
) -> bool:
    """
    Download a single granule with Bearer token auth (streaming).

    Args:
        url: Direct download URL for the .nc file.
        dest: Local path to write the file.
        session: requests.Session with Authorization header already set.

    Returns:
        True on success, False on failure (dest is cleaned up on error).
    """
    try:
        resp = session.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def download_granule_fsync(
    url: str,
    dest: Path,
    session: requests.Session,
    *,
    max_retries: int = 3,
    retry_backoff_s: float = 10.0,
) -> bool:
    """
    Download a granule with os.fsync + atomic rename for Lustre safety.

    On Lustre, H5Fclose() flushes to the kernel page cache but Lustre
    doesn't immediately commit to OSTs.  If another process reads the file
    before the write-behind cache is flushed, HDF5 metadata checksums can
    mismatch.  This function writes to a .tmp file, fsyncs it, then
    atomically renames — guaranteeing readers never see a partial or
    unflushed file.

    Retries with exponential backoff on failure (server timeouts, rate
    limiting). Waits retry_backoff_s * 3^attempt seconds between retries
    (default: 10s, 30s, 90s).

    Args:
        url: Direct download URL for the .nc file.
        dest: Final local path. A sibling .tmp.nc file is used during download.
        session: requests.Session with Authorization header already set.
        max_retries: Number of retry attempts after the first failure.
        retry_backoff_s: Base delay (seconds) for exponential backoff.

    Returns:
        True on success, False after all retries exhausted.
    """
    import time as _time

    tmp = dest.with_suffix(".tmp.nc")

    for attempt in range(1 + max_retries):
        try:
            resp = session.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)

            # Force dirty pages to Lustre OSTs before the rename
            fd = os.open(str(tmp), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

            tmp.rename(dest)  # Atomic: readers never see a partial file
            return True
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt < max_retries:
                wait = retry_backoff_s * (3 ** attempt)
                print(f"retry {attempt + 1}/{max_retries} in {wait:.0f}s ({e})")
                _time.sleep(wait)

    dest.unlink(missing_ok=True)
    return False
