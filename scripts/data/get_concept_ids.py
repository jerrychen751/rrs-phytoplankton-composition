#!/usr/bin/env python3
"""Interactive helper for looking up NASA Earthdata collection concept IDs.

This is a legacy utility that was useful when experimenting with Harmony OGC API
URLs for mapped (L3-style) products.

For the current PACE OCI L2 validation workflow, prefer
`scripts/data/download_pace_rrs.py`, which searches CMR directly by short name.
"""

import requests
import json

def get_concept_id(short_name, version):
    """
    Get the concept ID for a NASA Earthdata collection using CMR Search API.
    
    Args:
        short_name (str): The collection short name (e.g., "PACE_OCI_L3M_RRS")
        version (str): The collection version (e.g., "3.1")
    
    Returns:
        str: The concept ID if found, None otherwise
    """
    # 1. Define the CMR Search API Endpoint for Collections
    # Documentation: "Find all collections" -> https://cmr.earthdata.nasa.gov/search/collections
    base_url = "https://cmr.earthdata.nasa.gov/search/collections.json"

    # 2. Define the parameters based on "Collection Search By Parameters"
    params = {
        "short_name": short_name,
        "version": version,
        "page_size": 1  # We only need one result
    }

    print(f"Querying CMR for {params['short_name']} v{params['version']}...")

    # 3. Execute the GET request
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        
        # 4. Parse the response based on "Supported Result Formats -> JSON"
        # The doc says the structure is feed -> entry -> id
        entries = data.get('feed', {}).get('entry', [])
        
        if entries:
            concept_id = entries[0]['id']
            print("\n------------------------------------------------")
            print(f"✅ SUCCESS")
            print(f"Collection Concept ID: {concept_id}")
            print("------------------------------------------------")
            return concept_id
        else:
            print("No collection found. Please check the Short Name and Version.")
            return None
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Run the tool
if __name__ == "__main__":
    print("=" * 60)
    print("NASA Earthdata Collection Concept ID Lookup")
    print("=" * 60)
    print()
    
    # Prompt user for collection parameters
    short_name = input("Enter collection short name (e.g., PACE_OCI_L3M_RRS): ").strip()
    if not short_name:
        print("Error: Short name cannot be empty.")
        exit(1)
    
    version = input("Enter collection version (e.g., 3.1): ").strip()
    if not version:
        print("Error: Version cannot be empty.")
        exit(1)
    
    print()
    concept_id = get_concept_id(short_name, version)
    
    if concept_id:
        print("\n" + "=" * 60)
        print("Harmony API URL Template")
        print("=" * 60)
        print(f"\nBase URL:")
        print(f"https://harmony.earthdata.nasa.gov/{concept_id}/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset")
        print(f"\nComplete URL with parameters:")
        print(f"https://harmony.earthdata.nasa.gov/{concept_id}/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset?subset=lat(min_lat:max_lat)&subset=lon(min_lon:max_lon)&subset=time(\"start_time\":\"end_time\")&format=application%2Fx-netcdf4")
        print(f"\nParameter format examples:")
        print(f"  - Latitude: subset=lat(33:35)")
        print(f"  - Longitude: subset=lon(-121:-119)")
        print(f"  - Time: subset=time(\"2024-09-06T00:00:00Z\":\"2024-09-26T23:59:59Z\")")
        print(f"  - Format: format=application%2Fx-netcdf4 (for NetCDF)")
        print("=" * 60)
