"""Main entry point for SDP predictions.

This script operates on *mapped* (gridded) Rrs inputs (e.g., PACE OCI L3m files)
under the configured `io.input_dir/{rrs,sss,sst}/` folders.

It is not compatible with PACE OCI *L2* swath products (e.g., `L2.OC_AOP`), where
Rrs lives under netCDF groups and is not on a lat/lon grid. For L2 validation
matchups, use `scripts/data/download_pace_rrs.py` to build a per-station matchup
dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.pace import sdp_from_pace
from utils.config_loader import (
    load_config_from_file,
    get_experiment_name,
    get_experiment_description,
    get_bbox,
    get_pigments,
    get_input_dir,
    get_output_dir,
    get_output_filename,
)


def list_nc_files(directory: Path) -> list[str]:
    """Collect NetCDF files, supporting both .nc and .nc4 extensions."""
    files = set()
    for pattern in ("*.nc", "*.nc4"):
        for path in directory.glob(pattern):
            files.add(path.resolve())
    return sorted(str(path) for path in files)

if __name__ == "__main__":
    # Prompt for config file path
    config_input = input("Enter path to config file (relative to project root or absolute): ").strip()
    if not config_input:
        raise ValueError("Config file path cannot be empty")
    
    config_path = Path(config_input)
    config_file = config_path if config_path.is_absolute() else (project_root / config_path)
    config_file = config_file.resolve()
    config = load_config_from_file(config_file)
    
    # Extract experiment name from config
    experiment = get_experiment_name(config)
    bbox = get_bbox(config)
    pigments = get_pigments(config)
    output_filename = get_output_filename(config)
    
    print(f"Experiment: {experiment}")
    description = get_experiment_description(config)
    if description:
        print(f"Description: {description}")
    
    # Data directory paths
    input_dir = get_input_dir(config, project_root, config_dir=config_file.parent)
    
    # Collect file paths
    rrs_paths = list_nc_files(input_dir / "rrs")
    sss_paths = list_nc_files(input_dir / "sss")
    sst_paths = list_nc_files(input_dir / "sst")

    if any(".L2." in Path(p).name or ".L2_" in Path(p).name for p in rrs_paths):
        raise ValueError(
            "Detected what look like PACE L2 swath files in your Rrs inputs. "
            "`main.py` expects mapped/gridded Rrs (L3m-style) files under "
            f"{input_dir / 'rrs'}.\n\n"
            "If you're trying to do in-situ validation against L2 OC_AOP, run:\n"
            "  python scripts/data/download_pace_rrs.py --config <your_config.yaml>"
        )
    
    print(f"Found {len(rrs_paths)} RRS, {len(sss_paths)} SSS, {len(sst_paths)} SST files")
    print(f"Bbox: {bbox}")
    print(f"Pigments: {'all' if pigments is None else ', '.join(pigments)}")
    
    # Run model
    result = sdp_from_pace(rrs_paths, sss_paths, sst_paths, bbox, pigments)
    
    # Save results
    output_dir = get_output_dir(config, project_root, config_dir=config_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    result.to_netcdf(output_file)
    print(f"Results saved to {output_file}")
