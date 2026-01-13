"""Run SDP model prediction on PACE data.

This script is part of the original "mapped Rrs" workflow (PACE OCI L3m-style
gridded inputs under `io.input_dir/{rrs,sss,sst}/`).

It is not compatible with PACE OCI L2 swath products (e.g., `L2.OC_AOP`). For
L2 in-situ validation matchups, use `scripts/data/download_pace_rrs.py`.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
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

if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = project_root / "config" / "initial_test.yaml"
    config = load_config_from_file(config_path)
    
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
    input_dir = get_input_dir(config, project_root)
    
    # Collect file paths
    rrs_paths = sorted([str(p) for p in (input_dir / "rrs").glob("*.nc")])
    sss_paths = sorted([str(p) for p in (input_dir / "sss").glob("*.nc")])
    sst_paths = sorted([str(p) for p in (input_dir / "sst").glob("*.nc")])

    if any(".L2." in Path(p).name or ".L2_" in Path(p).name for p in rrs_paths):
        raise ValueError(
            "Detected what look like PACE L2 swath files in your Rrs inputs. "
            "`scripts/analysis/run_prediction.py` expects mapped/gridded Rrs (L3m-style) "
            f"files under {input_dir / 'rrs'}.\n\n"
            "If you're trying to do in-situ validation against L2 OC_AOP, run:\n"
            "  python scripts/data/download_pace_rrs.py --config <your_config.yaml>"
        )
    
    print(f"Found {len(rrs_paths)} RRS, {len(sss_paths)} SSS, {len(sst_paths)} SST files")
    print(f"Bbox: {bbox}")
    print(f"Pigments: {'all' if pigments is None else ', '.join(pigments)}")
    
    # Run model
    result = sdp_from_pace(rrs_paths, sss_paths, sst_paths, bbox, pigments)
    
    # Save results
    output_dir = get_output_dir(config, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    result.to_netcdf(output_file)
    print(f"Results saved to {output_file}")
