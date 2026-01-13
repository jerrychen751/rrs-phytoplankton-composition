"""Load experiment configuration from YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Assumes this file is in utils/ directory, so goes up one level.
    """
    return Path(__file__).resolve().parent.parent


def load_config(experiment: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.
    
    Looks for config file: config/<experiment>.yaml
    Returns dict with experiment parameters.
    """
    if config_dir is None:
        # Default to config/ directory in project root
        project_root = get_project_root()
        config_dir = project_root / "config"
    
    config_file = config_dir / f"{experiment}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Create a config file for experiment '{experiment}' in {config_dir}/"
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_bbox(config: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Extract bbox from config as tuple (west, south, east, north)."""
    bbox_dict = config.get("experiment", {}).get("bbox", {})
    west = bbox_dict.get("west")
    south = bbox_dict.get("south")
    east = bbox_dict.get("east")
    north = bbox_dict.get("north")

    if any(v is None for v in (west, south, east, north)):
        raise ValueError(
            "Bounding box must be specified in config under 'experiment.bbox'.\n"
            "Example:\n"
            "experiment:\n"
            "  bbox:\n"
            "    west: -121\n"
            "    south: 33\n"
            "    east: -119\n"
            "    north: 35\n"
        )

    return (west, south, east, north)


def get_pigments(config: Dict[str, Any]) -> Optional[List[str]]:
    """
    Extract pigments list from config. Returns None if all pigments should be predicted.
    
    In YAML, use 'null' (which becomes Python None when loaded).
    """
    pigments = config.get("sdp", {}).get("pigments")
    if pigments is None:
        return None
    return pigments if isinstance(pigments, list) else [pigments]


def get_experiment_name(config: Dict[str, Any]) -> str:
    """
    Extract experiment name from config.
    
    Requires 'experiment' section with 'name' specified in config.

    The experiment name is treated as a stable, filesystem-friendly identifier
    (used for output directory naming).
    """
    experiment_config = config.get("experiment", {})
    if "name" not in experiment_config:
        raise ValueError(
            "Experiment name must be specified in config file under 'experiment.name'.\n"
            "Example:\n"
            "experiment:\n"
            "  name: 'initial_test'"
        )
    return experiment_config["name"]


def get_experiment_description(config: Dict[str, Any]) -> Optional[str]:
    """Extract optional experiment description from config."""
    return config.get("experiment", {}).get("description")


def get_time_range(config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Extract time range from config.
    
    Returns tuple (start_date, end_date) as strings in YYYY-MM-DD format,
    or None if not specified.
    """
    time_range = config.get("experiment", {}).get("time_range")
    if time_range is None:
        return None
    return (time_range.get("start"), time_range.get("end"))


def get_region(config: Dict[str, Any]) -> Optional[str]:
    """Extract optional region name from config."""
    return config.get("experiment", {}).get("region")


def get_input_dir(config: Dict[str, Any], project_root: Path) -> Path:
    """
    Get input directory path from config.
    
    Requires 'io' section with 'input_dir' specified in config.
    Path can be relative to project_root or absolute.
    """
    input_config = config.get("io", {})
    if "input_dir" not in input_config:
        raise ValueError(
            "Input directory must be specified in config file under 'io.input_dir'.\n"
            "Example:\n"
            "io:\n"
            "  input_dir: 'data/input/<experiment_name>'"
        )
    
    input_dir = Path(input_config["input_dir"])
    if input_dir.is_absolute():
        return input_dir
    return project_root / input_dir


def get_output_base_dir(config: Dict[str, Any], project_root: Path) -> Path:
    """
    Get base output directory path from config.

    Uses io.output.base_dir if provided; otherwise defaults to data/outputs.
    Path can be relative to project_root or absolute.
    """
    output_cfg = config.get("io", {}).get("output", {})
    base_dir = output_cfg.get("base_dir", "data/outputs")
    base_path = Path(base_dir)
    if base_path.is_absolute():
        return base_path
    return project_root / base_path


def get_output_dir(config: Dict[str, Any], project_root: Path) -> Path:
    """
    Get output directory for this experiment.

    Output directory is derived as: io.output.base_dir / experiment.name
    """
    experiment_name = get_experiment_name(config)
    return get_output_base_dir(config, project_root) / experiment_name


def get_output_filename(config: Dict[str, Any]) -> str:
    """Get output filename (defaults to sdp_results.nc)."""
    return config.get("io", {}).get("output", {}).get("filename", "sdp_results.nc")


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file path.
    
    config_path can be relative or absolute.
    Returns dict with experiment parameters.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        # If relative, assume it's relative to project root
        project_root = get_project_root()
        config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}"
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
