"""Load experiment configuration from YAML or Python config files."""

import os
import importlib.util
import types
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Assumes this file is in utils/ directory, so goes up one level.
    """
    return Path(__file__).resolve().parent.parent


def _expand_path(path: Path) -> Path:
    """Expand ~ and environment variables in a filesystem path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _resolve_path(path_str: str, project_root: Path, config_dir: Optional[Path]) -> Path:
    """
    Resolve a path from config.

    - Absolute paths are returned as-is (after ~ and env var expansion)
    - Relative paths resolve against config_dir when provided (experiment-centric configs)
    - Otherwise, relative paths resolve against project_root (legacy behavior)
    """
    path = _expand_path(Path(path_str))
    if path.is_absolute():
        return path
    if config_dir is not None:
        return config_dir / path
    return project_root / path


def resolve_path(path_str: str, project_root: Path, config_dir: Optional[Path] = None) -> Path:
    """Public wrapper for config path resolution rules."""
    return _resolve_path(path_str, project_root, config_dir)


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


def get_input_dir(config: Dict[str, Any], project_root: Path, config_dir: Optional[Path] = None) -> Path:
    """
    Get input directory path from config.
    
    Requires 'io' section with 'input_dir' specified in config.
    Path can be relative to config_dir (preferred) or project_root.
    """
    input_config = config.get("io", {})
    if "input_dir" not in input_config:
        raise ValueError(
            "Input directory must be specified in config file under 'io.input_dir'.\n"
            "Example:\n"
            "io:\n"
            "  input_dir: 'inputs'"
        )
    
    return _resolve_path(str(input_config["input_dir"]), project_root, config_dir)


def get_output_dir(config: Dict[str, Any], project_root: Path, config_dir: Optional[Path] = None) -> Path:
    """
    Get output directory for this experiment.

    Required (experiment-centric) config style:
      io:
        output:
          dir: "outputs"
    """
    output_cfg = config.get("io", {}).get("output", {})
    output_dir = output_cfg.get("dir")
    if output_dir is not None:
        return _resolve_path(str(output_dir), project_root, config_dir)

    raise ValueError(
        "Output directory must be specified in config under 'io.output.dir'.\n"
        "Example:\n"
        "io:\n"
        "  output:\n"
        "    dir: 'outputs'\n"
        "    filename: 'sdp_results.nc'\n"
    )


def get_output_filename(config: Dict[str, Any]) -> str:
    """Get output filename (defaults to sdp_results.nc)."""
    return config.get("io", {}).get("output", {}).get("filename", "sdp_results.nc")


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from a config file path.
    
    Supported formats:
    - YAML (`.yaml` / `.yml`)
    - Python (`.py`) containing either a `CONFIG` dict or a `get_config()` function.

    `config_path` can be relative or absolute.
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
    
    suffix = config_file.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Expected YAML config to be a mapping/dict: {config_file}")
        return config

    if suffix == ".py":
        module_name = f"_rrs_sdp_experiment_config_{abs(hash(str(config_file)))}"
        spec = importlib.util.spec_from_file_location(module_name, config_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import config module from {config_file}")
        module = importlib.util.module_from_spec(spec)
        assert isinstance(module, types.ModuleType)
        spec.loader.exec_module(module)

        if hasattr(module, "get_config"):
            config = module.get_config()
        elif hasattr(module, "CONFIG"):
            config = getattr(module, "CONFIG")
        else:
            raise AttributeError(
                f"Python config file must define CONFIG (dict) or get_config(): {config_file}"
            )

        if not isinstance(config, dict):
            raise TypeError(
                f"Python config must evaluate to a dict; got {type(config).__name__}: {config_file}"
            )
        return config

    raise ValueError(
        f"Unsupported config format '{suffix}'. Expected .yaml/.yml or .py: {config_file}"
    )
