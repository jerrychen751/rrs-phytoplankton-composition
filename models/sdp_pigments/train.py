#!/usr/bin/env python3
"""
Train the SDP pigments model and write coefficient CSVs.

This is the canonical entrypoint for training in this repo. It assumes the
default training dataset and reference coefficient tables live under:

  models/sdp_pigments/training/

and writes trained coefficient ensembles under:

  models/sdp_pigments/coefficients/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sdp_pigments.core.physics import get_rrs_residuals
from models.sdp_pigments.core.training import train_model


DEFAULT_TRAINING_DATA = Path("models/sdp_pigments/training/Kramer-etal_2021.csv")

# Column order matters: the training implementation assumes this HPLC variable ordering.
HPLC_VARIABLES: Sequence[str] = (
    "Tchla",
    "Tchlb",
    "Tchlc",
    "ABcaro",
    "ButFuco",
    "HexFuco",
    "Allo",
    "Diadino",
    "Diato",
    "Fuco",
    "Perid",
    "Zea",
    "MVchla",
    "DVchla",
    "Chllide",
    "MVchlb",
    "DVchlb",
    "Chlc12",
    "Chlc3",
    "Lut",
    "Neo",
    "Viola",
    "Phytin",
    "Phide",
    "Pras",
)


def _expand_path(path: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _resolve_from_project_root(path: Path) -> Path:
    expanded = _expand_path(path)
    return expanded if expanded.is_absolute() else (PROJECT_ROOT / expanded)


def _expected_rrs_columns(wavelengths: Iterable[int]) -> list[str]:
    return [f"Rrs{wl}" for wl in wavelengths]


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=0)
    if suffix == ".tab":
        from models.sdp_pigments.training.tools.convert_pangaea_tab_to_csv import load_pangaea_data

        csv_path = path.with_suffix(".csv")
        print(f"Converting {path.name} -> {csv_path.name}")
        return load_pangaea_data(str(path), str(csv_path))

    raise ValueError(
        f"Training data must be a .csv or .tab file; got {path.name}.\n"
        f"Expected default: {DEFAULT_TRAINING_DATA}"
    )


def run(training_data: Path) -> None:
    training_data = _resolve_from_project_root(training_data)
    df = _load_training_data(training_data)

    print("=" * 72)
    print("SDP Pigments Trainer")
    print("=" * 72)
    print(f"Training data: {training_data}")
    print(f"Rows: {len(df)}")

    _require_columns(df, ["Sal", "Temp"], context="Training dataset")

    wavelengths = np.arange(400, 701)
    rrs_cols = _expected_rrs_columns(wavelengths)
    _require_columns(df, rrs_cols, context="Training dataset (Rrs columns)")
    _require_columns(df, list(HPLC_VARIABLES), context="Training dataset (HPLC columns)")

    sal = pd.to_numeric(df["Sal"], errors="coerce").to_numpy(dtype=float)
    temp = pd.to_numeric(df["Temp"], errors="coerce").to_numpy(dtype=float)

    rrs = df.loc[:, rrs_cols].copy()
    rrs.columns = [int(col.replace("Rrs", "")) for col in rrs.columns]

    # Below-surface and above-surface residuals.
    _, RrsD = get_rrs_residuals(rrs, temp, sal, wavelengths)

    hplc = df.loc[:, list(HPLC_VARIABLES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    print("Training model (writes coefficients to models/sdp_pigments/coefficients/)...")
    train_model(RrsD, hplc)
    print("Done.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train SDP pigments model coefficients.")
    parser.add_argument(
        "--training-data",
        type=Path,
        default=DEFAULT_TRAINING_DATA,
        help=f"Path to training dataset (.csv or .tab). Default: {DEFAULT_TRAINING_DATA}",
    )
    args = parser.parse_args(argv)
    run(args.training_data)


if __name__ == "__main__":
    main()
