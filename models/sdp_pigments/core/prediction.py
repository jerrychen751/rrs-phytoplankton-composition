"""SDP model prediction functions."""

from models.sdp_pigments.core.physics import get_rrs_residuals
from utils.config_loader import get_project_root
import pandas as pd
import numpy as np
from typing import Optional, List
import sys

# Import training function (need to add project root to path first)
_project_root = get_project_root()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from models.sdp_pigments.train import main as train_model_from_data

def run_sdp(
    rrs: pd.DataFrame, 
    wl: np.ndarray, 
    sst: np.ndarray, 
    sss: np.ndarray,
    pigments: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Predict pigment concentrations from Rrs spectra using trained SDP model.
    
    Computes Rrs residuals, takes 2nd derivative, then applies ensemble of 100 model
    permutations. Takes median over permutations. Negative predictions clipped to zero.
    
    Available pigments: Tchla, Zea, DVchla, ButFuco, HexFuco, Allo, MVchlb, Neo,
    Viola, Fuco, Chlc12, Chlc3, Perid.
    
    Returns DataFrame with predicted concentrations (µg/l).
    """
 
    # All 13 pigments that were trained (from Kramer_Rrs_pigments.py)
    all_available_pigments = ['Tchla','Zea','DVchla','ButFuco','HexFuco','Allo','MVchlb',
                              'Neo','Viola','Fuco','Chlc12','Chlc3','Perid']
    
    # Use all pigments if none specified, otherwise use the requested ones
    if pigments is None:
        sdp_names = all_available_pigments.copy()
    else:
        # Validate requested pigments
        invalid_pigments = [p for p in pigments if p not in all_available_pigments]
        if invalid_pigments:
            raise ValueError(
                f"Invalid pigment names: {invalid_pigments}. "
                f"Available pigments: {all_available_pigments}"
            )
        sdp_names = pigments.copy()
    
    # Display names for output DataFrame columns
    display_names = {
        'Tchla': 'T chla',
        'Zea': 'Zea',
        'DVchla': 'DV chla',
        'ButFuco': 'ButFuco',
        'HexFuco': 'HexFuco',
        'Allo': 'Allo',
        'MVchlb': 'MV chlb',
        'Neo': 'Neo',
        'Viola': 'Viola',
        'Fuco': 'Fuco',
        'Chlc12': 'chl c1+c2',
        'Chlc3': 'chl c3',
        'Perid': 'Perid'
    }

    # Check if model coefficients exist before processing
    project_root = get_project_root()
    coeff_dir = project_root / "models" / "sdp_pigments" / "coefficients"
    
    missing_coeffs = []
    for name in sdp_names:
        a_file = coeff_dir / f'a_coefs_{name}.csv'
        c_file = coeff_dir / f'c_coefs_{name}.csv'
        if not a_file.exists() or not c_file.exists():
            missing_coeffs.append(name)
    
    if missing_coeffs:
        print(f"Model coefficients not found for pigments: {missing_coeffs}")
        print(f"Expected location: {coeff_dir}")
        print("Training model automatically...")
        # Avoid leaking the caller's CLI args into the trainer's argparse.
        train_model_from_data([])
        print("Model training complete. Verifying coefficients...")
        
        # Verify coefficients were created
        still_missing = []
        for name in missing_coeffs:
            a_file = coeff_dir / f'a_coefs_{name}.csv'
            c_file = coeff_dir / f'c_coefs_{name}.csv'
            if not a_file.exists() or not c_file.exists():
                still_missing.append(name)
        
        if still_missing:
            raise FileNotFoundError(
                f"Model training completed but coefficients still missing for: {still_missing}\n"
                f"Expected location: {coeff_dir}"
            )

    rrs_residuals = get_rrs_residuals(rrs, sst, sss, wl)[1]
    rrs_residuals_d2 = np.diff(rrs_residuals, 2, axis=0).T

    print(rrs_residuals_d2)
    print(rrs_residuals_d2.shape)

    sdp = np.zeros((rrs_residuals_d2.shape[0],len(sdp_names)))

    for p, name in enumerate(sdp_names):
        print(f"Predicting {name} ({p+1}/{len(sdp_names)})...")

        # Load coefficient files (already verified to exist above)
        a_file = coeff_dir / f'a_coefs_{name}.csv'
        c_file = coeff_dir / f'c_coefs_{name}.csv'
        
        a_coefs = pd.read_csv(a_file).values  # shape: (n_wl, 100)
        c_coefs = pd.read_csv(c_file).values.flatten()  # shape: (100,)

        # Matrix multiplication to compute all runs at once for all samples
        # Result: run_vals_all shape (n_samples, 100)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        # Take median over runs axis (axis=1)
        median_run = np.median(run_vals_all, axis=1)

        # Enforce non-negative
        median_run[median_run < 0] = 0

        sdp[:, p] = median_run

    # Create column names from display_names mapping
    output_columns = [display_names[name] for name in sdp_names]
    return pd.DataFrame(sdp, columns=output_columns) # type: ignore
