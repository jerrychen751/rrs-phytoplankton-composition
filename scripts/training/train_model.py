"""Train SDP model and save coefficients to CSV files."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from core.physics import get_rrs_residuals
from core.training import train_model
from utils.config_loader import get_project_root
import pandas as pd
import numpy as np

def main() -> None:
    """Load training data, compute Rrs residuals, and train model."""
    project_root = get_project_root()
    
    # Load training data from CSV file
    data_file = project_root / 'data' / 'model_training' / 'Kramer-etal_2021.csv'
    data = pd.read_csv(data_file, header=0)
    print(f"Loaded {len(data)} samples from CSV file")
    print(data.head())

    sal = np.array(data.loc[:,'Sal'].values)
    temp = np.array(data.loc[:,'Temp'].values)
    Rrs = data.loc[:,'Rrs400':'Rrs700']
    # Rename columns from 'Rrs400', 'Rrs401', ... to 400, 401, ... for compatibility with Kramer_hyperRrs
    Rrs.columns = [int(col.replace('Rrs', '')) for col in Rrs.columns]
    wavelegnths = np.arange(400,701)

    rrsD, RrsD = get_rrs_residuals(Rrs, temp, sal, wavelegnths) # (just-below surface remote sensing residual, above surface remote-sensing reflectance residual)

    hplc = data.loc[:,'Tchla':'Pras'].values # inclusive

    # train model
    train_model(RrsD, hplc) # runs model and saves coefficients to CSV files in model_coefficients/ folder

if __name__ == "__main__":
    main()

