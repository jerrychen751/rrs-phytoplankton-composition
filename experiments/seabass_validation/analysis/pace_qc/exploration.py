import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PACE_L2_DIR = PROJECT_ROOT / "experiments" / "seabass_validation" / "inputs" / "pace_bgc"
filepath = PACE_L2_DIR / "PACE_OCI.20240906T202704.L2.OC_BGC.V3_1.nc"

# Essentially 32-bit flag --> storage of 32 independent flags at once; allows for efficient storage of multiple potential flags
excluded_bits = [0, 1, 3, 4, 8, 9, 14, 16, 25, 26]
exclude_mask = 0
for bit in excluded_bits:
    exclude_mask |= (1 << bit)

groups = xr.open_groups(filepath)
pprint(groups)

geophys_ds = groups.get('/geophysical_data')
