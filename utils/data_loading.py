"""Convert Pangaea .tab files to CSV format."""

import pandas as pd
import numpy as np
import os

def load_pangaea_data(input_file: str, output_file: str | None = None) -> pd.DataFrame:
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_row: int | None = None
    for i, line in enumerate(lines):
        if line.startswith('Campaign\t') or line.startswith('Campaign '): # around line 360
            header_row = i
            break
    
    if header_row is None:
        raise ValueError("Could not find header row")
    
    print(f"Found header at line {header_row + 1}")
    
    # Read the data starting from the header row
    # Skip the metadata lines before the header
    df = pd.read_csv(input_file, sep='\t', skiprows=header_row, encoding='utf-8', # tab-delimited
                     low_memory=False)
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    column_mapping: dict[str, str] = {}
    
    for col in df.columns:
        if col.startswith('Rrs_'):
            wavelength = col.split('_')[1].split()[0]
            column_mapping[col] = f'Rrs{wavelength}'
        elif col == 'Sal (PSU, CTD)':
            column_mapping[col] = 'Sal'
        elif col == 'Temp [°C] (CTD)':
            column_mapping[col] = 'Temp'
        elif 'TChl a' in col:
            column_mapping[col] = 'Tchla'
        elif 'Chl b + DV Chl b' in col:
            column_mapping[col] = 'Tchlb'
        elif 'Chl c1+c2+c3' in col:
            column_mapping[col] = 'Tchlc'
        elif 'a-Car + b-Car' in col:
            column_mapping[col] = 'ABcaro'
        elif 'But-fuco' in col:
            column_mapping[col] = 'ButFuco'
        elif 'Hex-fuco' in col:
            column_mapping[col] = 'HexFuco'
        elif col.startswith('Allo [µg/l]'):
            column_mapping[col] = 'Allo'
        elif col.startswith('Diadino [µg/l]'):
            column_mapping[col] = 'Diadino'
        elif col.startswith('Diato [µg/l]'):
            column_mapping[col] = 'Diato'
        elif col.startswith('Fuco [µg/l]'):
            column_mapping[col] = 'Fuco'
        elif col.startswith('Perid [µg/l]'):
            column_mapping[col] = 'Perid'
        elif col.startswith('Zea [µg/l]'):
            column_mapping[col] = 'Zea'
        elif 'MV chl a' in col:
            column_mapping[col] = 'MVchla'
        elif 'DV chl a' in col:
            column_mapping[col] = 'DVchla'
        elif 'Chlide a' in col:
            column_mapping[col] = 'Chllide'
        elif 'MV chl b' in col:
            column_mapping[col] = 'MVchlb'
        elif 'DV chl b' in col:
            column_mapping[col] = 'DVchlb'
        elif 'Chl c1+c2' in col and 'c3' not in col:
            column_mapping[col] = 'Chlc12'
        elif col.startswith('Chl c3 [µg/l]'):
            column_mapping[col] = 'Chlc3'
        elif col.startswith('Lut [µg/l]'):
            column_mapping[col] = 'Lut'
        elif col.startswith('Neo [µg/l]'):
            column_mapping[col] = 'Neo'
        elif col.startswith('Viola [µg/l]'):
            column_mapping[col] = 'Viola'
        elif col.startswith('Phaeophytin [µg/l]'):
            column_mapping[col] = 'Phytin'
        elif col.startswith('Phaeopho a [µg/l]'):
            column_mapping[col] = 'Phide'
        elif col.startswith('Pras [µg/l]'):
            column_mapping[col] = 'Pras'
    
    df_renamed = df.rename(columns=column_mapping)
    
    all_cols = df_renamed.columns.tolist()
    
    metadata_cols = [col for col in all_cols if col not in ['Sal', 'Temp'] 
                     and not col.startswith('Rrs') 
                     and col not in ['Tchla', 'Tchlb', 'Tchlc', 'ABcaro', 'ButFuco', 'HexFuco', 
                                     'Allo', 'Diadino', 'Diato', 'Fuco', 'Perid', 'Zea', 
                                     'MVchla', 'DVchla', 'Chllide', 'MVchlb', 'DVchlb', 
                                     'Chlc12', 'Chlc3', 'Lut', 'Neo', 'Viola', 'Phytin', 'Phide', 'Pras']]
    
    rrs_cols = [col for col in all_cols if col.startswith('Rrs')]
    rrs_cols_sorted = sorted(rrs_cols, key=lambda x: int(x.replace('Rrs', '')))
    
    pigment_cols = ['Tchla', 'Tchlb', 'Tchlc', 'ABcaro', 'ButFuco', 'HexFuco', 'Allo', 
                    'Diadino', 'Diato', 'Fuco', 'Perid', 'Zea', 'MVchla', 'DVchla', 
                    'Chllide', 'MVchlb', 'DVchlb', 'Chlc12', 'Chlc3', 'Lut', 'Neo', 
                    'Viola', 'Phytin', 'Phide', 'Pras']
    pigment_cols = [col for col in pigment_cols if col in df_renamed.columns]
    
    new_column_order = metadata_cols + ['Sal', 'Temp'] + rrs_cols_sorted + pigment_cols
    new_column_order = [col for col in new_column_order if col in df_renamed.columns]
    
    df_final = df_renamed[new_column_order]
    
    for col in rrs_cols_sorted:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    if 'Sal' in df_final.columns:
        df_final['Sal'] = pd.to_numeric(df_final['Sal'], errors='coerce')
    if 'Temp' in df_final.columns:
        df_final['Temp'] = pd.to_numeric(df_final['Temp'], errors='coerce')
    
    for col in pigment_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = base_name + '.csv'
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved {len(df_final)} rows and {len(df_final.columns)} columns to {output_file}")
    
    return df_final # type: ignore


if __name__ == "__main__":
    from pathlib import Path
    
    # Get project root (go up from src/rrs_sdp/utils/ to project root)
    project_root = Path(__file__).resolve().parent.parent.parent
    input_file = project_root / 'data' / 'model_training' / 'Kramer-etal_2021.tab'
    output_file = project_root / 'data' / 'model_training' / 'Kramer-etal_2021.csv'
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please update the input_file path in the script.")
    else:
        df = load_pangaea_data(str(input_file), str(output_file))

