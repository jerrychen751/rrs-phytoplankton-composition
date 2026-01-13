import json
import re
import os

def extract_column_headers(input_file: str, output_file: str) -> dict:
    print(f"Reading column headers from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_row = None
    for i, line in enumerate(lines):
        if line.startswith('Campaign\t') or line.startswith('Campaign '):
            header_row = i
            break
    
    if header_row is None:
        raise ValueError("Could not find header row")
    
    print(f"Found header at line {header_row + 1}")
    
    header_line = lines[header_row].strip()
    column_names = header_line.split('\t')
    
    print(f"Found {len(column_names)} columns in header")
    
    param_descriptions = {}
    
    for i, line in enumerate(lines[:header_row]):
        line = line.strip()
        if not line or line.startswith('/*') or line.startswith('*/'):
            continue
        
        if line.startswith('Parameter(s):'):
            continue
        elif '*' in line and ('PI:' in line or 'METHOD/DEVICE:' in line or 'GEOCODE' in line):
            param_line = line.strip()
            if not param_line:
                continue
            
            param_info = {}
            
            if '*' in param_line:
                parts = param_line.split('*')
                main_part = parts[0].strip()
                
                param_info['full_description'] = main_part
                
                for part in parts[1:]:
                    part = part.strip()
                    if part.startswith('PI:'):
                        pi_info = part.replace('PI:', '').strip()
                        if '(' in pi_info and ')' in pi_info:
                            name = pi_info.split('(')[0].strip()
                            orcid_match = re.search(r'https://orcid\.org/(\d{4}-\d{4}-\d{4}-\d{4})', pi_info)
                            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', pi_info)
                            param_info['principal_investigator'] = {
                                'name': name,
                                'orcid': orcid_match.group(1) if orcid_match else None,
                                'email': email_match.group(1) if email_match else None
                            }
                    elif part.startswith('METHOD/DEVICE:'):
                        method = part.replace('METHOD/DEVICE:', '').strip()
                        param_info['method_device'] = method
                    elif part.startswith('COMMENT:'):
                        comment = part.replace('COMMENT:', '').strip()
                        param_info['comment'] = comment
                    elif part.startswith('GEOCODE'):
                        param_info['is_geocode'] = True
                
                if '(' in main_part and ')' in main_part:
                    short_name_match = re.search(r'\(([^)]+)\)', main_part)
                    if short_name_match:
                        param_info['short_name'] = short_name_match.group(1)
                
                if '[' in main_part and ']' in main_part:
                    unit_match = re.search(r'\[([^\]]+)\]', main_part)
                    if unit_match:
                        param_info['unit'] = unit_match.group(1)
                
                param_info['name'] = main_part.split('[')[0].strip()
                if '(' in param_info['name']:
                    param_info['name'] = param_info['name'].split('(')[0].strip()
                
                if param_info:
                    short_name = param_info.get('short_name', '')
                    full_name = param_info.get('name', '')
                    
                    param_descriptions[short_name] = param_info
                    param_descriptions[full_name] = param_info
    
    columns_info = []
    
    for col_name in column_names:
        col_info = {
            'column_name': col_name,
            'column_index': len(columns_info)
        }
        
        col_name_clean = col_name.strip()
        
        if col_name_clean.startswith('Rrs_'):
            wavelength = col_name_clean.split('_')[1].split()[0]
            col_info['wavelength_nm'] = int(wavelength)
            col_info['variable_type'] = 'remote_sensing_reflectance'
            col_info['unit'] = '1/sr'
            col_info['method_device'] = 'Hyperspectral radiometer'
            col_info['description'] = f'Remote sensing reflectance at {wavelength} nm'
        else:
            col_info['variable_type'] = 'unknown'
            
            is_pigment = '[µg/l]' in col_name_clean and 'High Performance Liquid Chrom' in col_name_clean
            
            for key, param_info in param_descriptions.items():
                if key in col_name_clean or col_name_clean in key:
                    col_info.update({
                        'full_description': param_info.get('full_description', ''),
                        'name': param_info.get('name', ''),
                        'short_name': param_info.get('short_name', ''),
                        'unit': param_info.get('unit', ''),
                        'method_device': param_info.get('method_device', ''),
                        'comment': param_info.get('comment', ''),
                        'is_geocode': param_info.get('is_geocode', False),
                        'principal_investigator': param_info.get('principal_investigator', {})
                    })
                    if is_pigment:
                        col_info['variable_type'] = 'pigment'
                        pigment_name = col_name_clean.split('[')[0].strip()
                        col_info['pigment_name'] = pigment_name
                        col_info['description'] = f'Phytoplankton pigment concentration: {pigment_name}'
                    else:
                        col_info['variable_type'] = 'geocode' if param_info.get('is_geocode') else 'measurement'
                    break
            
            if col_info['variable_type'] == 'unknown' or (col_info['variable_type'] == 'measurement' and '[µg/l]' in col_name_clean):
                if '[µg/l]' in col_name_clean and 'High Performance Liquid Chrom' in col_name_clean:
                    col_info['variable_type'] = 'pigment'
                    col_info['unit'] = 'µg/l'
                    col_info['method_device'] = 'High Performance Liquid Chromatography (HPLC)'
                    pigment_name = col_name_clean.split('[')[0].strip()
                    col_info['pigment_name'] = pigment_name
                    col_info['description'] = f'Phytoplankton pigment concentration: {pigment_name}'
                    for key, param_info in param_descriptions.items():
                        if key in pigment_name or pigment_name in key or any(key.lower() in pigment_name.lower() for key in [param_info.get('short_name', ''), param_info.get('name', '')] if key):
                            col_info.update({
                                'full_description': param_info.get('full_description', ''),
                                'name': param_info.get('name', ''),
                                'short_name': param_info.get('short_name', ''),
                                'principal_investigator': param_info.get('principal_investigator', {})
                            })
                            break
                elif 'Campaign' in col_name_clean:
                    col_info['variable_type'] = 'metadata'
                    col_info['description'] = 'Campaign identifier'
                elif 'URL' in col_name_clean or 'ref' in col_name_clean.lower():
                    col_info['variable_type'] = 'metadata'
                    col_info['description'] = 'Reference URL'
                elif 'PI' in col_name_clean and 'Principal' not in col_name_clean:
                    col_info['variable_type'] = 'metadata'
                    col_info['description'] = 'Principal investigator identifier'
                elif 'Date' in col_name_clean or 'Time' in col_name_clean:
                    col_info['variable_type'] = 'geocode'
                    col_info['description'] = 'Date and time of measurement'
                elif 'Latitude' in col_name_clean:
                    col_info['variable_type'] = 'geocode'
                    col_info['unit'] = 'degrees'
                    col_info['description'] = 'Latitude coordinate'
                elif 'Longitude' in col_name_clean:
                    col_info['variable_type'] = 'geocode'
                    col_info['unit'] = 'degrees'
                    col_info['description'] = 'Longitude coordinate'
                elif 'Depth' in col_name_clean:
                    col_info['variable_type'] = 'geocode'
                    col_info['unit'] = 'm'
                    col_info['description'] = 'Water depth'
                elif 'Temp' in col_name_clean:
                    col_info['variable_type'] = 'measurement'
                    col_info['unit'] = '°C'
                    col_info['description'] = 'Water temperature'
                    col_info['method_device'] = 'CTD'
                elif 'Sal' in col_name_clean:
                    col_info['variable_type'] = 'measurement'
                    col_info['unit'] = 'PSU'
                    col_info['description'] = 'Salinity'
                    col_info['method_device'] = 'CTD'
        
        columns_info.append(col_info)
    
    output_data = {
        'total_columns': len(columns_info),
        'columns': columns_info,
        'column_summary': {
            'metadata': len([c for c in columns_info if c['variable_type'] == 'metadata']),
            'geocode': len([c for c in columns_info if c['variable_type'] == 'geocode']),
            'pigment': len([c for c in columns_info if c['variable_type'] == 'pigment']),
            'remote_sensing_reflectance': len([c for c in columns_info if c['variable_type'] == 'remote_sensing_reflectance']),
            'measurement': len([c for c in columns_info if c['variable_type'] == 'measurement']),
            'unknown': len([c for c in columns_info if c['variable_type'] == 'unknown'])
        }
    }
    
    print(f"Extracted information for {len(columns_info)} columns")
    print(f"Column types: {output_data['column_summary']}")
    
    print(f"Saving column headers to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved column headers to {output_file}")
    
    return output_data

if __name__ == "__main__":
    from pathlib import Path
    
    # models/sdp_pigments/training/tools/<this_file>.py -> repo root
    project_root = Path(__file__).resolve().parents[4]
    input_file = project_root / "models" / "sdp_pigments" / "training" / "Kramer-etal_2021.tab"
    output_file = project_root / "models" / "sdp_pigments" / "training" / "Kramer-etal_2021_column_headers.json"
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
    else:
        column_data = extract_column_headers(str(input_file), str(output_file))
        print(f"\nColumn header summary:")
        print(f"  Total columns: {column_data['total_columns']}")
        print(f"  Column types breakdown:")
        for col_type, count in column_data['column_summary'].items():
            print(f"    {col_type}: {count}")
