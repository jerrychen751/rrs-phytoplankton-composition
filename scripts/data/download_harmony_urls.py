import json
import requests
import os
import sys
from pathlib import Path

EARTHACCESS_TOKEN = os.getenv("EARTHACCESS_TOKEN")

def get_output_dir():
    """Returns a path to Downloads/pace_harmony_data inside the user's home directory."""
    home = Path.home()
    downloads = home / "Downloads"
    target_dir = downloads / "pace_harmony_data"
    return target_dir

def get_json_input():
    """Reads JSON from stdin until EOF."""
    print("-" * 60)
    print("PASTE INSTRUCTIONS:")
    print("1. Paste the full Harmony JSON response below.")
    print("2. Press ENTER to ensure you are on a new line.")
    print("3. Signal end of input:")
    print("   - Windows: Press Ctrl-Z then Enter")
    print("   - Mac/Linux: Press Ctrl-D")
    print("-" * 60)
    print("Waiting for input...")
    
    try:
        return sys.stdin.read()
    except KeyboardInterrupt:
        return ""

def download_harmony_files():
    if not EARTHACCESS_TOKEN:
        raise RuntimeError(
            "Missing Earthdata token: set EARTHACCESS_TOKEN in your environment.\n"
            "Example:\n"
            "  export EARTHACCESS_TOKEN='...'\n"
        )
    # 1. Get JSON from user paste
    json_str = get_json_input()
    
    if not json_str.strip():
        print("\nNo input received. Exiting.")
        return

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"\nError: Invalid JSON content provided.\nDetails: {e}")
        return

    # 2. Setup Output Directory
    output_dir = get_output_dir()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated directory: {output_dir}")
    else:
        print(f"\nUsing existing directory: {output_dir}")

    links = data.get('links', [])
    print(f"Found {len(links)} total links in the response.")

    # 3. Configure Session with Authentication
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {EARTHACCESS_TOKEN}',
        'User-Agent': 'HarmonyDownloadScript/1.0'
    })

    # 4. Download Loop
    count = 0
    for link in links:
        # Only process actual data links (ignore STAC/Self links)
        if link.get('rel') == 'data':
            url = link.get('href')
            filename = link.get('title')
            
            # Fallback filename if title is missing
            if not filename:
                filename = url.split('/')[-1]

            filepath = output_dir / filename
            
            print(f"\n[{count+1}] Downloading: {filename}")
            # print(f"      URL: {url}") # Optional: Uncomment to see URLs

            try:
                # Stream download to handle large files without memory issues
                with session.get(url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    
                    with open(filepath, 'wb') as f:
                        if total_size == 0:
                            f.write(r.content)
                        else:
                            downloaded = 0
                            for chunk in r.iter_content(chunk_size=8192 * 4): # Increased chunk size for speed
                                f.write(chunk)
                                downloaded += len(chunk)
                                # Simple progress indicator
                                percent = (downloaded / total_size) * 100
                                print(f"      Progress: {percent:.1f}%", end='\r')
                
                print(f"\n      ✅ Saved to {filepath}")
                count += 1
                
            except Exception as e:
                print(f"\n      ❌ Error downloading {filename}: {e}")

    print(f"\nDone! Successfully downloaded {count} files to {output_dir}.")

if __name__ == "__main__":
    download_harmony_files()
