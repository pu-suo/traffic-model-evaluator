import os
import requests
import pandas as pd
import logging
import json
import re
import time

# --- 1. Configuration ---
HERE_API_KEY = os.environ.get("HERE_API_KEY")
TRAFFIC_FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"
PEMS_META_FILE = 'PEMS-BAY-META.csv'
DICT_CACHE_FILE = 'here_translation_dict.json'

# --- 2. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mapping_pipeline.log"),
        logging.StreamHandler()
    ]
)

# --- 3. Helper Functions ---

def parse_segment_ref(ref_string):
    """
    Parses a HERE segmentRef string to extract the internal segment ID.
    We only need the ID, not the direction, for the dictionary.
    """
    if not isinstance(ref_string, str):
        return None
        
    # Pattern: :(\d+)(?:#|$)
    # :(\d+)     - Capture digits (the ID) that follow a colon
    # (?:#|$)   - Match a '#' or the end of the string
    match = re.search(r":(\d+)(?:#|$)", ref_string)
    
    if match:
        return match.group(1)
        
    return None

def load_pems_data(filename=PEMS_META_FILE):
    """Loads and cleans the PeMS metadata file to get a bounding box."""
    logging.info(f"Loading PeMS metadata from {filename}...")
    try:
        if not os.path.exists(filename):
            logging.error(f"FATAL: PeMS metadata file not found at '{filename}'")
            return None
            
        pems_df = pd.read_csv(filename)
        pems_df = pems_df.dropna(subset=['Latitude', 'Longitude'])
        logging.info(f"Loaded {len(pems_df)} PeMS sensor locations.")
        return pems_df
    except Exception as e:
        logging.error(f"Could not load PeMS data: {e}", exc_info=True)
        return None

def get_bounding_box(pems_df):
    """Calculates the bounding box string from PeMS metadata."""
    lat_min, lat_max = pems_df['Latitude'].min(), pems_df['Latitude'].max()
    lon_min, lon_max = pems_df['Longitude'].min(), pems_df['Longitude'].max()
    buffer = 0.01 # Add buffer to ensure segments are included
    bbox_str = f"{lon_min - buffer},{lat_min - buffer},{lon_max + buffer},{lat_max + buffer}"
    return f"bbox:{bbox_str}"

def fetch_traffic_data_for_dict(bbox_str):
    """
    Calls the Traffic API to get segments with BOTH
    TMC and segmentRef information.
    """
    params = {
        "in": bbox_str,
        "locationReferencing": "tmc,segmentRef", # The key parameter!
        "apiKey": HERE_API_KEY
    }
    logging.info(f"Calling Traffic API with bbox: {bbox_str[:50]}...")
    
    try:
        resp = requests.get(TRAFFIC_FLOW_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.warning("Traffic API rate limit hit. Waiting 10s...")
            time.sleep(10)
            return fetch_traffic_data_for_dict(bbox_str) # Retry
        logging.error(f"Traffic API HTTP Error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Traffic API Network Error: {e}", exc_info=True)
    return None

def build_translation_dictionary(pems_df):
    """
    Orchestrates Step 1: Calls Traffic API and builds the
    Internal Segment ID -> TMC Location ID dictionary.
    """
    logging.info("Building new translation dictionary (this may take a moment)...")
    bbox = get_bounding_box(pems_df)
    data = fetch_traffic_data_for_dict(bbox)
    
    if not data or "results" not in data:
        logging.error("Failed to fetch data for dictionary. Exiting.")
        return {}

    translation_dict = {}
    segments_processed = 0
    
    for seg in data["results"]:
        try:
            loc = seg.get("location", {})
            tmc_id = loc.get("tmc", {}).get("locationId")
            segment_ref_block = loc.get("segmentRef", {})
            
            if not tmc_id or not segment_ref_block:
                continue # Skip segments without both refs
                
            for segment in segment_ref_block.get("segments", []):
                ref_string = segment.get("ref")
                internal_id = parse_segment_ref(ref_string) # We only need the ID
                
                if internal_id and tmc_id:
                    if internal_id not in translation_dict:
                        translation_dict[internal_id] = str(tmc_id)
                    segments_processed += 1

        except Exception as e:
            logging.warning(f"Error parsing segment for dictionary: {e}", exc_info=True)
            
    logging.info(f"Successfully built translation dictionary with {len(translation_dict)} unique entries.")
    
    # Cache the dictionary
    try:
        with open(DICT_CACHE_FILE, 'w') as f:
            json.dump(translation_dict, f)
        logging.info(f"Saved dictionary to cache: {DICT_CACHE_FILE}")
    except Exception as e:
        logging.warning(f"Could not save dictionary cache: {e}")

    return translation_dict

# --- 4. Main Execution ---
def main():
    if not HERE_API_KEY:
        logging.error("FATAL: HERE_API_KEY environment variable not set.")
        return
        
    logging.info("--- Starting Dictionary Builder ---")
    
    pems_df = load_pems_data()
    if pems_df is None:
        return

    build_translation_dictionary(pems_df)
    logging.info("--- Dictionary Builder Finished ---")

if __name__ == "__main__":
    main()
