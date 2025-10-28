import os
import requests
import pandas as pd
import logging
import json
import math
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. Configuration ---

# Get API key from environment variable
HERE_API_KEY = os.environ.get("HERE_API_KEY")

# API Endpoints
ROUTING_API_URL = "https://router.hereapi.com/v8/routes"

# File I/O
PEMS_META_FILE = 'PEMS-BAY-META.csv'
OUTPUT_MAPPING_FILE = 'pems_here_mapping.csv' # Changed output filename
DICT_CACHE_FILE = 'here_translation_dict.json' # <-- Input dictionary

# --- 2. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mapping_pipeline.log"), # Append to existing log
        logging.StreamHandler()
    ]
)

# --- 3. Helper Functions ---

def parse_segment_ref(ref_string):
    """
    Parses a HERE segmentRef string to extract the internal segment ID
    and the direction symbol (+ or -).
    """
    if not isinstance(ref_string, str):
        return None, None
        
    # Pattern: :(\d+)[#?]([+-])
    match = re.search(r":(\d+)[#?]([+-])", ref_string)
    
    if match:
        internal_id = match.group(1)
        direction_symbol = match.group(2)
        return internal_id, direction_symbol
    else:
        match_id_only = re.search(r":(\d+)(?:#|$)", ref_string)
        if match_id_only:
            return match_id_only.group(1), None
        
    return None, None

def get_bearing_from_direction(pems_dir):
    """Converts PeMS cardinal direction to a bearing in degrees."""
    directions = {
        'N': 0, 'E': 90, 'S': 180, 'W': 270,
        'NE': 45, 'NW': 315, 'SE': 135, 'SW': 225
    }
    return directions.get(str(pems_dir).upper(), None)

def calculate_destination(lat, lon, bearing, distance_meters=50):
    """
    Calculates a destination lat/lon given a start point, bearing, and distance.
    """
    R = 6371000  # Earth radius in meters
    try:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)

        dest_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_meters / R) +
                                 math.cos(lat_rad) * math.sin(distance_meters / R) * math.cos(bearing_rad))
        
        dest_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_meters / R) * math.cos(lat_rad),
                                         math.cos(distance_meters / R) - math.sin(lat_rad) * math.sin(dest_lat_rad))
        
        return math.degrees(dest_lat_rad), math.degrees(dest_lon_rad)
    except (ValueError, TypeError, AttributeError):
        logging.warning(f"Could not calculate destination for {lat}, {lon}, {bearing}", exc_info=True)
        return None, None

# --- 4. Step 1: Load Prerequisite Data ---

def load_pems_data(filename=PEMS_META_FILE):
    """Loads and cleans the PeMS metadata file."""
    logging.info(f"Loading PeMS metadata from {filename}...")
    try:
        if not os.path.exists(filename):
            logging.error(f"FATAL: PeMS metadata file not found at '{filename}'")
            return None
            
        pems_df = pd.read_csv(filename)
        pems_df = pems_df.dropna(subset=['Latitude', 'Longitude', 'sensor_id'])
        pems_df = pems_df.drop_duplicates(subset=['sensor_id'], keep='first')
        pems_df['sensor_id'] = pems_df['sensor_id'].astype(int).astype(str)
        pems_df['Dir'] = pems_df['Dir'].astype(str).str.upper()
        logging.info(f"Loaded {len(pems_df)} unique PeMS sensors.")
        return pems_df
    except Exception as e:
        logging.error(f"Could not load PeMS data: {e}", exc_info=True)
        return None

def load_translation_dictionary(cache_file=DICT_CACHE_FILE):
    """Loads the translation dictionary from the cache file."""
    if not os.path.exists(cache_file):
        logging.error(f"FATAL: Dictionary cache file '{cache_file}' not found.")
        logging.error("Please run 'build_dictionary.py' first to create it.")
        return None
        
    logging.info(f"Loading translation dictionary from cache: {cache_file}")
    with open(cache_file, 'r') as f:
        return json.load(f)

# --- 5. Step 2: Map PeMS Sensors via Routing API ---

### MODIFICATION ###
# Renamed function slightly for clarity
def get_route_spans_info(lat, lon, pems_dir):
    """
    Calls the Routing API for a single sensor and returns a LIST 
    of (internal_id, direction_symbol) tuples for ALL spans in the route.
    Returns an empty list if routing fails or no spans are found.
    Also returns a status message.
    """
    bearing = get_bearing_from_direction(pems_dir)
    if bearing is None:
        return [], f"Invalid PeMS Dir: {pems_dir}"
        
    # Increase distance slightly to potentially cross more spans if needed
    start_lat, start_lon = calculate_destination(lat, lon, bearing - 180, 50) 
    end_lat, end_lon = calculate_destination(lat, lon, bearing, 100) # Increased to 100m ahead

    if start_lat is None or end_lat is None:
        return [], "Waypoint calculation failed"

    params = {
        'transportMode': 'car',
        'origin': f"{start_lat:.6f},{start_lon:.6f}",
        'destination': f"{end_lat:.6f},{end_lon:.6f}",
        'return': 'polyline',
        'spans': 'segmentRef',
        'apiKey': HERE_API_KEY
    }
    
    try:
        resp = requests.get(ROUTING_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Check for valid route and sections
        if not data.get('routes') or not data['routes'][0].get('sections'):
            return [], "No route found"
            
        sections = data['routes'][0]['sections']
        if not sections or 'spans' not in sections[0]:
             return [], "No spans found in route section"

        spans_info = []
        for span in sections[0]['spans']:
            ref_string = span.get('segmentRef')
            internal_id, direction_symbol = parse_segment_ref(ref_string)
            if internal_id: # Only add if we could parse an ID
                spans_info.append((internal_id, direction_symbol))
        
        if not spans_info:
             return [], "Could not parse any segmentRefs from spans"
             
        return spans_info, "Success (Got Spans)" # Return the list of tuples

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            time.sleep(1) # Simple backoff
            return get_route_spans_info(lat, lon, pems_dir) # Retry
        return [], f"HTTP Error {e.response.status_code}"
    except Exception as e:
        return [], f"Routing Error: {str(e)[:50]}"

def map_pems_sensors(pems_df, translation_dict):
    """
    Orchestrates Step 2: Iterates PeMS sensors, calls Routing API (getting all spans),
    checks spans against the dictionary, and builds the final mapping.
    """
    logging.info(f"Starting to map {len(pems_df)} sensors (using algorithmic fix)...")
    mapping_results = []
    
    max_workers = 10 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            ### MODIFICATION ### - Call the updated function
            executor.submit(get_route_spans_info, row.Latitude, row.Longitude, row.Dir): row
            for row in pems_df.itertuples() if pd.notna(row.Latitude) and pd.notna(row.Longitude)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mapping Sensors"):
            row = futures[future]
            try:
                ### MODIFICATION ### - Result is now a list and status
                spans_info_list, status = future.result()
                
                # Default result for failures before span checking
                result = {
                    'sensor_id': row.sensor_id, 'pems_dir': row.Dir,
                    'pems_lat': row.Latitude, 'pems_lon': row.Longitude,
                    'here_locationId': None, 'here_queuingDirection': None,
                    'match_status': status, 'internal_segment_id': None, 
                    'internal_travel_dir': None
                }

                found_match = False
                ### MODIFICATION ### - Iterate through the returned spans
                if status == "Success (Got Spans)":
                    for internal_id, direction_symbol in spans_info_list:
                        tmc_id = translation_dict.get(internal_id)
                        
                        # Check if this ID is in the dictionary AND has a direction symbol
                        if tmc_id and direction_symbol: 
                            result['here_locationId'] = tmc_id
                            
                            # Apply "Critical Inversion"
                            queuing_dir = '+' if direction_symbol == '−' else '−'
                            result['here_queuingDirection'] = queuing_dir
                            
                            # Store the ID and symbol that *worked*
                            result['internal_segment_id'] = internal_id
                            result['internal_travel_dir'] = direction_symbol
                            result['match_status'] = "Success" 
                            found_match = True
                            break # Found the first valid match, stop checking spans

                    # If loop finished without finding a match in the dictionary
                    if not found_match:
                         # Store the first ID found by routing, even if unmapped
                         if spans_info_list:
                             first_id, first_dir = spans_info_list[0]
                             result['internal_segment_id'] = first_id
                             result['internal_travel_dir'] = first_dir
                         result['match_status'] = "Dictionary Miss (No Match in Spans)"
                
                mapping_results.append(result)

            except Exception as e:
                logging.warning(f"Error processing sensor {row.sensor_id}: {e}")
                # Append error status
                mapping_results.append({
                    'sensor_id': row.sensor_id, 'pems_dir': row.Dir,
                    'pems_lat': row.Latitude, 'pems_lon': row.Longitude,
                    'match_status': f"Task Error: {e}" 
                    # Other fields default to None
                })


    return pd.DataFrame(mapping_results)


# --- 6. Main Execution ---
def main():
    if not HERE_API_KEY:
        logging.error("FATAL: HERE_API_KEY environment variable not set.")
        return

    logging.info("--- Starting PeMS Sensor Mapping Process (with Algorithmic Fix) ---")
    
    # Step 1: Load prerequisite data
    pems_df = load_pems_data()
    if pems_df is None:
        return

    dictionary = load_translation_dictionary()
    if dictionary is None:
        return
    logging.info(f"Loaded dictionary with {len(dictionary)} entries.")

    # Step 2: Map all sensors
    final_mapping_df = map_pems_sensors(pems_df, dictionary)

    # Step 3: Save results
    try:
        final_mapping_df.to_csv(OUTPUT_MAPPING_FILE, index=False)
        logging.info(f"\n--- Pipeline Complete ---")
        logging.info(f"Successfully saved final mapping to {OUTPUT_MAPPING_FILE}")
        
        success_count = len(final_mapping_df[final_mapping_df['match_status'] == 'Success'])
        total_count = len(final_mapping_df)
        logging.info(f"Successfully mapped {success_count} / {total_count} sensors.")
        
        logging.info("\nFailure summary (Top 10):")
        # Display counts of non-Success statuses
        failure_counts = final_mapping_df[final_mapping_df['match_status'] != 'Success']['match_status'].value_counts()
        logging.info(failure_counts.nlargest(10))


    except Exception as e:
        logging.error(f"FATAL: Could not save final mapping file: {e}", exc_info=True)

if __name__ == "__main__":
    main()

