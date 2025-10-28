import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import logging
import json
import time

# --- 1. Configuration ---
HERE_API_KEY = os.environ.get("HERE_API_KEY")

# API Endpoint
TRAFFIC_FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"

# File I/O
PEMS_META_FILE = 'PEMS-BAY-META.csv' # Used to get the bounding box
INPUT_MAPPING_FILE = 'pems_here_mapping.csv' # Your new mapping results
GEOMETRY_CACHE_FILE = 'geometry_dict.json' # New cache for shapes
OUTPUT_VIZ_FILE = 'pems_here_mapping_visualization.png'

# --- 2. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mapping_pipeline.log", mode='a'), # Append to main log
        logging.StreamHandler()
    ]
)

# --- 3. Helper Functions ---

def load_pems_data(filename=PEMS_META_FILE):
    """Loads PeMS metadata to calculate the bounding box."""
    logging.info(f"Loading PeMS metadata from {filename} for bounding box...")
    try:
        if not os.path.exists(filename):
            logging.error(f"FATAL: PeMS metadata file not found at '{filename}'")
            return None
        pems_df = pd.read_csv(filename)
        pems_df = pems_df.dropna(subset=['Latitude', 'Longitude'])
        return pems_df
    except Exception as e:
        logging.error(f"Could not load PeMS data: {e}", exc_info=True)
        return None

def get_bounding_box(pems_df):
    """Calculates the bounding box string from PeMS metadata."""
    lat_min, lat_max = pems_df['Latitude'].min(), pems_df['Latitude'].max()
    lon_min, lon_max = pems_df['Longitude'].min(), pems_df['Longitude'].max()
    buffer = 0.01
    bbox_str = f"{lon_min - buffer},{lat_min - buffer},{lon_max + buffer},{lat_max + buffer}"
    return f"bbox:{bbox_str}"

def fetch_traffic_shapes(bbox_str):
    """
    Calls the Traffic API to get segments with TMC and Shape information.
    """
    params = {
        "in": bbox_str,
        "locationReferencing": "tmc,shape", # Request TMC and Shape
        "apiKey": HERE_API_KEY
    }
    logging.info(f"Calling Traffic API for shapes with bbox: {bbox_str[:50]}...")
    
    try:
        resp = requests.get(TRAFFIC_FLOW_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.warning("Traffic API rate limit hit. Waiting 10s...")
            time.sleep(10)
            return fetch_traffic_shapes(bbox_str) # Retry
        logging.error(f"Traffic API HTTP Error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Traffic API Network Error: {e}", exc_info=True)
    return None

def build_geometry_dictionary(bbox_str):
    """
    Builds a dictionary mapping TMC locationId to its Shapely LineString.
    """
    # Check for cache first
    if os.path.exists(GEOMETRY_CACHE_FILE):
        logging.info(f"Loading geometry dictionary from cache: {GEOMETRY_CACHE_FILE}")
        with open(GEOMETRY_CACHE_FILE, 'r') as f:
            # Keys are strings, values are WKT strings
            data = json.load(f)
            return {key: LineString(json.loads(value)) for key, value in data.items()}

    logging.info("Building new geometry dictionary (this may take a moment)...")
    data = fetch_traffic_shapes(bbox_str)
    
    if not data or "results" not in data:
        logging.error("Failed to fetch data for geometry dictionary. Exiting.")
        return {}

    geometry_dict = {}
    
    for seg in data["results"]:
        try:
            loc = seg.get("location", {})
            tmc_id = loc.get("tmc", {}).get("locationId")
            shape_block = loc.get("shape", {})
            
            if not tmc_id or not shape_block or str(tmc_id) in geometry_dict:
                continue # Skip segments without both refs or if already processed
            
            all_points_coords = []
            for link in shape_block.get("links", []):
                for p in link.get("points", []):
                    all_points_coords.append((p.get('lng'), p.get('lat'))) # Lon, Lat
            
            if len(all_points_coords) >= 2:
                geometry_dict[str(tmc_id)] = LineString(all_points_coords)

        except Exception as e:
            logging.warning(f"Error parsing segment for geometry dict: {e}")
            
    logging.info(f"Successfully built geometry dictionary with {len(geometry_dict)} entries.")
    
    # Cache the dictionary (storing geometry as JSON-serializable list of coords)
    try:
        # Convert LineString to a format JSON can save (list of coordinate tuples)
        serializable_dict = {key: json.dumps(list(value.coords)) for key, value in geometry_dict.items()}
        with open(GEOMETRY_CACHE_FILE, 'w') as f:
            json.dump(serializable_dict, f)
        logging.info(f"Saved geometry dictionary to cache: {GEOMETRY_CACHE_FILE}")
    except Exception as e:
        logging.warning(f"Could not save geometry dictionary cache: {e}")

    return geometry_dict

def load_mapping_results(filename=INPUT_MAPPING_FILE):
    """Loads the CSV output from run_mapping.py"""
    logging.info(f"Loading mapping results from {filename}...")
    try:
        if not os.path.exists(filename):
            logging.error(f"FATAL: Mapping file not found at '{filename}'")
            return None
        
        # Read the CSV, ensuring here_locationId is treated as a string
        df = pd.read_csv(filename, dtype={'here_locationId': 'str'})
        logging.info(f"Loaded {len(df)} mapping results.")
        return df
    except Exception as e:
        logging.error(f"Could not load mapping results: {e}", exc_info=True)
        return None

def plot_results(pems_gdf, matched_segments_gdf, failed_sensors_gdf, filename):
    """
    Generates a Matplotlib plot of all sensors and matched segments.
    """
    logging.info(f"Generating visualization: {filename}...")
    try:
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # 1. Plot all PeMS sensors as small grey dots (background)
        pems_gdf.plot(
            ax=ax, color='grey', markersize=2, label='All PeMS Sensors', zorder=1, alpha=0.3
        )

        # 2. Plot the HERE road segments (LineStrings)
        if not matched_segments_gdf.empty:
            matched_segments_gdf.plot(
                ax=ax, color='blue', linewidth=1.5, 
                label='Matched HERE Segments (TMC)', zorder=2, alpha=0.8
            )

        # 3. Plot the FAILED PeMS sensors (Points)
        if not failed_sensors_gdf.empty:
            failed_sensors_gdf.plot(
                ax=ax, color='red', marker='x', markersize=40, 
                label=f"Failed Sensors ({len(failed_sensors_gdf)})", zorder=3
            )

        # 4. Plot the SUCCESSFUL PeMS sensors (Points)
        if not pems_gdf[pems_gdf['match_status'] == 'Success'].empty:
            pems_gdf[pems_gdf['match_status'] == 'Success'].plot(
                ax=ax, color='lime', marker='o', markersize=10, 
                label=f"Successful Sensors ({len(pems_gdf[pems_gdf['match_status'] == 'Success'])})", zorder=4,
                edgecolors='black', linewidth=0.5
            )

        ax.set_title(f"PeMS to HERE Mapping Validation (Success vs. Failures)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Set map bounds
        minx, miny, maxx, maxy = pems_gdf.total_bounds
        ax.set_xlim(minx - 0.05, maxx + 0.05)
        ax.set_ylim(miny - 0.05, maxy + 0.05)

        ax.legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        logging.info(f"Visualization saved to {filename}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Failed to generate visualization: {e}", exc_info=True)

# --- 4. Main Execution ---
def main():
    if not HERE_API_KEY:
        logging.error("FATAL: HERE_API_KEY environment variable not set.")
        return

    logging.info("--- Starting Mapping Visualization Process ---")
    
    # 1. Load PeMS data to get bounding box
    pems_meta_df = load_pems_data()
    if pems_meta_df is None:
        return
        
    # 2. Load the mapping results from run_mapping.py
    mapping_df = load_mapping_results()
    if mapping_df is None:
        return
        
    # 3. Build/Load the geometry dictionary (TMC ID -> LineString)
    geometry_dict = build_geometry_dictionary(get_bounding_box(pems_meta_df))
    if not geometry_dict:
        logging.error("FATAL: Could not build geometry dictionary. Exiting.")
        return
    logging.info(f"Loaded geometry dictionary with {len(geometry_dict)} shapes.")

    # 4. Create GeoDataFrames
    
    # Create GDF for ALL PeMS sensors (as points)
    pems_gdf = gpd.GeoDataFrame(
        mapping_df,
        geometry=gpd.points_from_xy(mapping_df.pems_lon, mapping_df.pems_lat),
        crs="EPSG:4326"
    )
    
    # Filter for successfully matched sensors
    matched_sensors_df = mapping_df[mapping_df['match_status'] == 'Success'].copy()
    
    # Filter for failed sensors
    failed_sensors_gdf = pems_gdf[pems_gdf['match_status'] != 'Success'].copy()
    
    # Create the GDF for the matched HERE segments (as lines)
    # Map the TMC ID to its geometry from our dictionary
    matched_sensors_df['geometry'] = matched_sensors_df['here_locationId'].map(geometry_dict)
    
    # Convert this DataFrame into a GeoDataFrame
    matched_segments_gdf = gpd.GeoDataFrame(
        matched_sensors_df,
        geometry='geometry',
        crs="EPSG:4326"
    )
    
    # Clean up any rows where the TMC ID didn't have a geometry in our dict
    original_count = len(matched_segments_gdf)
    matched_segments_gdf = matched_segments_gdf.dropna(subset=['geometry'])
    dropped_count = original_count - len(matched_segments_gdf)
    if dropped_count > 0:
        logging.warning(f"Could not find geometries for {dropped_count} matched TMC IDs.")

    # 5. Plot the results
    plot_results(pems_gdf, matched_segments_gdf, failed_sensors_gdf, OUTPUT_VIZ_FILE)
    
    logging.info("--- Visualization Complete ---")

if __name__ == "__main__":
    main()
