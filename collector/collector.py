# collector/collector.py
import pandas as pd
import requests
import json
import os
import time
from datetime import datetime, timedelta, timezone
import sqlalchemy
import boto3
import logging
import sys
import math

# --- Configuration ---
# *** USE CORRECT SECRET NAMES ***
SECRET_NAME_HERE_API = "HereTrafficApiKey"
SECRET_NAME_DB_CREDS = "HereTrafficDbCredentials"
AWS_REGION = "us-east-2"

MAPPING_FILE_PATH = 'pems_here_mapping.csv'
HERE_FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"
REQUEST_TIMEOUT = 30
MAX_IDS_PER_REQUEST = 500
COLLECTION_INTERVAL_MINUTES = 5
RETENTION_DAYS = 7
METERS_PER_SECOND_TO_MPH = 2.23694

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Global Variables ---
pems_to_here_map = {}
db_engine = None
secrets_cache = {}
# *** ADD GLOBAL VARIABLE FOR BOUNDING BOX ***
global_bounding_box = None

# --- AWS Secrets Manager ---
def get_secret_from_aws(secret_name, region_name=AWS_REGION):
    """Retrieves secret value ONLY from AWS Secrets Manager."""
    if secret_name in secrets_cache: return secrets_cache[secret_name]

    logging.info(f"Attempting to retrieve secret from AWS Secrets Manager: {secret_name} in {region_name}")
    # Use default session which should pick up IAM role credentials on EC2
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        logging.info(f"Successfully retrieved secret from AWS SM: {secret_name}")
        # ...(rest of parsing logic - unchanged)...
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            try: # Try parsing as JSON
                parsed_secret = json.loads(secret)
                secrets_cache[secret_name] = parsed_secret
                return parsed_secret
            except json.JSONDecodeError: # Fallback to plain string
                secrets_cache[secret_name] = secret
                return secret
        else:
            logging.warning(f"Secret '{secret_name}' from AWS SM might be binary.")
            return None
    except client.exceptions.ResourceNotFoundException:
        logging.error(f"Secret '{secret_name}' not found in AWS Secrets Manager region {region_name}. Ensure name is correct ('{SECRET_NAME_HERE_API}', '{SECRET_NAME_DB_CREDS}') and exists.")
        return None
    except Exception as e:
        logging.error(f"Failed to retrieve secret '{secret_name}' from AWS SM: {e}", exc_info=False)
        return None


# --- Mapping Loader ---
def load_mapping(filepath=MAPPING_FILE_PATH):
    """
    Loads the PeMS-HERE mapping CSV, stores the map, AND calculates the bounding box.
    Returns True on success.
    """
    global pems_to_here_map, global_bounding_box
    try:
        logging.info(f"Loading PeMS-HERE mapping from '{filepath}'...")
        if not os.path.isabs(filepath):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             filepath = os.path.join(script_dir, filepath)
        if not os.path.exists(filepath): raise FileNotFoundError(f"Mapping file not found at '{filepath}'.")

        mapping_df = pd.read_csv(filepath)
        # --- Use correct column names from user's file ---
        # Assume standard case or try lower if standard fails
        required_cols_std = ['sensor_id', 'pems_lat', 'pems_lon', 'here_locationId', 'here_queuingDirection']
        required_cols = []
        rename_map_needed = {}
        mapping_df_cols_lower = {c.lower(): c for c in mapping_df.columns}

        for std_col in required_cols_std:
             std_col_lower = std_col.lower()
             if std_col in mapping_df.columns:
                 required_cols.append(std_col)
             elif std_col_lower in mapping_df_cols_lower:
                 original_case_col = mapping_df_cols_lower[std_col_lower]
                 rename_map_needed[original_case_col] = std_col # Rename to standard
                 required_cols.append(std_col) # Use standard name going forward
             else:
                 raise ValueError(f"Mapping file must contain column like: {std_col}")

        if rename_map_needed:
            logging.debug(f"Renaming columns for consistency: {rename_map_needed}")
            mapping_df = mapping_df.rename(columns=rename_map_needed)

        # Ensure lat/lon columns exist for bbox calculation
        if 'pems_lat' not in required_cols or 'pems_lon' not in required_cols:
             raise ValueError("Mapping file must contain 'pems_lat' and 'pems_lon' columns for bounding box calculation.")

        # --- Calculate Bounding Box ---
        temp_df_for_bbox = mapping_df.dropna(subset=['pems_lat', 'pems_lon'])
        if temp_df_for_bbox.empty:
            raise ValueError("No valid lat/lon pairs found in mapping file to calculate bounding box.")
        lat_min, lat_max = temp_df_for_bbox['pems_lat'].min(), temp_df_for_bbox['pems_lat'].max()
        lon_min, lon_max = temp_df_for_bbox['pems_lon'].min(), temp_df_for_bbox['pems_lon'].max()
        buffer = 0.01
        global_bounding_box = f"bbox:{lon_min - buffer},{lat_min - buffer},{lon_max + buffer},{lat_max + buffer}"
        logging.info(f"Calculated global bounding box for API calls: {global_bounding_box}")
        # --- End BBox Calc ---

        # Filter out rows missing essential mapping info and ensure correct types
        mapping_df = mapping_df.dropna(subset=['sensor_id', 'here_locationId', 'here_queuingDirection'])
        mapping_df['sensor_id'] = mapping_df['sensor_id'].astype(str).str.strip()
        mapping_df['here_locationId'] = mapping_df['here_locationId'].apply(lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (float, int)) else str(x)).str.strip()
        mapping_df['here_queuingDirection'] = mapping_df['here_queuingDirection'].astype(str).str.strip()
        valid_dirs = ['+', '-','−']
        mapping_df = mapping_df[mapping_df['here_queuingDirection'].isin(valid_dirs)]

        temp_map = {}
        for _, row in mapping_df.iterrows():
            q_dir = row['here_queuingDirection']
            if q_dir == '−': q_dir = '-'
            temp_map[row['sensor_id']] = {'here_loc_id': row['here_locationId'], 'here_q_dir': q_dir}

        pems_to_here_map = temp_map
        logging.info(f"Successfully loaded mapping for {len(pems_to_here_map)} sensors.")
        if len(pems_to_here_map) == 0: logging.warning("Mapping resulted in zero valid sensor mappings.")
        return True

    except Exception as e:
        logging.error(f"CRITICAL: Failed to load or parse mapping file '{filepath}': {e}", exc_info=True)
        pems_to_here_map = {}
        global_bounding_box = None
        return False

# --- Database Connection ---
def connect_db():
    """Establishes or verifies the database connection using SQLAlchemy."""
    global db_engine
    if db_engine is not None:
        try:
             with db_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                 connection.execute(sqlalchemy.text("SELECT 1"))
             return db_engine
        except sqlalchemy.exc.OperationalError:
             logging.warning("DB connection test failed, attempting reconnect...")
             db_engine = None
        except Exception as e:
             logging.error(f"Unexpected error testing DB connection: {e}", exc_info=False)
             db_engine = None

    if db_engine is None:
        logging.info("Attempting to establish database connection using AWS Secrets Manager...")
        DATABASE_URL = None
        
        db_creds = get_secret_from_aws(SECRET_NAME_DB_CREDS)
        if not isinstance(db_creds, dict):
            logging.error(f"Failed to retrieve/parse DB credentials secret '{SECRET_NAME_DB_CREDS}' from AWS SM.")
            return None

        db_user = db_creds.get('username')
        db_pass = db_creds.get('password')
        db_host = db_creds.get('host')
        db_port = db_creds.get('port', 5432)
        db_name = db_creds.get('dbname')

        if not all([db_user, db_pass, db_host, db_name]):
            logging.error("DB credentials incomplete in AWS secret. Ensure 'username', 'password', 'host', and 'dbname' are set.")
            return None
            
        DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        logging.info(f"Using DB credentials from AWS Secrets Manager (Host: {db_host})")

        if DATABASE_URL:
            try:
                db_engine = sqlalchemy.create_engine(
                    DATABASE_URL, pool_pre_ping=True, echo=False,
                    connect_args={'connect_timeout': 10}
                )
                with db_engine.connect() as connection:
                    logging.info("Database connection successful.")
                return db_engine
            except Exception as e:
                logging.error(f"Failed to create database engine or connect: {e}", exc_info=True)
                db_engine = None
                return None
        else:
            logging.error("Could not determine database connection string.")
            return None
    return db_engine


# --- HERE API Fetcher ---
def fetch_here_flow_data(location_ids, api_key, bbox):
    """Fetches current flow data from HERE API using bbox and locationIds."""
    if not location_ids:
        logging.warning("fetch_here_flow_data: No location IDs provided.")
        return {}
    if not api_key:
        logging.error("fetch_here_flow_data: HERE API Key is missing.")
        return {}
    if not bbox:
        logging.error("fetch_here_flow_data: Bounding box is missing.")
        return {}

    all_flow_data = {}
    location_id_list = sorted(list(location_ids))
    num_chunks = math.ceil(len(location_id_list) / MAX_IDS_PER_REQUEST)

    logging.info(f"Fetching HERE flow data for {len(location_id_list)} IDs (using bbox='{bbox}')...")

    for i in range(0, len(location_id_list), MAX_IDS_PER_REQUEST):
        chunk_ids = location_id_list[i:i + MAX_IDS_PER_REQUEST]
        params = {
            "in": bbox, # Mandatory spatial filter
            "locationIds": ",".join(chunk_ids), # Specific IDs to retrieve within bbox
            "locationReferencing": "tmc",
            "functionalClasses": "1,2,3,4,5",
            "apiKey": api_key
        }
        retries = 3
        backoff_factor = 5
        chunk_num = i // MAX_IDS_PER_REQUEST + 1
        chunk_success = False

        while retries > 0:
            try:
                logging.debug(f"Fetching HERE chunk {chunk_num}/{num_chunks}...")
                resp = requests.get(HERE_FLOW_URL, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                
                #DEBUGGING
                if results: # If we got any results at all
                    logging.info("--- DEBUG: FIRST RESULT FROM API ---")
                    logging.info(json.dumps(results[0], indent=2))
                    logging.info("--- END DEBUG ---")

                processed_in_chunk = 0
                for segment_data in results:
                    loc = segment_data.get("location")
                    current_flow = segment_data.get("currentFlow")
                    tmc_info = loc.get("tmc") if loc else None # Get the 'tmc' sub-object
                    
                    if tmc_info and current_flow:
                        tmc_loc_id = tmc_info.get("locationId") # <-- GET TMC ID
                        
                        if tmc_loc_id:
                           # Convert API's integer ID to string to match our list
                           tmc_loc_id_str = str(tmc_loc_id) 
                           
                           # Only store data for IDs we actually requested in this chunk
                           if tmc_loc_id_str in chunk_ids:
                               all_flow_data[tmc_loc_id_str] = current_flow
                               processed_in_chunk += 1
                           # else: logging.debug(f"Ignoring segment {loc_id} not in requested chunk.")
                        # else: logging.debug("Segment data missing locationId.")
                logging.debug(f"Processed {processed_in_chunk} relevant segments from chunk {chunk_num}.")
                chunk_success = True
                break # Success

            except requests.exceptions.HTTPError as e:
                retries -= 1
                status_code = e.response.status_code
                if status_code == 429 and retries > 0:
                    wait_time = (3 - retries) * backoff_factor
                    logging.warning(f"Rate limit hit (429) on chunk {chunk_num}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                # Check specific error cause for 400 if possible
                elif status_code == 400:
                    error_detail = "Unknown 400 error"
                    try: error_detail = e.response.json().get('cause', error_detail)
                    except: pass
                    logging.error(f"HERE API Bad Request (400) on chunk {chunk_num}. Cause: '{error_detail}'. Check API parameters (bbox, IDs, key).")
                    break # Don't retry fundamental bad requests
                else:
                    logging.error(f"HERE API HTTP error ({status_code}) on chunk {chunk_num} after retries: {e.response.text[:200]}")
                    break
            except requests.exceptions.RequestException as e:
                retries -= 1
                logging.error(f"HERE API connection/timeout error on chunk {chunk_num}: {e}")
                if retries > 0: time.sleep(backoff_factor)
                else: break
            except json.JSONDecodeError as e:
                 logging.error(f"Error decoding HERE API response for chunk {chunk_num}: {e}. Response: {resp.text[:200]}")
                 break
            except Exception as e:
                logging.error(f"Unexpected error during HERE API call for chunk {chunk_num}: {e}", exc_info=True)
                break

        if not chunk_success: logging.error(f"Failed to fetch chunk {chunk_num} after multiple retries.")
        time.sleep(0.2) # Small delay

    logging.info(f"Finished fetching HERE data. Got flow data for {len(all_flow_data)} locations.")
    return all_flow_data


# --- Data Collection and Storage ---
def collect_and_store_data():
    """Fetches data based on mapping, stores in DB, cleans old data."""
    logging.info("--- Running single data collection and cleanup cycle ---")
    job_start_time = time.time()

    if not pems_to_here_map or not global_bounding_box: # Check both map and bbox
        logging.error("Mapping data or bounding box not loaded. Cannot collect data.")
        return

    # 1. Ensure DB connection
    engine = connect_db()
    if engine is None:
        logging.error("Database connection failed. Skipping cycle.")
        return

    # 2. Get HERE API Key
    here_api_key = os.environ.get("HERE_API_KEY")
    source = "environment variable"
    if not here_api_key:
        here_api_key = get_secret_from_aws(SECRET_NAME_HERE_API)
        source = "AWS Secrets Manager"

    if not here_api_key or not isinstance(here_api_key, str):
         logging.error("Failed to retrieve valid HERE API Key. Skipping cycle.")
         return
    logging.info(f"Using HERE API Key from {source}.")

    # 3. Fetch live flow data using mapping and global bounding box
    unique_here_ids = set(details['here_loc_id'] for details in pems_to_here_map.values())
    live_flow_data = {}
    if not unique_here_ids:
         logging.warning("No HERE Location IDs in mapping. Nothing to fetch.")
    else:
        logging.info(f"Requesting data for {len(unique_here_ids)} unique HERE IDs using bbox.")
        live_flow_data = fetch_here_flow_data(unique_here_ids, here_api_key, global_bounding_box)

    # 4. Prepare data for insertion
    records_to_insert = []
    now_utc = datetime.now(timezone.utc)
    current_timestamp = now_utc - timedelta(minutes=now_utc.minute % COLLECTION_INTERVAL_MINUTES,
                                           seconds=now_utc.second,
                                           microseconds=now_utc.microsecond)
    logging.info(f"Processing data for target timestamp: {current_timestamp.isoformat()}")

    processed_sensors_count = 0
    missing_speed_count = 0
    missing_segment_data_count = 0

    for pems_id, here_details in pems_to_here_map.items():
        here_loc_id = here_details['here_loc_id']
        here_q_dir = here_details['here_q_dir']
        segment_flow = live_flow_data.get(here_loc_id)
        speed_mph = None

        if segment_flow:
            speed_mps = segment_flow.get('speedUncapped') # Speed Uncapped in m/s
            
            if speed_mps is not None:
                try:
                    speed_float_mps = float(speed_mps)
                    if speed_float_mps >= 0:
                        speed_mph = speed_float_mps * METERS_PER_SECOND_TO_MPH
                        processed_sensors_count += 1
                    else:
                        # Log negative speed but store NULL
                        logging.warning(f"Negative uncapped speed ({speed_mps} m/s) for PeMS:{pems_id} (HERE:{here_loc_id}/{here_q_dir}). Storing NULL.")
                        missing_speed_count += 1
                except (ValueError, TypeError):
                     logging.debug(f"Non-numeric speedUncapped for PeMS:{pems_id} ({here_loc_id}/{here_q_dir}): {speed_mps}")
                     missing_speed_count += 1
            else: 
                missing_speed_count += 1 # 'speedUncapped' key missing from segment_flow
        else: 
            missing_segment_data_count += 1 # Segment data missing

        records_to_insert.append({
            'ts': current_timestamp,
            'sensor_id': pems_id,
            'actual_speed': speed_mph if speed_mph is not None and not math.isnan(speed_mph) else None
        })

    logging.info(f"Prepared {len(records_to_insert)} records.")
    logging.info(f"  Valid speed for {processed_sensors_count} entries.")
    logging.info(f"  Missing speed/dir for {missing_speed_count} entries.")
    logging.info(f"  Missing segment data for {missing_segment_data_count} entries.")


    # 5. Insert data into the database
    if records_to_insert:
        insert_start_time = time.time()
        try:
            insert_df = pd.DataFrame(records_to_insert)
            # Ensure None is used instead of NaN for DB insertion
            insert_df['actual_speed'] = insert_df['actual_speed'].astype(object).where(pd.notnull(insert_df['actual_speed']), None)

            with engine.connect() as connection:
                with connection.begin():
                     insert_df.to_sql(
                        name='traffic_data', con=connection, if_exists='append', index=False,
                        method='multi', chunksize=1000,
                        dtype={'ts': sqlalchemy.DateTime(timezone=True), 'actual_speed': sqlalchemy.REAL}
                     )
            insert_duration = time.time() - insert_start_time
            logging.info(f"Inserted {len(records_to_insert)} records in {insert_duration:.2f}s.")
        except sqlalchemy.exc.IntegrityError:
             logging.warning(f"Integrity error (likely duplicate PK for ts={current_timestamp}). Skipping batch.")
        except Exception as e:
            logging.error(f"Failed to insert data into database: {e}", exc_info=True)


    # 6. Delete old data (Janitor part)
    logging.info("Running data retention cleanup...")
    cleanup_start_time = time.time()
    try:
        cutoff_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=RETENTION_DAYS)
        with engine.connect() as connection:
             with connection.begin():
                 delete_statement = sqlalchemy.text("DELETE FROM traffic_data WHERE ts < :cutoff")
                 result = connection.execute(delete_statement, {"cutoff": cutoff_time})
                 cleanup_duration = time.time() - cleanup_start_time
                 logging.info(f"Deleted {result.rowcount} records older than {cutoff_time.isoformat()} in {cleanup_duration:.2f}s.")
    except Exception as e:
        logging.error(f"Failed to delete old data: {e}", exc_info=True)

    job_duration = time.time() - job_start_time
    logging.info(f"--- Collection and cleanup cycle finished in {job_duration:.2f} seconds ---")


# --- Main Execution (Single Run for Cron) ---
if __name__ == "__main__":
    logging.info("==========================================")
    logging.info("Starting HERE Traffic Collector")
    logging.info(f"PID: {os.getpid()}")
    logging.info("==========================================")

    # Load mapping
    if not load_mapping():
        logging.critical("Failed to load mapping. Exiting.")
        exit(1)

    # Perform one collection and cleanup cycle
    try:
        collect_and_store_data()
    except Exception as e:
        logging.critical(f"Unhandled error during collect_and_store_data: {e}", exc_info=True)
        exit(1) # Exit with error code

    logging.info("Collector script finished successfully.")
    exit(0) # Explicitly exit with success code