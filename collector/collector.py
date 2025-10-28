import pandas as pd
import requests
import json
import os
import time
from datetime import datetime, timedelta, timezone
# Removed: import schedule
import sqlalchemy
import boto3
import logging
import sys
import math

# --- Configuration ---
# Secrets will be fetched primarily from environment, fallback to AWS Secrets Manager
SECRET_NAME_HERE_API = "HereTrafficApiKey"
SECRET_NAME_DB_CREDS = "HereTrafficDbCredentials"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-2"))

MAPPING_FILE_PATH = 'pems_here_mapping.csv'
HERE_FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"
REQUEST_TIMEOUT = 30
MAX_IDS_PER_REQUEST = 500
COLLECTION_INTERVAL_MINUTES = 5 # Used for timestamp flooring
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
db_engine = None # Engine will be created per run now
secrets_cache = {} # Cache still useful within a single run if get_secret is called multiple times

# --- AWS Secrets Manager ---
def get_secret_from_aws(secret_name, region_name=AWS_REGION):
    """Retrieves secret value ONLY from AWS Secrets Manager."""
    # Check cache first
    # if secret_name in secrets_cache:
    #     logging.debug(f"Using cached secret for: {secret_name}")
    #     return secrets_cache[secret_name] # Disable cache for single run? Or keep? Let's keep.
    if secret_name in secrets_cache: return secrets_cache[secret_name]


    logging.info(f"Attempting to retrieve secret from AWS Secrets Manager: {secret_name} in {region_name}")
    session = boto3.session.Session(region_name=region_name)
    client = session.client(service_name='secretsmanager')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        logging.info(f"Successfully retrieved secret from AWS SM: {secret_name}")

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
        logging.error(f"Secret '{secret_name}' not found in AWS Secrets Manager region {region_name}.")
        return None
    except Exception as e:
        # Catch potential Boto3/IAM errors if run locally without credentials/role
        logging.error(f"Failed to retrieve secret '{secret_name}' from AWS SM: {e}", exc_info=False) # Don't need full traceback usually
        return None

# --- Mapping Loader ---
def load_mapping(filepath=MAPPING_FILE_PATH):
    """Loads the PeMS-HERE mapping CSV into the global dictionary. Returns True on success."""
    global pems_to_here_map
    try:
        # (Same loading logic as before - reading CSV, validation, populating dict)
        logging.info(f"Loading PeMS-HERE mapping from '{filepath}'...")
        if not os.path.isabs(filepath):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             filepath = os.path.join(script_dir, filepath)
        if not os.path.exists(filepath): raise FileNotFoundError(f"Mapping file not found at '{filepath}'.")

        mapping_df = pd.read_csv(filepath)
        required_cols = ['sensor_id', 'here_locationId', 'here_queuingDirection']
        # Handle case variations robustly
        mapping_df.columns = map(str.lower, mapping_df.columns)
        required_cols_lower = {c.lower(): c for c in required_cols} # Map lower to original required
        rename_map = {}
        found_cols = []
        for lower_req, orig_req in required_cols_lower.items():
            if lower_req in mapping_df.columns:
                 rename_map[lower_req] = orig_req # Rename to standard expected name
                 found_cols.append(orig_req)
            else:
                 raise ValueError(f"Mapping file must contain column: {orig_req} (or case variation)")
        mapping_df = mapping_df.rename(columns=rename_map)


        mapping_df = mapping_df.dropna(subset=required_cols)
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
        return True # Indicate success

    except Exception as e:
        logging.error(f"CRITICAL: Failed to load or parse mapping file '{filepath}': {e}", exc_info=True)
        pems_to_here_map = {}
        return False # Indicate failure

# --- Database Connection ---
def connect_db():
    """Establishes DB connection using ENV vars first, then AWS Secrets."""
    global db_engine
    if db_engine is not None: # Avoid reconnecting if already connected in this run
        return db_engine

    logging.info("Attempting to establish database connection...")
    # --- Try Environment Variables First (for local Docker Compose testing) ---
    db_user_env = os.environ.get("POSTGRES_USER")
    db_pass_env = os.environ.get("POSTGRES_PASSWORD")
    db_host_env = os.environ.get("DB_HOST", "database") # Default to service name for Docker
    db_port_env = os.environ.get("DB_PORT", 5432)
    db_name_env = os.environ.get("POSTGRES_DB")

    using_env_vars = False
    if all([db_user_env, db_pass_env, db_host_env, db_name_env]):
        logging.info(f"Using database credentials from environment variables (Host: {db_host_env})")
        DATABASE_URL = f"postgresql+psycopg2://{db_user_env}:{db_pass_env}@{db_host_env}:{db_port_env}/{db_name_env}"
        using_env_vars = True
    else:
        # --- Fallback to AWS Secrets Manager ---
        logging.info("Database environment variables not fully set, attempting AWS Secrets Manager...")
        db_creds = get_secret_from_aws(SECRET_NAME_DB_CREDS)
        if not isinstance(db_creds, dict):
            logging.error("Failed to retrieve/parse DB credentials secret from AWS SM.")
            return None

        db_user = db_creds.get('username')
        db_pass = db_creds.get('password')
        db_host = db_creds.get('host')
        db_port = db_creds.get('port', 5432)
        db_name = db_creds.get('dbname')

        if not all([db_user, db_pass, db_host, db_name]):
            logging.error("Database credentials incomplete in AWS secret.")
            return None
        DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        logging.info(f"Using database credentials from AWS Secrets Manager (Host: {db_host})")

    # --- Create Engine ---
    try:
        db_engine = sqlalchemy.create_engine(
            DATABASE_URL,
            pool_pre_ping=True, # Helps with connections that might go stale
            echo=False,
            connect_args={'connect_timeout': 10}
        )
        # Test connection immediately
        with db_engine.connect() as connection:
            logging.info("Database connection successful.")
        return db_engine
    except Exception as e:
        logging.error(f"Failed to create database engine or connect: {e}", exc_info=True)
        db_engine = None # Reset on failure
        return None

# --- HERE API Fetcher ---
# (fetch_here_flow_data function remains the same as previous version - handles chunking, retries)
def fetch_here_flow_data(location_ids, api_key):
    """Fetches current flow data from HERE API for a list of location IDs."""
    if not location_ids:
        logging.warning("fetch_here_flow_data: No location IDs provided.")
        return {}
    if not api_key:
        logging.error("fetch_here_flow_data: HERE API Key is missing.")
        return {}

    all_flow_data = {} # Maps locationId -> currentFlow object
    location_id_list = sorted(list(location_ids))

    logging.info(f"Fetching HERE flow data for {len(location_id_list)} unique IDs in chunks of up to {MAX_IDS_PER_REQUEST}...")
    num_chunks = math.ceil(len(location_id_list) / MAX_IDS_PER_REQUEST)

    for i in range(0, len(location_id_list), MAX_IDS_PER_REQUEST):
        chunk_ids = location_id_list[i:i + MAX_IDS_PER_REQUEST]
        params = {
            "locationIds": ",".join(chunk_ids),
            "locationReferencing": "tmc",
            "apiKey": api_key
        }
        retries = 3
        backoff_factor = 5 # seconds
        chunk_num = i // MAX_IDS_PER_REQUEST + 1
        chunk_success = False

        while retries > 0:
            try:
                logging.debug(f"Fetching HERE chunk {chunk_num}/{num_chunks} ({len(chunk_ids)} IDs)...")
                resp = requests.get(HERE_FLOW_URL, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])

                processed_in_chunk = 0
                for segment_data in results:
                    loc = segment_data.get("location")
                    current_flow = segment_data.get("currentFlow") # This contains '+', '-' keys
                    if loc and current_flow:
                        loc_id = loc.get("locationId")
                        if loc_id:
                           all_flow_data[loc_id] = current_flow
                           processed_in_chunk += 1
                        # else: logging.debug("Segment data missing locationId.") # Too verbose
                    # else: logging.debug("Segment data missing location or currentFlow.")
                logging.debug(f"Processed {processed_in_chunk} segments from chunk {chunk_num}.")
                chunk_success = True
                break # Success for this chunk

            except requests.exceptions.HTTPError as e:
                retries -= 1
                status_code = e.response.status_code
                if status_code == 429 and retries > 0:
                    wait_time = (3 - retries) * backoff_factor
                    logging.warning(f"Rate limit hit (429) on chunk {chunk_num}. Waiting {wait_time}s before retry {3-retries}...")
                    time.sleep(wait_time)
                elif status_code == 400:
                    logging.error(f"HERE API Bad Request (400) for chunk {chunk_num}. Check IDs? Error: {e.response.text[:200]}")
                    break
                else:
                    logging.error(f"HERE API call HTTP error ({status_code}) on chunk {chunk_num} after retries: {e}")
                    logging.error(f"Response: {e.response.text[:200]}")
                    break
            except requests.exceptions.RequestException as e:
                retries -= 1
                logging.error(f"HERE API call connection/timeout error on chunk {chunk_num}: {e}")
                if retries > 0:
                    logging.warning(f"Retrying in {backoff_factor} seconds...")
                    time.sleep(backoff_factor)
                else: break
            except json.JSONDecodeError as e:
                 logging.error(f"Error decoding HERE API response for chunk {chunk_num}: {e}")
                 logging.error(f"Response: {resp.text[:200]}")
                 break
            except Exception as e:
                logging.error(f"Unexpected error during HERE API call for chunk {chunk_num}: {e}", exc_info=True)
                break

        if not chunk_success: logging.error(f"Failed to fetch chunk {chunk_num} after multiple retries.")
        time.sleep(0.2) # Small delay between chunks

    logging.info(f"Finished fetching HERE data. Got flow data for {len(all_flow_data)} locations.")
    return all_flow_data


# --- Data Collection and Storage ---
def collect_and_store_data():
    """Fetches data based on mapping, stores in DB, cleans old data."""
    logging.info("--- Running single data collection and cleanup cycle ---")
    job_start_time = time.time()

    if not pems_to_here_map:
        logging.error("Mapping data is not loaded. Cannot collect data.")
        return # Exit the function for this run

    # 1. Ensure DB connection is active
    engine = connect_db()
    if engine is None:
        logging.error("Database connection failed. Skipping cycle.")
        return

    # 2. Get HERE API Key (from ENV first, then AWS SM)
    here_api_key = os.environ.get("HERE_API_KEY")
    source = "environment variable"
    if not here_api_key:
        here_api_key = get_secret_from_aws(SECRET_NAME_HERE_API)
        source = "AWS Secrets Manager"

    if not here_api_key or not isinstance(here_api_key, str):
         logging.error("Failed to retrieve valid HERE API Key from environment or AWS SM. Skipping cycle.")
         return
    logging.info(f"Using HERE API Key from {source}.")


    # 3. Fetch live flow data based on mapping
    unique_here_ids = set(details['here_loc_id'] for details in pems_to_here_map.values())
    live_flow_data = {}
    if not unique_here_ids:
         logging.warning("No HERE Location IDs found in the mapping. Nothing to fetch.")
    else:
        logging.info(f"Requesting data for {len(unique_here_ids)} unique HERE location IDs from mapping.")
        live_flow_data = fetch_here_flow_data(unique_here_ids, here_api_key)

    # 4. Prepare data for insertion
    records_to_insert = []
    # Use UTC, floor to the nearest COLLECTION_INTERVAL_MINUTES interval
    now_utc = datetime.now(timezone.utc)
    minutes_remainder = now_utc.minute % COLLECTION_INTERVAL_MINUTES
    current_timestamp = now_utc - timedelta(minutes=minutes_remainder,
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
            direction_flow = segment_flow.get(here_q_dir)
            if direction_flow:
                speed_mps = direction_flow.get('SU') # Speed Uncapped in m/s
                if speed_mps is not None:
                    try:
                        speed_float_mps = float(speed_mps)
                        if speed_float_mps >= 0:
                            speed_mph = speed_float_mps * METERS_PER_SECOND_TO_MPH
                            processed_sensors_count += 1
                        else:
                            logging.warning(f"Negative speed ({speed_mps} m/s) for {pems_id}. Storing NULL.")
                            missing_speed_count += 1
                    except (ValueError, TypeError):
                        missing_speed_count += 1
                else: missing_speed_count += 1
            else: missing_speed_count += 1
        else: missing_segment_data_count += 1

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
             logging.warning(f"Integrity error (likely duplicate primary key for ts={current_timestamp}). Skipping batch.")
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


# --- Main Execution (Single Run) ---
if __name__ == "__main__":
    logging.info("==========================================")
    logging.info("Starting HERE Traffic Collector (Single Run)")
    logging.info(f"PID: {os.getpid()}")
    logging.info("==========================================")

    # Load mapping - essential, exit if fails
    if not load_mapping():
        logging.critical("Failed to load mapping. Exiting.")
        exit(1)

    # Perform one collection and cleanup cycle
    try:
        collect_and_store_data()
    except Exception as e:
        # Catch any unexpected errors during the main function call
        logging.critical(f"Unhandled error during collect_and_store_data: {e}", exc_info=True)
        exit(1) # Exit with error code if the main function fails unexpectedly

    logging.info("Collector script finished.")
    exit(0) # Explicitly exit with success code

