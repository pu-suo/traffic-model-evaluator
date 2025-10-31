# evaluation_service/evaluation_logic.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sqlalchemy
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import boto3
import json
import os
import logging
import math
from zoneinfo import ZoneInfo
import folium # <-- NEW IMPORT

# --- Configuration ---
SECRET_NAME_DB_CREDS = "HereTrafficDbCredentials"
AWS_REGION = "us-east-2"
DEFAULT_ARIMA_ORDER = (1, 1, 0)
DATA_FREQUENCY = '5min'
COLLECTION_INTERVAL_MINUTES = 5
MAPPING_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'collector', 'pems_here_mapping.csv')


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
db_engine = None
secrets_cache = {}
sensor_locations = {}

# --- Load Sensor Locations ---
def load_sensor_locations(filepath=MAPPING_FILE_PATH):
    """
    Loads sensor lat/lon from the mapping file into the global cache.
    Called once on app startup.
    """
    global sensor_locations
    if not os.path.exists(filepath):
         # Try a fallback path if the first one fails
         fallback_path = os.path.join(os.path.dirname(__file__), '..', 'mapping', 'pems_here_mapping.csv')
         if os.path.exists(fallback_path):
             filepath = fallback_path
         else:
             raise FileNotFoundError(f"Cannot find pems_here_mapping.csv at {filepath} or {fallback_path}")

    logger.info(f"Loading sensor locations from {filepath}")
    mapping_df = pd.read_csv(filepath)
    mapping_df = mapping_df.dropna(subset=['sensor_id', 'pems_lat', 'pems_lon'])
    # Use 'sensor_id' if 'pems_vds' isn't present, or whatever your ID column is
    id_col = 'sensor_id'
    if id_col not in mapping_df.columns:
        raise ValueError(f"Column '{id_col}' not found in mapping file.")
        
    mapping_df[id_col] = mapping_df[id_col].astype(str)
    # Ensure no duplicates, taking the first lat/lon found
    mapping_df = mapping_df.drop_duplicates(subset=[id_col])
    
    temp_map = {}
    for _, row in mapping_df.iterrows():
        temp_map[row[id_col]] = (row['pems_lat'], row['pems_lon'])
    
    sensor_locations = temp_map
    logger.info(f"Cached {len(sensor_locations)} sensor locations.")


# --- Database ---
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

# --- AWS Secrets Manager ---
def get_secret_from_aws(secret_name, region_name=AWS_REGION):
    """Retrieves secret value ONLY from AWS Secrets Manager."""
    if secret_name in secrets_cache: return secrets_cache[secret_name]
    logging.info(f"Attempting to retrieve secret from AWS Secrets Manager: {secret_name} in {region_name}")
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        logging.info(f"Successfully retrieved secret from AWS SM: {secret_name}")
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            try:
                parsed_secret = json.loads(secret)
                secrets_cache[secret_name] = parsed_secret
                return parsed_secret
            except json.JSONDecodeError:
                secrets_cache[secret_name] = secret
                return secret
        else:
            logging.warning(f"Secret '{secret_name}' from AWS SM might be binary.")
            return None
    except client.exceptions.ResourceNotFoundException:
        logging.error(f"Secret '{secret_name}' not found in AWS Secrets Manager region {region_name}.")
        return None
    except Exception as e:
        logging.error(f"Failed to retrieve secret '{secret_name}' from AWS SM: {e}", exc_info=False)
        return None

# --- Forecasting Models ---
def predict_historical_average(series, horizon_steps):
    """Predicts using the mean of the input series, ignoring NaNs."""
    if series.empty or series.isnull().all():
        logger.debug("HA input series is empty or all NaN.")
        return np.full(horizon_steps, np.nan)
    prediction = np.nanmean(series.values)
    if np.isnan(prediction):
        logger.debug("HA prediction resulted in NaN (input likely all NaN).")
        return np.full(horizon_steps, np.nan)
    return np.full(horizon_steps, prediction)

def predict_arima(series, horizon_steps, order=DEFAULT_ARIMA_ORDER):
    """Fits ARIMA and predicts. Includes basic error handling and data prep."""
    series_clean = series.dropna()
    min_obs_needed = max(10, order[0] + order[2] + 2)
    if len(series_clean) < min_obs_needed :
        logger.debug(f"Not enough non-NaN data points ({len(series_clean)} < {min_obs_needed}) to fit ARIMA{order}.")
        return np.full(horizon_steps, np.nan)
    try:
        if not pd.api.types.is_datetime64_any_dtype(series_clean.index):
             series_clean.index = pd.to_datetime(series_clean.index, utc=True)
        if series_clean.index.freq is None:
            series_clean = series_clean.asfreq(DATA_FREQUENCY)
        if series_clean.nunique() < 2:
            logger.debug("Series data is constant/near-constant. Using HA instead.")
            return predict_historical_average(series_clean, horizon_steps)
        model = sm.tsa.ARIMA(series_clean, order=order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon_steps)
        return forecast.values if isinstance(forecast, pd.Series) else forecast
    except (ValueError, np.linalg.LinAlgError) as e:
         logger.warning(f"Statsmodels ARIMA fitting error: {e}. Returning NaN.")
         return np.full(horizon_steps, np.nan)
    except Exception as e:
        logger.error(f"Unexpected error fitting/predicting ARIMA: {e}", exc_info=False)
        return np.full(horizon_steps, np.nan)

# --- Map Generation Functions ---
def get_color_for_speed(speed):
    """Returns a color based on speed value."""
    if pd.isna(speed) or speed < 0:
        return '#808080' # Grey for missing data
    elif speed < 30:
        return '#FF0000' # Red
    elif speed < 55:
        return '#FFA500' # Orange
    else:
        return '#008000' # Green

def create_evaluation_map(df, value_column, map_title):
    """Creates a Folium map plotting sensors colored by speed."""
    if df.empty:
        logger.warning(f"Cannot create map '{map_title}', DataFrame is empty.")
        return folium.Map(location=[37.8, -122.2], zoom_start=9)._repr_html_()

    # Drop rows where the value to plot is missing
    plot_df = df.dropna(subset=[value_column, 'lat', 'lon'])
    if plot_df.empty:
        logger.warning(f"Cannot create map '{map_title}', no valid data points to plot.")
        return folium.Map(location=[37.8, -122.2], zoom_start=9)._repr_html_()

    map_center = [plot_df['lat'].mean(), plot_df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=10, tiles="cartodbdarkmatter")

    # --- REMOVED ---
    # The entire folium.plugins.FastMarkerCluster block has been removed.
    # --- END REMOVED ---


    # --- ADDED ---
    # Add individual CircleMarkers for each sensor
    for _, row in plot_df.iterrows():
        color = get_color_for_speed(row[value_column])
        value = row[value_column]
        
        # Create the circle marker
        marker = folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,  # Small, fixed-size circle
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        )
        
        # Add a Tooltip that appears on HOVER
        tooltip_html = f"<b>Sensor: {row['sensor_id']}</b><br>{map_title}: {value:.2f} mph"
        marker.add_child(folium.Tooltip(tooltip_html))
        
        # Add the marker to the map
        marker.add_to(m)
    # --- END ADDED ---


    # Add a simple title (Unchanged)
    title_html = f'''
                 <h3 align="center" style="font-size:16px; font-weight:bold; color: #FFFFFF; background-color: #333333; padding: 5px;">{map_title}</h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add a simple legend (Unchanged)
    legend_html = '''
    <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: 90px; 
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color: #333333; color: #FFFFFF;
         ">&nbsp; <b>Speed (mph)</b><br>
         &nbsp; <i class="fa fa-circle" style="color:#008000"></i>&nbsp; 55+<br>
         &nbsp; <i class="fa fa-circle" style="color:#FFA500"></i>&nbsp; 30 - 55<br>
         &nbsp; <i class="fa fa-circle" style="color:#FF0000"></i>&nbsp; &lt; 30
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Set the class for the CSS to target (Unchanged)
    m.get_root().get_name()
    m.get_root().html.add_child(folium.Element(f'<style>.{m.get_root().get_name()} {{ height: 100% !important; }}</style>'))
    
    # Return the map's HTML representation (Unchanged)
    iframe = m.get_root()._repr_html_()
    iframe = iframe.replace('<iframe', '<iframe class="folium-map"')
    return iframe


# --- Main Evaluation Function ---
def evaluate_forecast(prediction_start_time_iso, horizon_minutes, lookback_minutes, model_type):
    """
    Fetches data, runs model for a SINGLE future timestamp,
    compares, and returns metrics AND maps.
    """
    logger.info(f"Starting evaluation: start={prediction_start_time_iso}, horizon={horizon_minutes}, lookback={lookback_minutes}, model={model_type}")
    try:
        USER_TIMEZONE = ZoneInfo("America/Los_Angeles") 
        UTC_TIMEZONE = ZoneInfo("UTC") 
    except Exception:
        logger.critical("zoneinfo library failed. Is Python >= 3.9? Install 'tzdata' package if needed.")
        return {"error": "CRITICAL: Server timezone configuration error."}

    engine = connect_db()
    if engine is None:
        logger.error("Evaluation failed: Database connection error.")
        return {"error": "Database connection failed"}

    if not sensor_locations:
        logger.error("Sensor locations not loaded. Cannot generate maps.")
        return {"error": "CRITICAL: Sensor location mapping file not loaded on server."}

    try:
        # --- Time Window Calculation ---
        try: 
            naive_dt = datetime.fromisoformat(prediction_start_time_iso)
            local_dt = naive_dt.replace(tzinfo=USER_TIMEZONE)
            prediction_start_time_utc = local_dt.astimezone(UTC_TIMEZONE)
        except ValueError:
             return {"error": f"Invalid prediction_start_time format: {prediction_start_time_iso}. Use YYYY-MM-DDTHH:MM."}

        lookback_interval = timedelta(minutes=int(lookback_minutes))
        horizon_interval = timedelta(minutes=int(horizon_minutes))
        history_start_time_utc = prediction_start_time_utc - lookback_interval
        horizon_end_time_utc = prediction_start_time_utc + horizon_interval
        logger.info(f"Time window: History=[{history_start_time_utc}, {prediction_start_time_utc}), Horizon=[{prediction_start_time_utc}, {horizon_end_time_utc})")

        # --- Sensor/Data Fetching ---
        sensors_query = sqlalchemy.text("""
            SELECT DISTINCT sensor_id FROM traffic_data
            WHERE ts >= :start AND ts < :pred_start
        """)
        try:
            with engine.connect() as connection:
                 sensor_ids_result = connection.execute(sensors_query, {
                     "start": history_start_time_utc,
                     "pred_start": prediction_start_time_utc
                 }).fetchall()
            sensor_ids_to_evaluate = [row[0] for row in sensor_ids_result] if sensor_ids_result else []
        except Exception as e:
             logger.error(f"Failed to query sensor IDs from database: {e}", exc_info=True)
             return {"error": "Failed to query sensor IDs from database."}

        if not sensor_ids_to_evaluate:
            logger.warning("No sensors found with data in the lookback period.")
            return {"error": "No sensor data found in the specified lookback period."}
        logger.info(f"Found {len(sensor_ids_to_evaluate)} sensors with data in lookback period.")

        data_query = sqlalchemy.text("""
            SELECT ts, sensor_id, actual_speed
            FROM traffic_data
            WHERE sensor_id = ANY(:sids) AND ts >= :start AND ts < :pred_end
            ORDER BY sensor_id, ts
        """)
        try:
            all_data_df = pd.read_sql(data_query, engine, params={
                "sids": sensor_ids_to_evaluate,
                "start": history_start_time_utc,
                "pred_end": horizon_end_time_utc
            }, parse_dates=['ts'])
            all_data_df['ts'] = pd.to_datetime(all_data_df['ts'], utc=True)
            logger.info(f"Fetched {len(all_data_df)} total data points.")
        except Exception as e:
            logger.error(f"Failed to fetch traffic data from database: {e}", exc_info=True)
            return {"error": "Failed to fetch traffic data from database."}

        hist_df = all_data_df[all_data_df['ts'] < prediction_start_time_utc]
        truth_df = all_data_df[all_data_df['ts'] >= prediction_start_time_utc]

        if hist_df.empty:
            logger.warning("History DataFrame is empty after filtering.")
            return {"error": "No historical data points found (unexpected error)."}

        # --- Data Preparation & Prediction Loop ---
        results_list = [] # Store tuples of (sensor_id, pred, truth)
        expected_steps = int(horizon_minutes / COLLECTION_INTERVAL_MINUTES)
        
        # --- single timestamp predicting ---
        final_prediction_timestamp = prediction_start_time_utc + timedelta(minutes=(expected_steps - 1) * COLLECTION_INTERVAL_MINUTES)
        logger.info(f"Per user request, evaluating for single timestamp: {final_prediction_timestamp.isoformat()}")


        hist_index = pd.date_range(start=history_start_time_utc, end=prediction_start_time_utc, freq=DATA_FREQUENCY, inclusive='left', tz='UTC')
        hist_pivot = hist_df.pivot(index='ts', columns='sensor_id', values='actual_speed')
        hist_pivot = hist_pivot.reindex(index=hist_index, columns=sensor_ids_to_evaluate)

        truth_index = pd.date_range(start=prediction_start_time_utc, end=horizon_end_time_utc, freq=DATA_FREQUENCY, inclusive='left', tz='UTC')
        truth_pivot = truth_df.pivot(index='ts', columns='sensor_id', values='actual_speed')
        truth_pivot = truth_pivot.reindex(index=truth_index, columns=sensor_ids_to_evaluate)

        logger.info(f"Generating predictions for {len(sensor_ids_to_evaluate)} sensors...")
        for sensor in sensor_ids_to_evaluate:
            sensor_history = hist_pivot[sensor]
            
            final_pred_value = np.nan
            final_truth_value = np.nan

            if model_type.upper() == 'HA':
                predictions = predict_historical_average(sensor_history, expected_steps)
            elif model_type.upper() == 'ARIMA':
                predictions = predict_arima(sensor_history, expected_steps)
            else:
                predictions = np.full(expected_steps, np.nan)

            # --- Get ONLY the final prediction ---
            if predictions is not None and len(predictions) == expected_steps:
                final_pred_value = predictions[-1] # Get the last value

            # Get the corresponding ground truth for single timestamp
            if final_prediction_timestamp in truth_pivot.index:
                final_truth_value = truth_pivot.loc[final_prediction_timestamp, sensor]
            
            # Get sensor location
            lat, lon = sensor_locations.get(sensor, (None, None))

            results_list.append({
                "sensor_id": sensor,
                "lat": lat,
                "lon": lon,
                "final_pred": final_pred_value,
                "final_truth": final_truth_value
            })

        # --- Post-Loop Processing ---
        
        # Convert list of results to a DataFrame for easier metric calculation and mapping
        results_df = pd.DataFrame(results_list)
        results_df = results_df.dropna(subset=['final_pred', 'final_truth']) # Drop rows where we can't compare
        
        # Calculate bias for the valid results
        if not results_df.empty:
            results_df['bias'] = results_df['final_pred'] - results_df['final_truth']

        # --- Calculate Overall Metrics ---
        total_points = len(results_df)
        if total_points > 0:
            overall_mae = mean_absolute_error(results_df['final_truth'], results_df['final_pred'])
            overall_mse = mean_squared_error(results_df['final_truth'], results_df['final_pred'])
            overall_rmse = np.sqrt(overall_mse)
            overall_bias = results_df['bias'].mean()
            
            overall_metrics = {
                 'mae': round(overall_mae, 2),
                 'rmse': round(overall_rmse, 2),
                 'bias': round(overall_bias, 2),
                 'total_points_compared': total_points,
                 'sensors_evaluated': len(sensor_ids_to_evaluate),
                 'sensors_with_results': total_points 
            }
        else:
            overall_metrics = {
                 'mae': None, 'rmse': None, 'bias': None,
                 'total_points_compared': 0,
                 'sensors_evaluated': len(sensor_ids_to_evaluate),
                 'sensors_with_results': 0
            }
        
        logger.info(f"Finished evaluation. Overall Metrics: {overall_metrics}")

        # --- Generate Maps ---
        logger.info("Generating predicted speed map...")
        pred_map_html = create_evaluation_map(results_df, 'final_pred', f'Predicted Speeds ({model_type})')
        
        logger.info("Generating actual speed map...")
        truth_map_html = create_evaluation_map(results_df, 'final_truth', 'Actual Speeds (Ground Truth)')

        # --- Format Per-Sensor Results for Table ---
        per_sensor_results = {}
        for _, row in results_df.iterrows():
            bias = row['bias']
            per_sensor_results[row['sensor_id']] = {
                'prediction': round(row['final_pred'], 2),
                'actual': round(row['final_truth'], 2),
                'bias': round(bias, 2)
            }
        
        # Add sensors that had no valid comparison
        sensors_with_no_results = set(sensor_ids_to_evaluate) - set(results_df['sensor_id'])
        for sensor in sensors_with_no_results:
             per_sensor_results[sensor] = {
                 'prediction': None, 'actual': None, 'bias': None
             }

        # Final return structure
        return {
            "overall_metrics": overall_metrics,
            "per_sensor_results": per_sensor_results,
            "maps": {
                "predicted": pred_map_html,
                "actual": truth_map_html
            }
        }

    except Exception as e:
        logger.exception("An critical error occurred during the evaluation process:")
        return {"error": f"An unexpected critical error occurred: {str(e)}"}