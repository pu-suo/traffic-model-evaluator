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

# --- Configuration ---
SECRET_NAME_DB_CREDS = "HereTrafficDbCredentials" # Must match name used in AWS setup
# Attempt to get region from standard environment variables, fallback if needed
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-2"))
DEFAULT_ARIMA_ORDER = (1, 1, 0) # Default (p,d,q) order for ARIMA
DATA_FREQUENCY = '5min' # Expected frequency of data in the database

# --- Logging Setup ---
# Basic configuration, Flask app might override this later
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
db_engine = None
secrets_cache = {}

# --- AWS Secrets Manager ---
def get_secret(secret_name, region_name=AWS_REGION):
    """Retrieves secret value from AWS Secrets Manager, with caching."""
    if secret_name in secrets_cache:
        logger.debug(f"Using cached secret for: {secret_name}")
        return secrets_cache[secret_name]

    logger.info(f"Attempting to retrieve secret: {secret_name} from region {region_name}")
    session = boto3.session.Session(region_name=region_name)
    client = session.client(service_name='secretsmanager')

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        logger.info(f"Successfully retrieved secret: {secret_name}")

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
            logger.warning(f"Secret '{secret_name}' retrieved but might be binary format.")
            return None

    except client.exceptions.ResourceNotFoundException:
        logger.error(f"Secret '{secret_name}' not found in AWS Secrets Manager region {region_name}.")
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve secret '{secret_name}' from AWS SM: {e}", exc_info=False)
        return None

# --- Database Connection ---
def connect_db():
    """Establishes or verifies the database connection using SQLAlchemy."""
    global db_engine
    if db_engine is not None:
        try: # Quick check
             with db_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                 connection.execute(sqlalchemy.text("SELECT 1"))
             return db_engine
        except sqlalchemy.exc.OperationalError:
             logger.warning("DB connection test failed, attempting reconnect...")
             db_engine = None
        except Exception as e:
             logger.error(f"Unexpected error testing DB connection: {e}", exc_info=False)
             db_engine = None

    if db_engine is None:
        logger.info("Attempting to establish new database connection...")
        # Try environment variables first (useful for local testing with .env)
        db_user_env = os.environ.get("DB_USER") # Use DB_USER/PASS/HOST/PORT/NAME
        db_pass_env = os.environ.get("DB_PASSWORD")
        db_host_env = os.environ.get("DB_HOST")
        db_port_env = os.environ.get("DB_PORT", 5432)
        db_name_env = os.environ.get("DB_NAME")
        DATABASE_URL = None

        if all([db_user_env, db_pass_env, db_host_env, db_name_env]):
            logger.info(f"Using DB credentials from environment variables (Host: {db_host_env})")
            DATABASE_URL = f"postgresql+psycopg2://{db_user_env}:{db_pass_env}@{db_host_env}:{db_port_env}/{db_name_env}"
        else:
            # Fallback to AWS Secrets Manager
            logger.info("DB environment variables not fully set, attempting AWS Secrets Manager...")
            db_creds = get_secret(SECRET_NAME_DB_CREDS)
            if not isinstance(db_creds, dict):
                logger.error("Failed to retrieve/parse DB credentials secret from AWS SM.")
                return None

            db_user = db_creds.get('username')
            db_pass = db_creds.get('password')
            db_host = db_creds.get('host')
            db_port = db_creds.get('port', 5432)
            db_name = db_creds.get('dbname')

            if not all([db_user, db_pass, db_host, db_name]):
                logger.error("DB credentials incomplete in AWS secret.")
                return None
            DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            logger.info(f"Using DB credentials from AWS Secrets Manager (Host: {db_host})")

        if DATABASE_URL:
            try:
                db_engine = sqlalchemy.create_engine(
                    DATABASE_URL, pool_pre_ping=True, echo=False,
                    connect_args={'connect_timeout': 10}
                )
                with db_engine.connect() as connection:
                    logger.info("Database connection successful.")
                return db_engine
            except Exception as e:
                logger.error(f"Failed to create database engine or connect: {e}", exc_info=True)
                db_engine = None
                return None
        else:
             logger.error("Could not determine database connection string.")
             return None

    return db_engine


# --- Forecasting Models ---
def predict_historical_average(series, horizon_steps):
    """Predicts using the mean of the input series, ignoring NaNs."""
    if series.empty or series.isnull().all():
        logger.debug("HA input series is empty or all NaN.")
        return np.full(horizon_steps, np.nan)
    # Calculate mean ignoring NaNs
    prediction = np.nanmean(series.values)
    if np.isnan(prediction):
        logger.debug("HA prediction resulted in NaN (input likely all NaN).")
        return np.full(horizon_steps, np.nan)
    return np.full(horizon_steps, prediction)

def predict_arima(series, horizon_steps, order=DEFAULT_ARIMA_ORDER):
    """Fits ARIMA and predicts. Includes basic error handling and data prep."""
    series_clean = series.dropna()
    min_obs_needed = max(10, order[0] + order[2] + 2) # Heuristic for minimum observations
    if len(series_clean) < min_obs_needed :
        logger.debug(f"Not enough non-NaN data points ({len(series_clean)} < {min_obs_needed}) to fit ARIMA{order}.")
        return np.full(horizon_steps, np.nan)

    try:
        # Ensure series index is datetime and has a frequency for statsmodels
        if not pd.api.types.is_datetime64_any_dtype(series_clean.index):
             series_clean.index = pd.to_datetime(series_clean.index, utc=True) # Ensure UTC

        # Attempt to set frequency if missing (common after pivoting)
        if series_clean.index.freq is None:
            series_clean = series_clean.asfreq(DATA_FREQUENCY) # Use configured frequency

        # Check for near-constant data
        if series_clean.nunique() < 2:
            logger.debug("Series data is constant/near-constant. Using HA instead.")
            return predict_historical_average(series_clean, horizon_steps)

        # Fit ARIMA model
        # Note: ARIMA requires tuning (order selection, stationarity checks) for good performance.
        # This uses a default order and basic fitting. Consider SARIMA for seasonality.
        # Use simple_differencing=False if d > 0 and you expect non-stationarity
        model = sm.tsa.ARIMA(series_clean, order=order, enforce_stationarity=False, enforce_invertibility=False)
        # Use suppress_warnings=True for common convergence issues, check logs if needed
        model_fit = model.fit()

        # Forecast future steps
        forecast = model_fit.forecast(steps=horizon_steps)

        # Ensure forecast is numpy array
        return forecast.values if isinstance(forecast, pd.Series) else forecast

    except (ValueError, np.linalg.LinAlgError) as e: # Catch common statsmodels errors
         logger.warning(f"Statsmodels ARIMA fitting error: {e}. Returning NaN.")
         return np.full(horizon_steps, np.nan)
    except Exception as e:
        logger.error(f"Unexpected error fitting/predicting ARIMA: {e}", exc_info=False) # Log less verbosely
        return np.full(horizon_steps, np.nan)

# --- Main Evaluation Function ---
def evaluate_forecast(prediction_start_time_iso, horizon_minutes, lookback_minutes, model_type):
    """Fetches data for ALL sensors, runs model, compares, returns results."""
    logger.info(f"Starting evaluation: start={prediction_start_time_iso}, horizon={horizon_minutes}, lookback={lookback_minutes}, model={model_type}")
    engine = connect_db()
    if engine is None:
        logger.error("Evaluation failed: Database connection error.")
        return {"error": "Database connection failed"}

    try:
        # --- Time Window Calculation ---
        try: # Parse start time, ensure UTC
             prediction_start_time_utc = datetime.fromisoformat(prediction_start_time_iso.replace('Z', '+00:00'))
             if prediction_start_time_utc.tzinfo is None:
                 prediction_start_time_utc = prediction_start_time_utc.replace(tzinfo=timezone.utc)
        except ValueError:
             return {"error": f"Invalid prediction_start_time format: {prediction_start_time_iso}. Use ISO 8601 UTC."}

        lookback_interval = timedelta(minutes=int(lookback_minutes))
        horizon_interval = timedelta(minutes=int(horizon_minutes))
        history_start_time_utc = prediction_start_time_utc - lookback_interval
        horizon_end_time_utc = prediction_start_time_utc + horizon_interval
        logger.info(f"Time window: History=[{history_start_time_utc}, {prediction_start_time_utc}), Horizon=[{prediction_start_time_utc}, {horizon_end_time_utc})")

        # --- Determine Sensors to Evaluate ---
        # Query distinct sensors that have *any* data in the lookback period
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

        # --- Fetch Data ---
        # Fetch both history and ground truth in one query for efficiency
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
                "pred_end": horizon_end_time_utc # Fetch up to end of horizon
            }, parse_dates=['ts'])
            # Ensure timezone is set correctly after read_sql
            all_data_df['ts'] = pd.to_datetime(all_data_df['ts'], utc=True)
            logger.info(f"Fetched {len(all_data_df)} total data points.")
        except Exception as e:
            logger.error(f"Failed to fetch traffic data from database: {e}", exc_info=True)
            return {"error": "Failed to fetch traffic data from database."}

        # Separate history and truth based on timestamp
        hist_df = all_data_df[all_data_df['ts'] < prediction_start_time_utc]
        truth_df = all_data_df[all_data_df['ts'] >= prediction_start_time_utc]

        if hist_df.empty:
            # Should not happen if sensor query worked, but safeguard
            logger.warning("History DataFrame is empty after filtering.")
            return {"error": "No historical data points found (unexpected error)."}

        # --- Data Preparation & Prediction Loop ---
        results = {}
        expected_steps = int(horizon_minutes / COLLECTION_INTERVAL_MINUTES) # Should be int if using 5min interval

        # Pivot history - crucial step
        # Create a full date range for the history period to ensure gaps are NaNs
        hist_index = pd.date_range(start=history_start_time_utc, end=prediction_start_time_utc, freq=DATA_FREQUENCY, closed='left', tz='UTC')
        hist_pivot = hist_df.pivot(index='ts', columns='sensor_id', values='actual_speed')
        # Reindex to ensure all time steps and sensors are present, filling missing data with NaN
        hist_pivot = hist_pivot.reindex(index=hist_index, columns=sensor_ids_to_evaluate)

        # Pivot truth similarly for easy lookup
        truth_index = pd.date_range(start=prediction_start_time_utc, end=horizon_end_time_utc, freq=DATA_FREQUENCY, closed='left', tz='UTC')
        truth_pivot = truth_df.pivot(index='ts', columns='sensor_id', values='actual_speed')
        truth_pivot = truth_pivot.reindex(index=truth_index, columns=sensor_ids_to_evaluate)

        logger.info(f"Generating predictions for {len(sensor_ids_to_evaluate)} sensors...")
        for sensor in sensor_ids_to_evaluate:
            sensor_history = hist_pivot[sensor] # This series includes NaNs for missing data

            # Generate predictions
            if model_type.upper() == 'HA':
                predictions = predict_historical_average(sensor_history, expected_steps)
            elif model_type.upper() == 'ARIMA':
                predictions = predict_arima(sensor_history, expected_steps)
            else: # Should be caught by validation, but safeguard
                logger.warning(f"Unknown model_type '{model_type}' for sensor {sensor}. Skipping.")
                predictions = np.full(expected_steps, np.nan)

            # Align predictions with expected timestamps in the horizon
            pred_timestamps = truth_index # Use the truth index for alignment
            # Handle cases where prediction array might be shorter if horizon isn't multiple of interval
            if len(predictions) < len(pred_timestamps):
                predictions = np.pad(predictions, (0, len(pred_timestamps) - len(predictions)), constant_values=np.nan)
            elif len(predictions) > len(pred_timestamps):
                predictions = predictions[:len(pred_timestamps)]

            pred_series = pd.Series(predictions, index=pred_timestamps, name='predicted_speed')

            # Get corresponding ground truth for this sensor from the pivoted truth table
            sensor_truth = truth_pivot[sensor]

            # --- Calculate Metrics ---
            # Combine prediction and truth, drop rows where EITHER is NaN for metric calculation
            comparison_df = pd.DataFrame({'predicted_speed': pred_series, 'actual_speed': sensor_truth})
            eval_comparison_df = comparison_df.dropna() # Drop rows missing pred OR truth

            mae = np.nan
            rmse = np.nan
            bias = np.nan
            count = 0
            if not eval_comparison_df.empty:
                try:
                    mae = mean_absolute_error(eval_comparison_df['actual_speed'], eval_comparison_df['predicted_speed'])
                    rmse = mean_squared_error(eval_comparison_df['actual_speed'], eval_comparison_df['predicted_speed'], squared=False)
                    bias = (eval_comparison_df['predicted_speed'] - eval_comparison_df['actual_speed']).mean()
                    count = len(eval_comparison_df)
                except Exception as metric_e:
                     logger.warning(f"Error calculating metrics for sensor {sensor}: {metric_e}")


            # --- Prepare result format (JSON compatible) ---
            # Convert NaN to None, format timestamps as ISO UTC strings
            predictions_list = [
                (ts.strftime('%Y-%m-%dT%H:%M:%SZ'), None if np.isnan(val) else round(float(val), 2))
                for ts, val in pred_series.items() # Use pred_series which has full horizon timestamps
            ]
            ground_truth_list = [
                 (ts.strftime('%Y-%m-%dT%H:%M:%SZ'), None if pd.isna(val) else round(float(val), 2))
                 for ts, val in sensor_truth.items() # Use sensor_truth which has full horizon timestamps
            ]

            results[sensor] = {
                'metrics': {
                    'mae': None if np.isnan(mae) else round(float(mae), 2),
                    'rmse': None if np.isnan(rmse) else round(float(rmse), 2),
                    'bias': None if np.isnan(bias) else round(float(bias), 2),
                    'count': count
                 },
                'predictions': predictions_list,
                'ground_truth': ground_truth_list
            }

        # --- Calculate Overall Metrics ---
        total_mae = 0
        total_sq_error = 0
        total_bias_sum = 0
        total_points = 0
        valid_sensors = 0
        for sensor, data in results.items():
            if data['metrics']['count'] > 0:
                valid_sensors += 1
                total_mae += data['metrics']['mae'] * data['metrics']['count']
                total_sq_error += (data['metrics']['rmse']**2) * data['metrics']['count']
                total_bias_sum += data['metrics']['bias'] * data['metrics']['count']
                total_points += data['metrics']['count']

        overall_metrics = {
             'mae': round(total_mae / total_points, 2) if total_points > 0 else None,
             'rmse': round(math.sqrt(total_sq_error / total_points), 2) if total_points > 0 else None,
             'bias': round(total_bias_sum / total_points, 2) if total_points > 0 else None,
             'total_points_compared': total_points,
             'sensors_evaluated': len(sensor_ids_to_evaluate),
             'sensors_with_results': valid_sensors
        }
        logger.info(f"Finished evaluation. Overall Metrics: {overall_metrics}")

        # Final return structure
        return {
            "overall_metrics": overall_metrics,
            "per_sensor_results": results
        }

    except Exception as e:
        logger.exception("An critical error occurred during the evaluation process:")
        return {"error": f"An unexpected critical error occurred: {str(e)}"}