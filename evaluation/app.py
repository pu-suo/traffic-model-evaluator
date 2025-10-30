# evaluation_service/app.py
from flask import Flask, request, jsonify, render_template, abort
import logging
import os
from datetime import datetime, timezone, timedelta

# Import the core logic function
try:
    from .evaluation_logic import evaluate_forecast, connect_db
except ImportError as e:
    logging.critical(f"Could not import evaluation_logic: {e}. Ensure evaluation_logic.py is in the same directory.")
    # Define dummy functions if import fails to allow Flask to start
    def evaluate_forecast(*args, **kwargs):
        return {"error": "CRITICAL: Evaluation logic module failed to import. Check server logs."}
    def connect_db():
        logging.error("CRITICAL: connect_db function not loaded.")
        return None

# --- Flask App Setup ---
app = Flask(__name__)

# Configure logging for Flask app - use Gunicorn's logger if available
if __name__ != '__main__': # When run by Gunicorn
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
else: # When run directly (flask run)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')
    app.logger.setLevel(logging.INFO)

# Try initial DB connection check on startup (optional)
# Using app_context ensures Flask context is available if needed by connect_db
with app.app_context():
     engine = connect_db()
     if engine is None:
         app.logger.error("Initial DB connection failed on Flask app startup. Evaluation endpoint WILL fail.")
     else:
          app.logger.info("Flask app startup: DB connection check successful.")


# --- Routes ---
@app.route('/')
def index():
    """Serves the basic HTML form for submitting evaluation requests."""
    app.logger.info("Serving index.html")
    # You could pass default values to the template here if desired
    # e.g., default_start_time = ...
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def handle_evaluate():
    """
    API endpoint to handle evaluation requests.
    Expects JSON payload with parameters.
    Returns JSON results or error messages.
    """
    endpoint_start_time = time.time()
    app.logger.info("Received POST request at /evaluate")
    if not request.is_json:
        app.logger.warning("Request content type is not JSON")
        abort(415, description="Request must be JSON") # Use abort for standard HTTP errors

    data = request.get_json()
    app.logger.debug(f"Request JSON payload: {data}")

    # --- Parameter Validation ---
    required_params = ['prediction_start_time_iso', 'horizon_minutes', 'lookback_minutes', 'model_type']
    missing_params = [p for p in required_params if p not in data or data[p] is None]
    if missing_params:
        msg = f"Missing required parameters: {', '.join(missing_params)}"
        app.logger.warning(msg)
        return jsonify({"error": msg}), 400

    try:
        start_time_iso = data['prediction_start_time_iso']
        horizon = int(data['horizon_minutes'])
        lookback = int(data['lookback_minutes'])
        model = data['model_type']

        # Basic type/value range validation
        if not isinstance(start_time_iso, str) or not start_time_iso:
             raise ValueError("prediction_start_time_iso must be a non-empty string")
        # Validate time format loosely (detailed validation in logic)
        try:
             # Try parsing to catch obvious format errors early
             temp_dt = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
        except ValueError:
             raise ValueError("prediction_start_time_iso format invalid (Expected ISO 8601 UTC: YYYY-MM-DDTHH:MM:SSZ)")

        # Sensible limits for horizon and lookback
        if not (5 <= horizon <= 180 and horizon % 5 == 0): # e.g., 5 mins to 3 hours, multiple of 5
             raise ValueError("horizon_minutes must be between 5 and 180, and a multiple of 5")
        if not (15 <= lookback <= 10080): # e.g., 15 mins to 7 days
             raise ValueError("lookback_minutes must be between 15 and 10080")
        if not isinstance(model, str) or model.upper() not in ['HA', 'ARIMA']:
             raise ValueError("model_type must be 'HA' or 'ARIMA'")

    except (ValueError, TypeError, KeyError) as e:
        msg = f"Invalid parameter value or type: {e}"
        app.logger.warning(msg)
        return jsonify({"error": msg}), 400

    # --- Call Evaluation Logic ---
    app.logger.info(f"Calling evaluation logic for start={start_time_iso}, h={horizon}, l={lookback}, m={model}")
    try:
        results = evaluate_forecast(
            prediction_start_time_iso=start_time_iso,
            horizon_minutes=horizon,
            lookback_minutes=lookback,
            model_type=model.upper() # Pass uppercase model type
        )
        # Check if the logic function returned an error dictionary
        if isinstance(results, dict) and 'error' in results:
             # Log the specific error from the logic function
             app.logger.error(f"Evaluation logic failed: {results['error']}")
             # Determine appropriate status code based on error type
             status_code = 500 if "Database" in results.get("error","") or "critical" in results.get("error","").lower() else 400
             return jsonify(results), status_code

        endpoint_duration = time.time() - endpoint_start_time
        app.logger.info(f"Evaluation successful. Request duration: {endpoint_duration:.2f}s. Returning results.")
        return jsonify(results)

    except Exception as e:
        # Catch unexpected errors in the Flask route itself
        app.logger.exception("CRITICAL: Unexpected error occurred in /evaluate route:")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

# --- Main Execution Block (for `flask run`) ---
if __name__ == '__main__':
    # Use 0.0.0.0 to be accessible on network, port 80 standard HTTP
    # Debug=False is crucial for production/deployment testing
    # Use waitress or gunicorn instead of Flask's built-in server for anything beyond basic testing
    app.logger.info("Starting Flask development server...")
    app.run(host='0.0.0.0', port=80, debug=False)