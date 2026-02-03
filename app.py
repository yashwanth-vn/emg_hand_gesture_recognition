"""
EMG Gesture Recognition - Flask Backend

This is the main Flask application providing REST API endpoints for:
- CSV file upload and batch inference
- Simulated live EMG streaming (SSE)
- Single-sample prediction
- Session calibration
- Metrics and feature importance

All endpoints return JSON responses.
Target inference latency: < 100ms
"""
import os
import sys
import json
import time
import threading
from flask import Flask, request, jsonify, render_template, Response

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    MAX_UPLOAD_SIZE_MB, MODELS_DIR, STREAM_INTERVAL_MS
)
from src.emg_sources import CSVSource, SimulatedSource
from src.ml.training import train_models_if_needed
from src.ml.inference import get_predictor
from src.ml.calibration import SessionCalibrator
from src.actions import ActionMapper
from src.monitoring import get_latency_tracker

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching

# Global state
simulated_source = SimulatedSource()
session_calibrator = SessionCalibrator()
action_mapper = ActionMapper()
latency_tracker = get_latency_tracker()

# Streaming state
_stream_active = False
_stream_lock = threading.Lock()


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_app():
    """
    Initialize the application on startup.
    
    This function:
    1. Ensures models directory exists
    2. Trains models if not present
    3. Loads models into memory
    """
    global simulated_source
    
    print("\n" + "="*60)
    print("EMG GESTURE RECOGNITION SYSTEM - INITIALIZATION")
    print("="*60 + "\n")
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Train models if needed
    train_models_if_needed()
    
    # Pre-load models
    predictor = get_predictor()
    if predictor.ensure_loaded():
        print("[OK] Models loaded and ready for inference")
    else:
        print("[X] Warning: Could not load models")
    
    print("\n" + "="*60)
    print("SYSTEM READY")
    print("="*60 + "\n")


# =============================================================================
# ROUTES - FRONTEND
# =============================================================================

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


# =============================================================================
# ROUTES - CSV UPLOAD AND BATCH INFERENCE
# =============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """
    Upload a CSV file and perform batch inference.
    
    Expected: CSV file with 8 EMG channels, no labels.
    Returns: Array of predictions for each row.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    try:
        # Read file content
        content = file.read()
        
        # Load into CSV source
        csv_source = CSVSource()
        if not csv_source.load_from_file(content):
            return jsonify({'error': 'Failed to parse CSV file'}), 400
        
        # Get all data
        data = csv_source.get_all_data()
        if data is None or len(data) == 0:
            return jsonify({'error': 'No data in CSV file'}), 400
        
        # Get predictor
        predictor = get_predictor()
        
        # Batch prediction
        results = predictor.predict_batch(data)
        
        # Summary statistics
        gesture_counts = {}
        for r in results:
            g = r['gesture']
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        return jsonify({
            'success': True,
            'sample_count': len(results),
            'predictions': results,
            'summary': {
                'gesture_counts': gesture_counts,
                'latency_stats': latency_tracker.get_current_stats()
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES - SINGLE PREDICTION
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a prediction from raw EMG data.
    
    Expected JSON: {"emg_data": [ch1, ch2, ..., ch8]}
    Returns: Prediction with gesture, confidence, action, etc.
    """
    try:
        data = request.get_json()
        if not data or 'emg_data' not in data:
            return jsonify({'error': 'emg_data required'}), 400
        
        emg_data = data['emg_data']
        if len(emg_data) != 8:
            return jsonify({'error': 'Expected 8 EMG channels'}), 400
        
        import numpy as np
        emg_array = np.array(emg_data, dtype=np.float64)
        
        # Get predictor and make prediction
        predictor = get_predictor()
        
        # Set calibrator if available
        if session_calibrator.is_calibrated:
            predictor.set_calibrator(session_calibrator)
        
        result = predictor.predict(emg_array)
        
        # Add action mapping for UI Bulb
        actions = {'fist': 'ON', 'open': 'OFF', 'pinch': 'TOGGLE', 'rest': 'IDLE'}
        result['action'] = actions.get(result.get('gesture', 'rest').lower(), 'IDLE')
        
        # Record latency
        if 'latency' in result:
            latency_tracker.end_prediction(
                result['latency']['preprocessing_ms'],
                result['latency']['feature_extraction_ms'],
                result['latency']['inference_ms']
            )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES - RANDOM DETECTION (Generate random EMG and detect gesture)
# =============================================================================

@app.route('/api/detect_random', methods=['POST'])
def detect_random():
    """
    Generate random EMG data and detect the gesture.
    
    This endpoint generates completely random EMG patterns (not tied to any
    specific gesture) and runs them through the prediction pipeline.
    This demonstrates the model's ability to classify unknown signals.
    
    Returns: Prediction with gesture, confidence, action, and the generated EMG data.
    """
    try:
        import numpy as np
        
        # Generate truly random EMG-like data
        # Random values between 0 and 1, simulating normalized EMG signals
        emg_data = np.random.uniform(0.0, 1.0, 8).astype(np.float64)
        
        # Add some structure to make it more EMG-like
        # Occasionally generate patterns that look more like specific gestures
        pattern = np.random.choice(['random', 'low', 'high', 'mixed'])
        
        if pattern == 'low':
            # Low amplitude - likely rest
            emg_data = np.random.uniform(0.0, 0.15, 8).astype(np.float64)
        elif pattern == 'high':
            # High amplitude - likely fist or open
            emg_data = np.random.uniform(0.4, 0.9, 8).astype(np.float64)
        elif pattern == 'mixed':
            # Mixed - some channels high, some low
            for i in range(8):
                if np.random.random() > 0.5:
                    emg_data[i] = np.random.uniform(0.5, 0.9)
                else:
                    emg_data[i] = np.random.uniform(0.0, 0.2)
        # else: keep fully random
        
        # Get predictor and make prediction
        predictor = get_predictor()
        
        # Set calibrator if available
        if session_calibrator.is_calibrated:
            predictor.set_calibrator(session_calibrator)
        
        result = predictor.predict(emg_data)
        
        # Add action mapping for UI Bulb
        actions = {'fist': 'ON', 'open': 'OFF', 'pinch': 'TOGGLE', 'rest': 'IDLE'}
        result['action'] = actions.get(result.get('gesture', 'rest').lower(), 'IDLE')
        
        # Add the generated EMG data to the result
        result['emg_data'] = emg_data.tolist()
        result['pattern_type'] = pattern
        
        # Record latency
        if 'latency' in result:
            latency_tracker.end_prediction(
                result['latency']['preprocessing_ms'],
                result['latency']['feature_extraction_ms'],
                result['latency']['inference_ms']
            )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES - SIMULATED EMG STREAM
# =============================================================================

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start the simulated EMG stream."""
    global _stream_active
    
    with _stream_lock:
        if _stream_active:
            return jsonify({'message': 'Stream already active'}), 200
        
        simulated_source.start_stream()
        _stream_active = True
    
    return jsonify({
        'success': True,
        'message': 'Stream started',
        'interval_ms': STREAM_INTERVAL_MS
    })


@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop the simulated EMG stream."""
    global _stream_active
    
    with _stream_lock:
        simulated_source.stop_stream()
        _stream_active = False
    
    return jsonify({
        'success': True,
        'message': 'Stream stopped'
    })


@app.route('/api/stream/gesture', methods=['POST'])
def set_stream_gesture():
    """
    Set the gesture to simulate.
    
    Expected JSON: {"gesture": "fist|open|pinch|rest"}
    """
    try:
        data = request.get_json()
        gesture = data.get('gesture', 'rest')
        
        if simulated_source.set_gesture(gesture):
            return jsonify({
                'success': True,
                'gesture': gesture
            })
        else:
            return jsonify({'error': 'Invalid gesture'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream/data')
def stream_data():
    """
    Server-Sent Events endpoint for live EMG data and predictions.
    
    Returns a stream of JSON objects with EMG data and predictions.
    """
    def generate():
        predictor = get_predictor()
        
        # Set calibrator if available
        if session_calibrator.is_calibrated:
            predictor.set_calibrator(session_calibrator)
        
        while _stream_active:
            try:
                # Get sample from simulated source
                sample = simulated_source.get_sample()
                
                if sample is not None:
                    # Make prediction
                    result = predictor.predict(sample)
                    
                    # Add EMG data to result
                    result['emg_data'] = sample.tolist()
                    result['current_gesture_simulated'] = simulated_source.current_gesture
                    
                    # Send as SSE
                    yield f"data: {json.dumps(result)}\n\n"
                
                # Wait for next interval
                time.sleep(STREAM_INTERVAL_MS / 1000.0)
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
        
        yield f"data: {json.dumps({'status': 'stream_ended'})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# =============================================================================
# ROUTES - CALIBRATION
# =============================================================================

@app.route('/api/calibrate/start', methods=['POST'])
def start_calibration():
    """Start a new calibration session."""
    session_calibrator.reset()
    return jsonify({
        'success': True,
        'message': 'Calibration started. Send rest samples to /api/calibrate/sample',
        'samples_needed': session_calibrator._target_samples
    })


@app.route('/api/calibrate/sample', methods=['POST'])
def add_calibration_sample():
    """
    Add a sample to the calibration buffer.
    
    Expected JSON: {"emg_data": [ch1, ch2, ..., ch8]}
    """
    try:
        data = request.get_json()
        emg_data = data.get('emg_data')
        
        if not emg_data or len(emg_data) != 8:
            return jsonify({'error': 'Expected 8 EMG channels'}), 400
        
        import numpy as np
        sample = np.array(emg_data, dtype=np.float64)
        
        is_complete = session_calibrator.add_calibration_sample(sample)
        progress = session_calibrator.get_calibration_progress()
        
        return jsonify({
            'success': True,
            'is_complete': is_complete,
            'progress': progress
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibrate/status', methods=['GET'])
def calibration_status():
    """Get current calibration status."""
    return jsonify(session_calibrator.get_calibration_progress())


@app.route('/api/calibrate/baseline', methods=['GET'])
def calibration_baseline():
    """Get the computed baseline if calibrated."""
    baseline = session_calibrator.get_baseline_info()
    if baseline:
        return jsonify(baseline)
    else:
        return jsonify({'error': 'Not calibrated'}), 400


# =============================================================================
# ROUTES - METRICS AND FEATURE IMPORTANCE
# =============================================================================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get latency and performance metrics."""
    return jsonify({
        'latency': latency_tracker.get_current_stats(),
        'breakdown': latency_tracker.get_breakdown_stats(),
        'latest': latency_tracker.get_latest()
    })


@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from Random Forest model."""
    predictor = get_predictor()
    
    return jsonify({
        'feature_importance': predictor.get_feature_importance(),
        'channel_importance': predictor.get_channel_importance()
    })


@app.route('/api/action_mapping', methods=['GET'])
def get_action_mapping():
    """Get the current gesture-to-action mapping."""
    return jsonify(action_mapper.get_mapping())


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get overall system status."""
    predictor = get_predictor()
    
    return jsonify({
        'models_loaded': predictor._is_loaded,
        'stream_active': _stream_active,
        'calibrated': session_calibrator.is_calibrated,
        'latency_target_ms': 100,
        'within_latency_target': latency_tracker.is_within_target()
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Initialize on startup
    initialize_app()
    
    # Run Flask server
    print(f"\n[*] Starting server at http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(
        host=FLASK_HOST,
        port=5001,
        debug=FLASK_DEBUG,
        threaded=True
    )
