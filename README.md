# EMG Gesture Recognition Web Application

A production-ready, hardware-agnostic EMG gesture recognition system with real-time inference and a premium medical-tech themed web interface.

## Features

### Core Functionality
- **Multi-Model Consensus**: Uses Random Forest, Histogram Gradient Boosting, and Logistic Regression with majority voting
- **Uncertainty Detection**: Returns "uncertain" when models disagree beyond threshold
- **Real-time Inference**: Target latency < 100ms with latency monitoring
- **Hardware Agnostic**: Pluggable data source architecture supports CSV, simulated, and real sensor input

### Signal Processing
- Mean imputation for missing values
- Z-score thresholding for artifact removal
- Min-Max normalization with persistent statistics
- MAV and RMS feature extraction (16 features total)

### Novel Enhancements
- **Session Calibration**: Capture rest baseline for amplitude drift correction
- **Gesture-to-Action Mapping**: Abstract gestures to configurable actions (ON, OFF, TOGGLE, IDLE)
- **Feature Importance**: Explainable AI with channel importance visualization

### Supported Gestures
| Gesture | Action | Description |
|---------|--------|-------------|
| ✊ Fist | ON | Activate |
| 🖐️ Open | OFF | Deactivate |
| 🤏 Pinch | TOGGLE | Switch state |
| ✋ Rest | IDLE | No action |

## Quick Start

### Prerequisites
- Python 3.9+ or Docker
- 8GB RAM recommended for training

### Local Development

1. **Clone and navigate to project**:
   ```bash
   cd emg_gesture_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open browser**: Navigate to `http://localhost:5000`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t emg-gesture-app .
docker run -p 5000:5000 emg-gesture-app
```

## Project Structure

```
emg_gesture_app/
├── app.py                 # Flask application entry point
├── config.py              # Centralized configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build configuration
├── docker-compose.yml     # Docker Compose for development
│
├── data/
│   └── emg_data.csv       # Training dataset (80,001 samples)
│
├── models/                # Trained models (auto-generated)
│   ├── random_forest.joblib
│   ├── hist_gradient_boosting.joblib
│   ├── logistic_regression.joblib
│   ├── normalization_stats.joblib
│   └── label_encoder.joblib
│
├── src/
│   ├── emg_sources/       # Hardware abstraction layer
│   │   ├── base_source.py     # Abstract interface
│   │   ├── csv_source.py      # CSV file input
│   │   ├── simulated_source.py # Simulation with realistic noise
│   │   └── hardware_source.py  # Real sensor template
│   │
│   ├── signal_processing/ # Signal processing pipeline
│   │   ├── preprocessing.py   # Imputation, artifact removal, normalization
│   │   └── feature_extraction.py # MAV, RMS features
│   │
│   ├── ml/                # Machine learning
│   │   ├── training.py        # Model training pipeline
│   │   ├── inference.py       # Multi-model consensus prediction
│   │   └── calibration.py     # Session-based calibration
│   │
│   ├── actions/           # Gesture-to-action mapping
│   │   └── action_mapper.py   # Action abstraction layer
│   │
│   └── monitoring/        # Performance monitoring
│       └── latency_tracker.py # Latency-aware inference
│
├── templates/
│   └── index.html         # Main UI template
│
└── static/
    ├── css/
    │   └── styles.css     # Medical-tech themed styles
    └── js/
        └── app.js         # Frontend logic
```

## API Endpoints

### Prediction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Single sample prediction |
| `/api/upload` | POST | Batch CSV processing |

### Streaming

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stream/start` | POST | Start simulated stream |
| `/api/stream/stop` | POST | Stop stream |
| `/api/stream/data` | GET | SSE data stream |
| `/api/stream/gesture` | POST | Set simulated gesture |

### Calibration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/calibrate/start` | POST | Begin calibration |
| `/api/calibrate/sample` | POST | Add calibration sample |
| `/api/calibrate/status` | GET | Get calibration progress |

### Metrics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics` | GET | Latency statistics |
| `/api/feature_importance` | GET | Feature importance |
| `/api/status` | GET | System status |

## Hardware Integration

To connect a real EMG sensor, implement the `HardwareSource` class:

```python
from src.emg_sources import HardwareSource

class MyoSensor(HardwareSource):
    def connect_sensor(self):
        # Open serial port or Bluetooth
        pass
    
    def get_sample(self):
        # Read and parse sensor data
        # Return numpy array of 8 channels
        pass
```

See `src/emg_sources/hardware_source.py` for detailed integration documentation.

## Configuration

Edit `config.py` to customize:

- **Signal Processing**: Artifact threshold, normalization bounds
- **Training**: Split ratio, hyperparameters
- **Inference**: Uncertainty threshold, target latency
- **Actions**: Gesture-to-action mapping

## Training

Models are automatically trained on first startup if not present. To manually retrain:

```bash
python -c "from src.ml.training import ModelTrainer; ModelTrainer().run_full_pipeline()"
```

## Performance

| Metric | Target | Typical |
|--------|--------|---------|
| End-to-end latency | < 100ms | 15-30ms |
| Model accuracy | > 90% | 95%+ |
| Inference rate | 20 Hz | 20 Hz |

## License

MIT License
