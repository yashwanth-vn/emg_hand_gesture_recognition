"""
Multi-Model Inference with Consensus Voting

This module handles gesture prediction using all three trained models,
implementing a majority voting system with uncertainty detection.

Key features:
- Load and use all three models
- Majority voting for final prediction
- Confidence variance calculation
- Uncertainty detection when models disagree
"""
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import Counter
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    RANDOM_FOREST_MODEL_PATH,
    HISTGB_MODEL_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
    NORMALIZATION_STATS_PATH,
    LABEL_ENCODER_PATH,
    UNCERTAINTY_VARIANCE_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    GESTURE_ACTION_MAP
)
from src.signal_processing.preprocessing import EMGPreprocessor
from src.signal_processing.feature_extraction import FeatureExtractor


class GesturePredictor:
    """
    Multi-model gesture predictor with consensus voting.
    
    This class uses all three trained models to make predictions:
    1. Each model votes on the gesture
    2. Majority vote determines the final prediction
    3. Confidence variance across models indicates uncertainty
    4. If disagreement is too high, returns "uncertain"
    
    Attributes:
        models: Dictionary of loaded model objects
        preprocessor: Fitted EMGPreprocessor
        feature_extractor: FeatureExtractor instance
        label_encoder: Fitted LabelEncoder
        calibrator: Optional SessionCalibrator for drift correction
    """
    
    def __init__(self):
        """Initialize the predictor (models loaded on first use)."""
        self.models: Dict[str, Any] = {}
        self.preprocessor: Optional[EMGPreprocessor] = None
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = None
        self.calibrator = None
        
        self._is_loaded = False
        
        # Latency tracking
        self._last_preprocess_time = 0.0
        self._last_feature_time = 0.0
        self._last_inference_time = 0.0
    
    def load_models(self) -> bool:
        """
        Load all trained models and preprocessing artifacts.
        
        Returns:
            True if all models loaded successfully, False otherwise
        """
        try:
            # Load all three models
            self.models['random_forest'] = joblib.load(RANDOM_FOREST_MODEL_PATH)
            self.models['hist_gradient_boosting'] = joblib.load(HISTGB_MODEL_PATH)
            self.models['logistic_regression'] = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)
            
            # Load preprocessing artifacts
            self.preprocessor = EMGPreprocessor()
            self.preprocessor.load(NORMALIZATION_STATS_PATH)
            
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            
            self._is_loaded = True
            print("All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def ensure_loaded(self) -> bool:
        """
        Ensure models are loaded before prediction.
        
        Returns:
            True if models are ready, False otherwise
        """
        if not self._is_loaded:
            return self.load_models()
        return True
    
    def predict(self, emg_data: np.ndarray, apply_calibration: bool = True) -> Dict[str, Any]:
        """
        Make a gesture prediction from raw EMG data.
        
        Args:
            emg_data: Raw EMG of shape (num_channels,) or (n_samples, num_channels)
            apply_calibration: Whether to apply session calibration if available
            
        Returns:
            Dictionary containing:
            - gesture: Predicted gesture name or "uncertain"
            - confidence: Confidence score (0-1) or None if uncertain
            - action: Mapped action (ON, OFF, TOGGLE, IDLE, HOLD)
            - model_votes: Individual model predictions
            - variance: Confidence variance across models
            - latency: Timing breakdown in ms
        """
        if not self.ensure_loaded():
            return self._error_result("Models not loaded")
        
        start_time = time.perf_counter()
        
        # Handle single sample case
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(1, -1)
        
        # Apply calibration if available
        if apply_calibration and self.calibrator is not None:
            emg_data = self.calibrator.apply_correction(emg_data)
        
        # Step 1: Preprocessing
        preprocess_start = time.perf_counter()
        processed_data = self.preprocessor.transform(emg_data)
        self._last_preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Step 2: Feature extraction
        feature_start = time.perf_counter()
        features = self.feature_extractor.extract_batch(processed_data)
        self._last_feature_time = (time.perf_counter() - feature_start) * 1000
        
        # Step 3: Multi-model inference
        inference_start = time.perf_counter()
        result = self._multi_model_predict(features)
        self._last_inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Total latency
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Add latency breakdown
        result['latency'] = {
            'preprocessing_ms': round(self._last_preprocess_time, 2),
            'feature_extraction_ms': round(self._last_feature_time, 2),
            'inference_ms': round(self._last_inference_time, 2),
            'total_ms': round(total_time, 2)
        }
        
        # Add timestamp
        result['timestamp'] = time.time()
        
        return result
    
    def _multi_model_predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run inference with all models and apply consensus voting.
        
        Args:
            features: Feature vector of shape (n_samples, num_features)
            
        Returns:
            Prediction result with consensus and variance
        """
        # Get predictions and probabilities from each model
        votes = {}
        probabilities = {}
        
        for name, model in self.models.items():
            # Predict class
            pred_encoded = model.predict(features)[0]
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            votes[name] = pred_label
            
            # Get probability for predicted class
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                probabilities[name] = float(np.max(proba))
            else:
                # For models without predict_proba, use 1.0 as placeholder
                probabilities[name] = 1.0
        
        # Majority voting
        vote_counts = Counter(votes.values())
        final_gesture, vote_count = vote_counts.most_common(1)[0]
        
        # Calculate confidence metrics
        confidences = list(probabilities.values())
        mean_confidence = np.mean(confidences)
        confidence_variance = np.var(confidences)
        
        # Check for uncertainty conditions
        # Uncertainty is flagged when:
        # 1. Models disagree significantly (no clear majority)
        # 2. Confidence variance is high
        # 3. Mean confidence is below threshold
        is_uncertain = (
            vote_count < 2 or  # Less than 2 models agree
            confidence_variance > UNCERTAINTY_VARIANCE_THRESHOLD or
            mean_confidence < MIN_CONFIDENCE_THRESHOLD
        )
        
        if is_uncertain:
            return {
                'gesture': 'uncertain',
                'confidence': None,
                'action': GESTURE_ACTION_MAP.get('uncertain', 'HOLD'),
                'model_votes': votes,
                'model_confidences': probabilities,
                'variance': round(confidence_variance, 4),
                'is_uncertain': True
            }
        
        return {
            'gesture': final_gesture,
            'confidence': round(mean_confidence, 4),
            'action': GESTURE_ACTION_MAP.get(final_gesture, 'HOLD'),
            'model_votes': votes,
            'model_confidences': probabilities,
            'variance': round(confidence_variance, 4),
            'is_uncertain': False
        }
    
    def predict_batch(self, emg_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        
        Args:
            emg_data: EMG data of shape (n_samples, num_channels)
            
        Returns:
            List of prediction results, one per sample
        """
        results = []
        for i in range(len(emg_data)):
            result = self.predict(emg_data[i])
            results.append(result)
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the Random Forest model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.ensure_loaded():
            return {}
        
        importance = self.models['random_forest'].feature_importances_
        feature_names = self.feature_extractor.get_feature_names()
        
        return dict(zip(feature_names, importance))
    
    def get_channel_importance(self) -> Dict[str, float]:
        """
        Get aggregated importance by channel.
        
        Returns:
            Dictionary mapping channel names to total importance
        """
        feature_importance = self.get_feature_importance()
        
        channel_importance = {}
        for ch in range(1, 9):
            ch_name = f'ch{ch}'
            mav_key = f'{ch_name}_mav'
            rms_key = f'{ch_name}_rms'
            
            ch_importance = (
                feature_importance.get(mav_key, 0) +
                feature_importance.get(rms_key, 0)
            )
            channel_importance[ch_name] = round(ch_importance, 4)
        
        return channel_importance
    
    def set_calibrator(self, calibrator) -> None:
        """
        Set the session calibrator for drift correction.
        
        Args:
            calibrator: SessionCalibrator instance
        """
        self.calibrator = calibrator
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get the latency from the last prediction.
        
        Returns:
            Dictionary with timing breakdown
        """
        return {
            'preprocessing_ms': round(self._last_preprocess_time, 2),
            'feature_extraction_ms': round(self._last_feature_time, 2),
            'inference_ms': round(self._last_inference_time, 2),
            'total_ms': round(
                self._last_preprocess_time +
                self._last_feature_time +
                self._last_inference_time, 2
            )
        }
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """
        Create an error result dictionary.
        
        Args:
            message: Error message
            
        Returns:
            Error result dictionary
        """
        return {
            'gesture': 'error',
            'confidence': None,
            'action': 'HOLD',
            'error': message,
            'is_uncertain': True
        }


# Global predictor instance for the Flask app
_global_predictor: Optional[GesturePredictor] = None


def get_predictor() -> GesturePredictor:
    """
    Get or create the global predictor instance.
    
    Returns:
        GesturePredictor instance (singleton pattern for Flask)
    """
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = GesturePredictor()
        _global_predictor.load_models()
    return _global_predictor
