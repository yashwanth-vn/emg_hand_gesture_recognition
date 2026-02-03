"""
Model Training Pipeline

This module handles training of the gesture recognition models:
- Random Forest (primary production model)
- Histogram-Based Gradient Boosting
- Logistic Regression

All three models are trained and persisted for consensus voting during inference.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    TRAINING_DATA_PATH,
    MODELS_DIR,
    RANDOM_FOREST_MODEL_PATH,
    HISTGB_MODEL_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
    LABEL_ENCODER_PATH,
    NORMALIZATION_STATS_PATH,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    HISTGB_MAX_ITER,
    HISTGB_MAX_DEPTH,
    HISTGB_LEARNING_RATE,
    LR_MAX_ITER,
    LR_C,
    GESTURE_CLASSES
)
from src.signal_processing.preprocessing import EMGPreprocessor
from src.signal_processing.feature_extraction import FeatureExtractor


class ModelTrainer:
    """
    Trains and evaluates gesture recognition models.
    
    This class handles the complete training pipeline:
    1. Load and validate training data
    2. Preprocess and extract features
    3. Train three different models
    4. Evaluate performance
    5. Persist models and preprocessing artifacts
    
    Attributes:
        preprocessor: EMGPreprocessor instance
        feature_extractor: FeatureExtractor instance
        label_encoder: LabelEncoder for gesture classes
        models: Dictionary of trained models
    """
    
    def __init__(self):
        """Initialize the trainer with preprocessing components."""
        self.preprocessor = EMGPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        
        self.models: Dict[str, Any] = {}
        self.training_metrics: Dict[str, Any] = {}
    
    def load_data(self, filepath: str = TRAINING_DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from CSV file.
        
        Args:
            filepath: Path to the training CSV file
            
        Returns:
            Tuple of (X, y) where X is EMG data and y is labels
            
        Expected CSV format:
        - Columns: ch1, ch2, ..., ch8, label
        - Label column contains gesture names
        """
        print(f"Loading training data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Validate structure
        expected_cols = [f'ch{i}' for i in range(1, 9)] + ['label']
        if list(df.columns) != expected_cols:
            # Try to handle different column names
            if len(df.columns) == 9:
                # Assume last column is label
                df.columns = expected_cols
            else:
                raise ValueError(f"Expected columns {expected_cols}, got {list(df.columns)}")
        
        # Extract features and labels
        X = df[[f'ch{i}' for i in range(1, 9)]].values.astype(np.float64)
        y = df['label'].values
        
        print(f"Loaded {len(X)} samples with {len(np.unique(y))} classes")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data and split into train/test sets.
        
        Args:
            X: Raw EMG data
            y: Gesture labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) with features extracted
        """
        print("Preprocessing data and extracting features...")
        
        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Stratified split to maintain class distribution
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_SEED,
            stratify=y_encoded  # Ensures proportional class representation
        )
        
        print(f"Train set: {len(X_train_raw)} samples")
        print(f"Test set: {len(X_test_raw)} samples")
        
        # Fit preprocessor on training data only (prevent data leakage)
        self.preprocessor.fit(X_train_raw)
        
        # Transform both sets using training statistics
        X_train_processed = self.preprocessor.transform(X_train_raw)
        X_test_processed = self.preprocessor.transform(X_test_raw)
        
        # Extract features
        X_train_features = self.feature_extractor.extract_batch(X_train_processed)
        X_test_features = self.feature_extractor.extract_batch(X_test_processed)
        
        print(f"Feature vector size: {X_train_features.shape[1]}")
        
        return X_train_features, X_test_features, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all three models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained model objects
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Model 1: Random Forest (Primary Production Model)
        # Random Forest is chosen as primary because:
        # - Handles non-linear relationships well
        # - Provides feature importance
        # - Robust to overfitting with proper parameters
        # - Good balance of accuracy and speed
        print("\n[1/3] Training Random Forest (Primary Model)...")
        rf_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RANDOM_SEED,
            n_jobs=-1  # Use all CPU cores
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        print("  [OK] Random Forest trained")
        
        # Model 2: Histogram-Based Gradient Boosting
        # HistGradientBoosting is used because:
        # - Very fast training and inference
        # - Handles large datasets efficiently
        # - Built-in handling of missing values
        print("\n[2/3] Training Histogram Gradient Boosting...")
        histgb_model = HistGradientBoostingClassifier(
            max_iter=HISTGB_MAX_ITER,
            max_depth=HISTGB_MAX_DEPTH,
            learning_rate=HISTGB_LEARNING_RATE,
            random_state=RANDOM_SEED
        )
        histgb_model.fit(X_train, y_train)
        self.models['hist_gradient_boosting'] = histgb_model
        print("  [OK] Histogram Gradient Boosting trained")
        
        # Model 3: Logistic Regression
        # Logistic Regression is included because:
        # - Fast inference (important for <100ms latency)
        # - Provides probability estimates
        # - Different decision boundary from tree-based models
        print("\n[3/3] Training Logistic Regression...")
        lr_model = LogisticRegression(
            max_iter=LR_MAX_ITER,
            C=LR_C,
            random_state=RANDOM_SEED,
            multi_class='multinomial',
            solver='lbfgs'
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        print("  [OK] Logistic Regression trained")
        
        return self.models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n--- {name.replace('_', ' ').title()} ---")
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            
            # Get class names for report
            class_names = self.label_encoder.classes_
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=class_names)
            print(f"\nClassification Report:\n{report}")
            
            # Store metrics
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        # Highlight primary model
        primary_accuracy = results['random_forest']['accuracy']
        print(f"\n{'='*60}")
        print(f"PRIMARY MODEL (Random Forest) ACCURACY: {primary_accuracy:.4f}")
        print(f"{'='*60}")
        
        self.training_metrics = results
        return results
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the Random Forest model.
        
        Returns:
            Array of importance scores for each feature
        """
        if 'random_forest' not in self.models:
            raise RuntimeError("Random Forest model not trained yet")
        
        return self.models['random_forest'].feature_importances_
    
    def get_channel_importance(self) -> Dict[str, float]:
        """
        Aggregate feature importance by channel.
        
        Returns:
            Dictionary mapping channel names to importance scores
        """
        importance = self.get_feature_importance()
        feature_names = self.feature_extractor.get_feature_names()
        
        channel_importance = {}
        for ch in range(1, 9):
            ch_name = f'ch{ch}'
            # Sum importance of both features for this channel
            ch_indices = [i for i, name in enumerate(feature_names) if name.startswith(ch_name)]
            channel_importance[ch_name] = sum(importance[i] for i in ch_indices)
        
        return channel_importance
    
    def save_models(self) -> None:
        """
        Save all trained models and preprocessing artifacts.
        """
        print("\nSaving models and preprocessing artifacts...")
        
        # Ensure directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save models
        joblib.dump(self.models['random_forest'], RANDOM_FOREST_MODEL_PATH)
        print(f"  [OK] Random Forest saved to {RANDOM_FOREST_MODEL_PATH}")
        
        joblib.dump(self.models['hist_gradient_boosting'], HISTGB_MODEL_PATH)
        print(f"  [OK] Hist Gradient Boosting saved to {HISTGB_MODEL_PATH}")
        
        joblib.dump(self.models['logistic_regression'], LOGISTIC_REGRESSION_MODEL_PATH)
        print(f"  [OK] Logistic Regression saved to {LOGISTIC_REGRESSION_MODEL_PATH}")
        
        # Save preprocessing artifacts
        self.preprocessor.save(NORMALIZATION_STATS_PATH)
        print(f"  [OK] Normalization stats saved to {NORMALIZATION_STATS_PATH}")
        
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        print(f"  [OK] Label encoder saved to {LABEL_ENCODER_PATH}")
    
    def run_full_pipeline(self) -> Dict[str, Dict]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Evaluation metrics for all models
        """
        # Load data
        X, y = self.load_data()
        
        # Prepare data (preprocess + feature extraction)
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train all models
        self.train_models(X_train, y_train)
        
        # Evaluate performance
        results = self.evaluate_models(X_test, y_test)
        
        # Save everything
        self.save_models()
        
        # Print feature importance
        print("\n" + "="*60)
        print("CHANNEL IMPORTANCE (from Random Forest)")
        print("="*60)
        importance = self.get_channel_importance()
        for ch, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            bar = '#' * int(imp * 50)
            print(f"{ch}: {imp:.4f} {bar}")
        
        return results


def train_models_if_needed() -> bool:
    """
    Check if models exist and train if they don't.
    
    Returns:
        True if training was performed, False if models already exist
    """
    models_exist = all([
        os.path.exists(RANDOM_FOREST_MODEL_PATH),
        os.path.exists(HISTGB_MODEL_PATH),
        os.path.exists(LOGISTIC_REGRESSION_MODEL_PATH),
        os.path.exists(NORMALIZATION_STATS_PATH),
        os.path.exists(LABEL_ENCODER_PATH)
    ])
    
    if models_exist:
        print("Models already exist. Skipping training.")
        return False
    
    print("Models not found. Starting training pipeline...")
    trainer = ModelTrainer()
    trainer.run_full_pipeline()
    return True


if __name__ == '__main__':
    # Run training when executed directly
    trainer = ModelTrainer()
    trainer.run_full_pipeline()
