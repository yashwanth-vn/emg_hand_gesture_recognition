"""
Machine Learning Package

This package contains modules for model training and inference.
"""
from .training import ModelTrainer
from .inference import GesturePredictor
from .calibration import SessionCalibrator

__all__ = ['ModelTrainer', 'GesturePredictor', 'SessionCalibrator']
