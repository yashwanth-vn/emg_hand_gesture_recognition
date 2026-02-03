"""
Signal Processing Package

This package contains modules for EMG signal preprocessing and feature extraction.
All processing follows the same pipeline regardless of data source.
"""
from .preprocessing import EMGPreprocessor
from .feature_extraction import FeatureExtractor

__all__ = ['EMGPreprocessor', 'FeatureExtractor']
