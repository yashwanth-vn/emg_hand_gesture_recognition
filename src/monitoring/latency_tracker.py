"""
Latency-Aware Inference Monitoring

This module tracks and reports latency metrics for the inference pipeline.
It measures:
- Preprocessing time
- Feature extraction time
- Model inference time
- Total prediction time

Target latency: < 100ms for real-time gesture recognition.
"""
import time
from typing import Dict, List, Optional
from collections import deque
import statistics

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TARGET_LATENCY_MS


class LatencyTracker:
    """
    Tracks latency metrics for the prediction pipeline.
    
    This class records timing for each stage of prediction and provides
    statistics for monitoring and optimization.
    
    Key metrics tracked:
    - Preprocessing latency
    - Feature extraction latency
    - Model inference latency
    - Total end-to-end latency
    
    Attributes:
        target_latency_ms: Target maximum latency in milliseconds
        history_size: Number of recent readings to track
    """
    
    def __init__(self, history_size: int = 100, target_ms: float = TARGET_LATENCY_MS):
        """
        Initialize the latency tracker.
        
        Args:
            history_size: Number of readings to keep for statistics
            target_ms: Target latency in milliseconds
        """
        self.target_latency_ms = target_ms
        self.history_size = history_size
        
        # Use deques for efficient fixed-size history
        self._preprocess_history = deque(maxlen=history_size)
        self._feature_history = deque(maxlen=history_size)
        self._inference_history = deque(maxlen=history_size)
        self._total_history = deque(maxlen=history_size)
        
        # Timing state for active measurement
        self._current_start: Optional[float] = None
        self._stage_times: Dict[str, float] = {}
        
        # Count of predictions that exceeded target
        self._exceeded_count = 0
        self._total_count = 0
    
    def start_prediction(self) -> None:
        """Start timing a new prediction."""
        self._current_start = time.perf_counter()
        self._stage_times = {}
    
    def mark_stage(self, stage: str) -> float:
        """
        Mark the completion of a stage and record elapsed time.
        
        Args:
            stage: Name of the completed stage
            
        Returns:
            Time in milliseconds since the last mark
        """
        current = time.perf_counter()
        
        if not self._stage_times:
            # First stage
            elapsed = (current - self._current_start) * 1000
        else:
            # Subsequent stages - time since last mark
            last_mark = max(self._stage_times.values())
            elapsed = (current - last_mark) * 1000
        
        self._stage_times[stage] = current
        return elapsed
    
    def end_prediction(self, preprocess_ms: float, feature_ms: float, inference_ms: float) -> Dict[str, float]:
        """
        End timing and record the metrics.
        
        Args:
            preprocess_ms: Preprocessing time in ms
            feature_ms: Feature extraction time in ms
            inference_ms: Inference time in ms
            
        Returns:
            Dictionary with all timing information
        """
        total_ms = preprocess_ms + feature_ms + inference_ms
        
        # Record in history
        self._preprocess_history.append(preprocess_ms)
        self._feature_history.append(feature_ms)
        self._inference_history.append(inference_ms)
        self._total_history.append(total_ms)
        
        # Track target compliance
        self._total_count += 1
        if total_ms > self.target_latency_ms:
            self._exceeded_count += 1
        
        return {
            'preprocessing_ms': round(preprocess_ms, 2),
            'feature_extraction_ms': round(feature_ms, 2),
            'inference_ms': round(inference_ms, 2),
            'total_ms': round(total_ms, 2),
            'within_target': total_ms <= self.target_latency_ms
        }
    
    def get_current_stats(self) -> Dict[str, any]:
        """
        Get current latency statistics.
        
        Returns:
            Dictionary with mean, median, max, and compliance metrics
        """
        if not self._total_history:
            return {
                'mean_total_ms': 0.0,
                'median_total_ms': 0.0,
                'max_total_ms': 0.0,
                'min_total_ms': 0.0,
                'target_ms': self.target_latency_ms,
                'compliance_rate': 1.0,
                'sample_count': 0
            }
        
        total_list = list(self._total_history)
        
        return {
            'mean_total_ms': round(statistics.mean(total_list), 2),
            'median_total_ms': round(statistics.median(total_list), 2),
            'max_total_ms': round(max(total_list), 2),
            'min_total_ms': round(min(total_list), 2),
            'target_ms': self.target_latency_ms,
            'compliance_rate': round(1 - (self._exceeded_count / max(1, self._total_count)), 4),
            'sample_count': len(total_list)
        }
    
    def get_breakdown_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics breakdown by pipeline stage.
        
        Returns:
            Nested dictionary with stats for each stage
        """
        def calc_stats(values: deque) -> Dict[str, float]:
            if not values:
                return {'mean': 0.0, 'median': 0.0, 'max': 0.0}
            vals = list(values)
            return {
                'mean': round(statistics.mean(vals), 2),
                'median': round(statistics.median(vals), 2),
                'max': round(max(vals), 2)
            }
        
        return {
            'preprocessing': calc_stats(self._preprocess_history),
            'feature_extraction': calc_stats(self._feature_history),
            'inference': calc_stats(self._inference_history),
            'total': calc_stats(self._total_history)
        }
    
    def get_latest(self) -> Dict[str, float]:
        """
        Get the most recent latency readings.
        
        Returns:
            Dictionary with latest timings
        """
        return {
            'preprocessing_ms': self._preprocess_history[-1] if self._preprocess_history else 0.0,
            'feature_extraction_ms': self._feature_history[-1] if self._feature_history else 0.0,
            'inference_ms': self._inference_history[-1] if self._inference_history else 0.0,
            'total_ms': self._total_history[-1] if self._total_history else 0.0
        }
    
    def is_within_target(self) -> bool:
        """
        Check if the latest prediction was within target latency.
        
        Returns:
            True if within target, False otherwise
        """
        if not self._total_history:
            return True
        return self._total_history[-1] <= self.target_latency_ms
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self._preprocess_history.clear()
        self._feature_history.clear()
        self._inference_history.clear()
        self._total_history.clear()
        self._exceeded_count = 0
        self._total_count = 0


# Global tracker instance
_global_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    """
    Get or create the global latency tracker.
    
    Returns:
        LatencyTracker instance (singleton pattern)
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LatencyTracker()
    return _global_tracker
