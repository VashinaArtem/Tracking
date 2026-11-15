"""
Performance metrics tracking for hand tracking system.

Provides utilities for tracking FPS, inference times, and network latency.
"""
from collections import deque
from typing import Optional


class PerformanceMetrics:
    """Tracks performance metrics for the tracking system."""
    
    def __init__(
        self,
        fps_window: int = 30,
        inference_window: int = 10,
        network_window: int = 10
    ):
        """
        Initialize performance metrics tracker.
        
        Args:
            fps_window: Number of frames to average for FPS calculation
            inference_window: Number of samples for inference time averaging
            network_window: Number of samples for network time averaging
        """
        self.fps_history = deque(maxlen=fps_window)
        self.inference_times = deque(maxlen=inference_window)
        self.network_times = deque(maxlen=network_window)
        
        self.last_inference_time = 0.0
    
    def add_fps_sample(self, fps: float):
        """Add a FPS sample to the history."""
        self.fps_history.append(fps)
    
    def add_inference_time(self, time_seconds: float):
        """Add an inference time sample."""
        self.inference_times.append(time_seconds)
        self.last_inference_time = time_seconds
    
    def add_network_time(self, time_seconds: float):
        """Add a network time sample."""
        self.network_times.append(time_seconds)
    
    def get_average_fps(self) -> Optional[float]:
        """Get average FPS, or None if no samples."""
        if not self.fps_history:
            return None
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_average_inference_time(self) -> Optional[float]:
        """Get average inference time, or None if no samples."""
        if not self.inference_times:
            return None
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_average_network_time(self) -> Optional[float]:
        """Get average network time, or None if no samples."""
        if not self.network_times:
            return None
        return sum(self.network_times) / len(self.network_times)
