"""
MiniCPM-V-4.5 GPU Tracking Server Package
Modular implementation for GPU-accelerated object tracking
"""

from .models import DetectionResponse
from .tracking_engine import TrackingServerGPU
from .config import ServerConfig
from .api import create_app

__version__ = "1.0.0"
__all__ = [
    "DetectionResponse",
    "TrackingServerGPU",
    "ServerConfig",
    "create_app",
]
