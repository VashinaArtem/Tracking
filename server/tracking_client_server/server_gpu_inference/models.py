"""
Data models for tracking server
"""

from typing import Optional, List
from pydantic import BaseModel


class DetectionResponse(BaseModel):
    """Response model for object detection."""
    success: bool
    bbox: Optional[List[int]] = None
    error: Optional[str] = None
    inference_time: float
    prompt: str


class ServerStats(BaseModel):
    """Server statistics model."""
    total_requests: int
    successful_detections: int
    success_rate: str
    avg_inference_time: str
    device: str
    dtype: str
    multi_gpu_mode: str
