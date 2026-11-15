"""
Message types for the parallel tracking pipeline.

These dataclasses are passed between threads via queues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    ts: float
    frame_id: int


@dataclass(frozen=True)
class LLMTask:
    frame: np.ndarray
    prompt: str
    ts: float
    frame_id: int


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    prompt: str
    ts: float
    frame_id: int
    bbox: Optional[List[int]] = None
    frame: Optional[np.ndarray] = None
    inference_time: float = 0.0
    network_time: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class DisplayPacket:
    frame: np.ndarray
    bbox: Optional[List[int]]
    prompt: str

    avg_fps: Optional[float]
    avg_inference: Optional[float]
    avg_network: Optional[float]

    is_tracking: bool
    tracker_type: str
    time_to_next_check: float

    pan_position: float
    tilt_position: float
    pan_output: float
    tilt_output: float

    voice_state: str  # "idle" | "active" | "processing"
    server_online: bool
    image_center: Optional[Tuple[int, int]]

    bbox_age_ms: Optional[float] = None

