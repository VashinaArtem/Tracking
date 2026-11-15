"""
Coordinate conversion utilities for bounding boxes.

This module provides pure functions for converting between normalized (0-1000)
and pixel coordinate systems, as well as calculating bbox centers.
"""
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_bbox_center(bbox: List[int], frame_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert bbox [x1,y1,x2,y2] (0-1000) to pixel center coordinates.
    
    Args:
        bbox: Normalized bounding box [x1, y1, x2, y2] in range 0-1000
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Tuple of (center_x, center_y) in pixel coordinates
    """
    h, w = frame_shape[:2]
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x, center_y)


def normalize_bbox_to_pixels(
    bbox: List[int], 
    frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Convert normalized bbox [x1,y1,x2,y2] (0-1000) to pixel coordinates (x,y,w,h).
    
    Validates and clamps bbox to frame bounds to prevent tracker initialization failures.
    
    Args:
        bbox: Normalized bounding box [x1, y1, x2, y2] in range 0-1000
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Tuple of (x, y, width, height) in pixel coordinates
        
    Raises:
        ValueError: If bbox results in non-positive dimensions
    """
    h, w = frame_shape[:2]
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    
    # Validate and clamp to frame bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    width = x2 - x1
    height = y2 - y1
    
    # Ensure positive dimensions
    if width <= 0 or height <= 0:
        logger.error(f"Invalid bbox dimensions: width={width}, height={height} from bbox={bbox}")
        raise ValueError(f"Invalid bbox: {bbox} resulted in non-positive dimensions")
    
    return (x1, y1, width, height)


def pixels_to_normalized_bbox(
    bbox_pixels: Tuple[int, int, int, int], 
    frame_shape: Tuple[int, int]
) -> List[int]:
    """
    Convert pixel bbox (x,y,w,h) to normalized [x1,y1,x2,y2] (0-1000).
    
    Args:
        bbox_pixels: Bounding box in pixel coordinates (x, y, width, height)
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Normalized bounding box [x1, y1, x2, y2] in range 0-1000
    """
    h, w = frame_shape[:2]
    x, y, width, height = bbox_pixels
    x1 = int((x / w) * 1000)
    y1 = int((y / h) * 1000)
    x2 = int(((x + width) / w) * 1000)
    y2 = int(((y + height) / h) * 1000)
    return [x1, y1, x2, y2]
