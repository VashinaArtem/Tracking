"""
OpenCV tracker management.

Provides a wrapper for OpenCV trackers (KCF, CSRT) with state management.
"""
import logging
from typing import Optional, List, Tuple
import cv2
import numpy as np

from utils.coordinates import normalize_bbox_to_pixels, pixels_to_normalized_bbox

logger = logging.getLogger(__name__)


class TrackerManager:
    """Manages OpenCV tracker instances for object tracking."""
    
    def __init__(self, tracker_type: str = "KCF"):
        """
        Initialize tracker manager.
        
        Args:
            tracker_type: Type of tracker to use ("KCF" or "CSRT")
        """
        self.tracker_type = tracker_type
        self.tracker: Optional[cv2.Tracker] = None
        self.tracking_active = False
        self.current_bbox: Optional[List[int]] = None  # Normalized bbox [x1,y1,x2,y2]
    
    def _create_tracker(self) -> cv2.Tracker:
        """Create OpenCV tracker based on tracker_type."""
        if self.tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        else:  # Default to KCF
            if self.tracker_type != "KCF":
                logger.warning(f"Unknown tracker type: {self.tracker_type}, using KCF")
            return cv2.legacy.TrackerKCF_create()
    
    def reset(self):
        """Reset tracker state and release resources."""
        if self.tracker is not None:
            self.tracker = None  # Let GC handle OpenCV tracker cleanup
        self.tracking_active = False
        self.current_bbox = None
    
    def init_with_bbox(
        self, 
        frame: np.ndarray, 
        bbox: List[int]
    ) -> bool:
        """
        Initialize tracker with LLM-detected bbox.
        
        Args:
            frame: Current video frame (BGR)
            bbox: Normalized bbox [x1,y1,x2,y2] (0-1000)
            
        Returns:
            True if tracker initialized successfully, False otherwise
        """
        try:
            # Reset any existing tracker first
            self.reset()
            
            # Create new tracker instance
            self.tracker = self._create_tracker()
            bbox_pixels = normalize_bbox_to_pixels(bbox, frame.shape)
            success = self.tracker.init(frame, bbox_pixels)
            
            # OpenCV tracker.init() may return None on success (version-dependent)
            # Check for explicit False rather than truthy success
            if success is False:
                logger.warning("Tracker initialization failed")
                self.reset()
                return False
            else:
                # None or True both indicate success
                self.tracking_active = True
                self.current_bbox = bbox
                logger.info(f"Tracker initialized: {self.tracker_type} at {bbox_pixels}")
                return True
        except ValueError as e:
            # Bbox validation failed
            logger.error(f"Invalid bbox for tracker init: {e}")
            self.reset()
            return False
        except Exception as e:
            logger.error(f"Error initializing tracker: {e}")
            self.reset()
            return False
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[List[int]]]:
        """
        Update tracker with new frame.
        
        Args:
            frame: Current video frame (BGR)
            
        Returns:
            Tuple of (success, bbox) where:
            - success: True if tracking succeeded
            - bbox: Normalized bbox [x1,y1,x2,y2] (0-1000) or None if failed
        """
        if not self.tracking_active or self.tracker is None:
            return False, None
        
        try:
            success, bbox_pixels = self.tracker.update(frame)
            
            if success and bbox_pixels is not None and len(bbox_pixels) == 4:
                # Convert tracker bbox back to normalized format
                self.current_bbox = pixels_to_normalized_bbox(bbox_pixels, frame.shape)
                return True, self.current_bbox
            else:
                # Tracking lost
                logger.warning("Tracker lost object")
                self.reset()
                return False, None
        except Exception as e:
            logger.error(f"Error updating tracker: {e}")
            self.reset()
            return False, None
    
    def is_active(self) -> bool:
        """Check if tracker is currently active."""
        return self.tracking_active and self.tracker is not None
    
    def get_current_bbox(self) -> Optional[List[int]]:
        """Get current tracked bbox in normalized coordinates."""
        return self.current_bbox
