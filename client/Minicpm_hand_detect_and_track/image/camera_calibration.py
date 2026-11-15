"""
Camera calibration for lens distortion correction.

Handles loading camera calibration data and undistorting images
to remove fisheye/lens distortion effects.
"""
import json
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Handles camera calibration for lens distortion correction."""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize camera calibrator.
        
        Args:
            calibration_file: Path to camera calibration JSON file (optional)
        """
        self.calibration_file = calibration_file
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.image_size: Optional[tuple] = None
        self.model: str = "pinhole"  # default
        self.enabled = False
        
        if calibration_file:
            self.load_calibration(calibration_file)
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load camera calibration from JSON file.
        
        Args:
            filepath: Path to calibration JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            calib_path = Path(filepath)
            if not calib_path.exists():
                logger.warning(f"Camera calibration file not found: {filepath}")
                return False
            
            with open(calib_path, 'r') as f:
                data = json.load(f)
            
            # Load parameters
            self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
            self.dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
            self.image_size = tuple(data.get('image_size', [640, 480]))
            self.model = data.get('model', 'pinhole')
            
            self.enabled = True
            logger.info(f"Loaded camera calibration from {filepath}")
            logger.info(f"  Model: {self.model}")
            logger.info(f"  Image size: {self.image_size}")
            logger.info(f"  Camera matrix shape: {self.camera_matrix.shape}")
            logger.info(f"  Distortion coeffs: {self.dist_coeffs.tolist()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load camera calibration: {e}")
            self.enabled = False
            return False
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Undistorted image
        """
        if not self.enabled:
            return image
        
        try:
            if self.model == "fisheye":
                # Use fisheye undistortion
                h, w = image.shape[:2]
                
                # Create new camera matrix for undistorted image
                new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.camera_matrix,
                    self.dist_coeffs,
                    (w, h),
                    np.eye(3),
                    balance=0.0
                )
                
                # Create undistortion maps
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.camera_matrix,
                    self.dist_coeffs,
                    np.eye(3),
                    new_camera_matrix,
                    (w, h),
                    cv2.CV_16SC2
                )
                
                # Apply undistortion
                undistorted = cv2.remap(
                    image,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                
            else:
                # Standard pinhole model undistortion
                undistorted = cv2.undistort(
                    image,
                    self.camera_matrix,
                    self.dist_coeffs
                )
            
            return undistorted
            
        except Exception as e:
            logger.error(f"Error during undistortion: {e}")
            return image
    
    def is_enabled(self) -> bool:
        """Check if calibration is loaded and enabled."""
        return self.enabled
