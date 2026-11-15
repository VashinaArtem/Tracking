"""
Color correction utilities for camera images.

Provides functions for adjusting brightness, contrast, saturation,
gamma, temperature, and other color parameters.
"""
import cv2
import numpy as np
from typing import Dict


class ColorCorrector:
    """Applies color correction to camera frames."""
    
    def __init__(self, params: Dict[str, int], cap: cv2.VideoCapture):
        """
        Initialize color corrector.
        
        Args:
            params: Dictionary of color correction parameters
            cap: OpenCV VideoCapture object for camera control
        """
        self.params = params
        self.cap = cap
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all color corrections to the image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Color corrected image
        """
        # Apply effects in order
        image = self._apply_exposure(image)
        image = self._apply_black_white_points(image)
        image = self._apply_gamma(image)
        image = self._apply_brightness_contrast(image)
        image = self._apply_saturation_hue(image)
        image = self._apply_temperature(image)
        image = self._apply_sharpness(image)
        
        return image
    
    def _apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        brightness = (self.params['brightness'] - 50) * 2  # -100 to +100
        contrast = self.params['contrast'] / 50.0  # 0 to 2
        
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return image
    
    def _apply_exposure(self, image: np.ndarray) -> np.ndarray:
        """Apply exposure adjustment."""
        exposure = self.params['exposure'] / 50.0  # 0 to 2
        
        # Try to set camera exposure
        if self.params['exposure'] == 50:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, (exposure - 1) * 10)
        
        # Software exposure correction
        image = np.clip(image * exposure, 0, 255).astype(np.uint8)
        return image
    
    def _apply_saturation_hue(self, image: np.ndarray) -> np.ndarray:
        """Apply saturation and hue adjustments."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Apply hue shift
        hue_shift = self.params['hue']
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Apply saturation
        saturation = self.params['saturation'] / 50.0  # 0 to 2
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Convert back to BGR
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image
    
    def _apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction."""
        gamma = self.params['gamma'] / 50.0  # 0.02 to 6
        
        if gamma == 1.0:
            return image
        
        # Create lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Apply gamma correction
        image = cv2.LUT(image, table)
        return image
    
    def _apply_black_white_points(self, image: np.ndarray) -> np.ndarray:
        """Apply black and white point adjustments."""
        black_point = self.params['black_point']
        white_point = max(self.params['white_point'], black_point + 1)
        
        # Apply levels
        image = image.astype(np.float32)
        image = np.clip((image - black_point) * (255.0 / (white_point - black_point)), 0, 255)
        image = image.astype(np.uint8)
        return image
    
    def _apply_sharpness(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpness adjustment."""
        sharpness = (self.params['sharpness'] - 50) / 50.0  # -1 to 1
        
        if abs(sharpness) < 0.01:
            return image
        
        # Create sharpness kernel
        if sharpness > 0:
            # Sharpen
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * sharpness
            kernel[1, 1] = 1 + 8 * sharpness
            image = cv2.filter2D(image, -1, kernel)
        else:
            # Blur
            kernel_size = int(abs(sharpness) * 10) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def _apply_temperature(self, image: np.ndarray) -> np.ndarray:
        """Apply color temperature adjustment."""
        temperature = (self.params['temperature'] - 50) / 50.0  # -1 to 1
        
        if abs(temperature) < 0.01:
            return image
        
        # Convert to float for accurate calculations
        image = image.astype(np.float32)
        
        if temperature > 0:
            # Warm tones (more red/yellow)
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 + temperature * 0.3), 0, 255)  # Red
            image[:, :, 1] = np.clip(image[:, :, 1] * (1 + temperature * 0.1), 0, 255)  # Green
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 - temperature * 0.2), 0, 255)  # Blue
        else:
            # Cool tones (more blue)
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 - temperature * 0.3), 0, 255)  # Blue
            image[:, :, 1] = np.clip(image[:, :, 1] * (1 - temperature * 0.1), 0, 255)  # Green
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 + temperature * 0.2), 0, 255)  # Red
        
        return image.astype(np.uint8)
