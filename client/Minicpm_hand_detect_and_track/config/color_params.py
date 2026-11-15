"""
Color correction parameters for camera image processing.
"""
from dataclasses import dataclass


@dataclass
class ColorCorrectionParams:
    """Parameters for color correction of camera images."""
    
    brightness: int = 50       # Brightness (0-100, 50 = normal)
    contrast: int = 50         # Contrast (0-100, 50 = normal)
    exposure: int = 50         # Exposure (0-100, 50 = auto)
    saturation: int = 50       # Saturation (0-100, 50 = normal)
    hue: int = 0               # Hue (-180 to +180, 0 = normal)
    gamma: int = 50            # Gamma (1-300, 50 = 1.0)
    black_point: int = 0       # Black point (0-255)
    white_point: int = 255     # White point (0-255)
    sharpness: int = 50        # Sharpness (0-100, 50 = normal)
    temperature: int = 50      # Color temperature (0-100, 50 = normal)
    
    def to_dict(self):
        """Convert to dictionary for compatibility."""
        return {
            'brightness': self.brightness,
            'contrast': self.contrast,
            'exposure': self.exposure,
            'saturation': self.saturation,
            'hue': self.hue,
            'gamma': self.gamma,
            'black_point': self.black_point,
            'white_point': self.white_point,
            'sharpness': self.sharpness,
            'temperature': self.temperature,
        }
