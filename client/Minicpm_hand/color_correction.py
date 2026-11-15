"""
Real-time camera color-correction tool.
Adjust exposure, contrast, black/white points, and related parameters.
"""
# python Minicpm_hand\color_correction.py --camera 0   

import cv2
import numpy as np
from typing import Dict, Any


class ColorCorrectionTool:
    def __init__(self, camera_id: int = 0):
        """
        Initialize the color-correction tool.

        Args:
            camera_id: Camera device ID (default: 0)
        """
        self.camera_id = camera_id
        self.cap = None
        self.window_name = "Color Correction Tool"
        
        # Color-correction parameters (defaults)
        self.params = {
            'brightness': 50,      # Brightness (0-100, 50 = normal)
            'contrast': 50,        # Contrast (0-100, 50 = normal)
            'exposure': 50,        # Exposure (0-100, 50 = auto)
            'saturation': 50,      # Saturation (0-100, 50 = normal)
            'hue': 0,              # Hue (-180 to +180, 0 = normal)
            'gamma': 50,           # Gamma (1-300, 50 = 1.0)
            'black_point': 0,      # Black point (0-255)
            'white_point': 255,    # White point (0-255)
            'sharpness': 50,       # Sharpness (0-100, 50 = normal)
            'temperature': 50,     # Color temperature (0-100, 50 = normal)
        }
        
        self.running = False
        
    def initialize_camera(self) -> bool:
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: failed to open camera {self.camera_id}")
            return False
        
        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Try enabling auto-exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto mode
        
        print(f"Camera {self.camera_id} initialized successfully")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
        
        return True
    
    def create_trackbars(self):
        """Create trackbars for parameter control."""
        cv2.namedWindow(self.window_name)
        
        # Create a trackbar for each parameter
        cv2.createTrackbar('Brightness', self.window_name, 
                          self.params['brightness'], 100, self.on_trackbar)
        cv2.createTrackbar('Contrast', self.window_name, 
                          self.params['contrast'], 100, self.on_trackbar)
        cv2.createTrackbar('Exposure', self.window_name, 
                          self.params['exposure'], 100, self.on_trackbar)
        cv2.createTrackbar('Saturation', self.window_name, 
                          self.params['saturation'], 100, self.on_trackbar)
        cv2.createTrackbar('Hue', self.window_name, 
                          180, 360, self.on_trackbar)  # -180 to +180
        cv2.createTrackbar('Gamma', self.window_name, 
                          self.params['gamma'], 300, self.on_trackbar)
        cv2.createTrackbar('Black Point', self.window_name, 
                          self.params['black_point'], 255, self.on_trackbar)
        cv2.createTrackbar('White Point', self.window_name, 
                          self.params['white_point'], 255, self.on_trackbar)
        cv2.createTrackbar('Sharpness', self.window_name, 
                          self.params['sharpness'], 100, self.on_trackbar)
        cv2.createTrackbar('Temperature', self.window_name, 
                          self.params['temperature'], 100, self.on_trackbar)
    
    def on_trackbar(self, value):
        """Trackbar callback (unused, but required by OpenCV)."""
        pass
    
    def get_trackbar_values(self):
        """Read current values from the trackbars."""
        self.params['brightness'] = cv2.getTrackbarPos('Brightness', self.window_name)
        self.params['contrast'] = cv2.getTrackbarPos('Contrast', self.window_name)
        self.params['exposure'] = cv2.getTrackbarPos('Exposure', self.window_name)
        self.params['saturation'] = cv2.getTrackbarPos('Saturation', self.window_name)
        self.params['hue'] = cv2.getTrackbarPos('Hue', self.window_name) - 180
        self.params['gamma'] = cv2.getTrackbarPos('Gamma', self.window_name)
        self.params['black_point'] = cv2.getTrackbarPos('Black Point', self.window_name)
        self.params['white_point'] = cv2.getTrackbarPos('White Point', self.window_name)
        self.params['sharpness'] = cv2.getTrackbarPos('Sharpness', self.window_name)
        self.params['temperature'] = cv2.getTrackbarPos('Temperature', self.window_name)
    
    def apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        brightness = (self.params['brightness'] - 50) * 2  # -100 to +100
        contrast = self.params['contrast'] / 50.0  # 0 to 2
        
        # Apply contrast and brightness
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        return image
    
    def apply_exposure(self, image: np.ndarray) -> np.ndarray:
        """
        Apply exposure.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        exposure = self.params['exposure'] / 50.0  # 0 to 2
        
        # Try setting camera exposure
        if self.params['exposure'] == 50:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, (exposure - 1) * 10)
        
        # Software exposure correction
        image = np.clip(image * exposure, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_saturation_hue(self, image: np.ndarray) -> np.ndarray:
        """
        Apply saturation and hue.

        Args:
            image: Input image

        Returns:
            Processed image
        """
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
    
    def apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        gamma = self.params['gamma'] / 50.0  # 0.02 to 6
        
        if gamma == 1.0:
            return image
        
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Apply gamma correction
        image = cv2.LUT(image, table)
        
        return image
    
    def apply_black_white_points(self, image: np.ndarray) -> np.ndarray:
        """
        Apply black and white points.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        black_point = self.params['black_point']
        white_point = max(self.params['white_point'], black_point + 1)
        
        # Apply levels
        image = image.astype(np.float32)
        image = np.clip((image - black_point) * (255.0 / (white_point - black_point)), 0, 255)
        image = image.astype(np.uint8)
        
        return image
    
    def apply_sharpness(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpness.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        sharpness = (self.params['sharpness'] - 50) / 50.0  # -1 to 1
        
        if abs(sharpness) < 0.01:
            return image
        
        # Build sharpening kernel
        if sharpness > 0:
            # Sharpen
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * sharpness
            kernel[1, 1] = 1 + 8 * sharpness
        else:
            # Blur
            kernel_size = int(abs(sharpness) * 10) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return image
        
        # Apply sharpening filter
        image = cv2.filter2D(image, -1, kernel)
        
        return image
    
    def apply_temperature(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color temperature.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        temperature = (self.params['temperature'] - 50) / 50.0  # -1 to 1
        
        if abs(temperature) < 0.01:
            return image
        
        # Use float for more precise math
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
    
    def apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all color-correction effects.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        # Apply effects in a fixed order
        image = self.apply_exposure(image)
        image = self.apply_black_white_points(image)
        image = self.apply_gamma(image)
        image = self.apply_brightness_contrast(image)
        image = self.apply_saturation_hue(image)
        image = self.apply_temperature(image)
        image = self.apply_sharpness(image)
        
        return image
    
    def draw_info(self, image: np.ndarray) -> np.ndarray:
        """
        Draw parameter info onto the image.

        Args:
            image: Input image

        Returns:
            Image with overlay text
        """
        info_lines = [
            f"Brightness: {self.params['brightness']}",
            f"Contrast: {self.params['contrast']}",
            f"Exposure: {self.params['exposure']}",
            f"Saturation: {self.params['saturation']}",
            f"Hue: {self.params['hue']}",
            f"Gamma: {self.params['gamma'] / 50.0:.2f}",
            f"Black Point: {self.params['black_point']}",
            f"White Point: {self.params['white_point']}",
            f"Sharpness: {self.params['sharpness']}",
            f"Temperature: {self.params['temperature']}",
        ]
        
        # Semi-transparent background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 270), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        
        # Draw text
        y_offset = 30
        for line in info_lines:
            cv2.putText(image, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Control hints
        cv2.putText(image, "Q - Quit | R - Reset | S - Save settings", 
                   (20, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image
    
    def reset_parameters(self):
        """Reset parameters to defaults."""
        self.params = {
            'brightness': 50,
            'contrast': 50,
            'exposure': 50,
            'saturation': 50,
            'hue': 0,
            'gamma': 50,
            'black_point': 0,
            'white_point': 255,
            'sharpness': 50,
            'temperature': 50,
        }
        
        # Update trackbars
        cv2.setTrackbarPos('Brightness', self.window_name, 50)
        cv2.setTrackbarPos('Contrast', self.window_name, 50)
        cv2.setTrackbarPos('Exposure', self.window_name, 50)
        cv2.setTrackbarPos('Saturation', self.window_name, 50)
        cv2.setTrackbarPos('Hue', self.window_name, 180)
        cv2.setTrackbarPos('Gamma', self.window_name, 50)
        cv2.setTrackbarPos('Black Point', self.window_name, 0)
        cv2.setTrackbarPos('White Point', self.window_name, 255)
        cv2.setTrackbarPos('Sharpness', self.window_name, 50)
        cv2.setTrackbarPos('Temperature', self.window_name, 50)
        
        print("Parameters reset to defaults")
    
    def save_parameters(self, filename: str = "color_correction_params.txt"):
        """
        Save parameters to a file.

        Args:
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Color-correction parameters\n")
            for key, value in self.params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Parameters saved to {filename}")
    
    def run(self):
        """Main processing loop."""
        if not self.initialize_camera():
            return
        
        self.create_trackbars()
        self.running = True
        
        print("\n=== Color Correction Tool ===")
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset parameters")
        print("  S - Save parameters")
        print("\nUse the sliders to adjust parameters")
        print("=====================================\n")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: failed to grab a frame from the camera")
                break
            
            # Read current parameter values
            self.get_trackbar_values()
            
            # Apply color correction
            corrected_frame = self.apply_color_correction(frame.copy())
            
            # Build display image (original + corrected)
            # Downscale for comfortable display
            display_height = 480
            scale = display_height / frame.shape[0]
            display_width = int(frame.shape[1] * scale)
            
            original_resized = cv2.resize(frame, (display_width, display_height))
            corrected_resized = cv2.resize(corrected_frame, (display_width, display_height))
            
            # Add labels
            cv2.putText(original_resized, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(corrected_resized, "Corrected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Concatenate images horizontally
            combined = np.hstack([original_resized, corrected_resized])
            
            # Overlay parameter info
            combined = self.draw_info(combined)
            
            # Show result
            cv2.imshow(self.window_name, combined)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                self.running = False
            elif key == ord('r') or key == ord('R'):  # R - reset
                self.reset_parameters()
            elif key == ord('s') or key == ord('S'):  # S - save
                self.save_parameters()
        
        # Release resources
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera color-correction tool')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Create and run the tool
    tool = ColorCorrectionTool(camera_id=args.camera)
    tool.run()


if __name__ == "__main__":
    main()
