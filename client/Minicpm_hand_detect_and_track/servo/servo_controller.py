"""
Servo controller with PID-based tracking.

Manages pan/tilt servo control with PID feedback for smooth object tracking.
"""
import logging
from typing import Optional, Tuple
import numpy as np

from feetech_servo import FeetechServo
from calibration import CalibrationManager
from pid_controller import PIDController

logger = logging.getLogger(__name__)


class ServoController:
    """Controls pan and tilt servos for object tracking."""
    
    def __init__(
        self,
        pan_servo_id: int,
        tilt_servo_id: int,
        port: str,
        calibration_file: str,
        image_center: Tuple[int, int],
        pan_pid_config: dict,
        tilt_pid_config: dict,
        max_step_size: float = 0.02,
        mock_servos: bool = False
    ):
        """
        Initialize servo controller.
        
        Args:
            pan_servo_id: Servo ID for pan (horizontal) movement
            tilt_servo_id: Servo ID for tilt (vertical) movement
            port: Serial port for servo communication
            calibration_file: Path to calibration file
            image_center: Center coordinates of the image (width//2, height//2)
            pan_pid_config: PID configuration dict for pan servo
            tilt_pid_config: PID configuration dict for tilt servo
            max_step_size: Maximum step size for smooth motion
            mock_servos: If True, run without hardware
        """
        self.mock_servos = mock_servos
        self.image_center = image_center
        self.max_step_size = max_step_size
        
        # Current positions
        self.pan_position = 0.0
        self.tilt_position = 0.0
        
        # Last PID outputs and errors (for debugging/visualization)
        self.last_normalized_error_x = 0.0
        self.last_normalized_error_y = 0.0
        self.last_pan_output = 0.0
        self.last_tilt_output = 0.0
        
        # Exponential moving average for smoothing (reduces oscillations)
        self.smoothing_factor = 0.7  # Lower = smoother but slower
        self.smoothed_pan_output = 0.0
        self.smoothed_tilt_output = 0.0
        
        # Initialize PID controllers
        self.pan_pid = PIDController(**pan_pid_config)
        self.tilt_pid = PIDController(**tilt_pid_config)
        
        # Initialize servos
        if not self.mock_servos:
            self._init_hardware_servos(
                pan_servo_id, 
                tilt_servo_id, 
                port, 
                calibration_file
            )
        else:
            logger.info("Running in MOCK SERVO mode (no hardware)")
            self.pan_servo = None
            self.tilt_servo = None
    
    def _init_hardware_servos(
        self, 
        pan_id: int, 
        tilt_id: int, 
        port: str, 
        calibration_file: str
    ):
        """Initialize physical servo hardware."""
        # Load calibration
        self.calibration_manager = CalibrationManager(calibration_file)
        self.calibration_manager.load()
        
        pan_calib = self.calibration_manager.get_calibration("pan")
        tilt_calib = self.calibration_manager.get_calibration("tilt")
        
        # Initialize servos
        self.pan_servo = FeetechServo(
            pan_id,
            port,
            min_position=pan_calib.min_position if pan_calib else None,
            max_position=pan_calib.max_position if pan_calib else None,
            center_position=pan_calib.center_position if pan_calib else None,
        )
        self.tilt_servo = FeetechServo(
            tilt_id,
            port,
            min_position=tilt_calib.min_position if tilt_calib else None,
            max_position=tilt_calib.max_position if tilt_calib else None,
            center_position=tilt_calib.center_position if tilt_calib else None,
            port_handler=self.pan_servo.port_handler,
        )
        
        # Connect servos
        if not self.pan_servo.connect():
            logger.error("Failed to connect pan servo")
            self.pan_servo.disconnect()
            raise RuntimeError("Pan servo connection failed")
        
        if not self.tilt_servo.connect():
            logger.error("Failed to connect tilt servo")
            self.pan_servo.disconnect()
            raise RuntimeError("Tilt servo connection failed")
        
        # Read current positions
        pan_current = self.pan_servo.get_position_normalized()
        tilt_current = self.tilt_servo.get_position_normalized()
        
        if pan_current is not None:
            self.pan_position = pan_current
            logger.info(f"Pan servo current position: {pan_current:.3f}")
        else:
            logger.warning("Could not read pan servo position, starting at center")
            self.pan_position = 0.0
            
        if tilt_current is not None:
            self.tilt_position = tilt_current
            logger.info(f"Tilt servo current position: {tilt_current:.3f}")
        else:
            logger.warning("Could not read tilt servo position, starting at center")
            self.tilt_position = 0.0
        
        logger.info("Servos initialized successfully")
    
    def update_position(self, object_center: Tuple[int, int]):
        """
        Update servo positions to track object center using PID control.
        
        Args:
            object_center: Pixel coordinates of object center (x, y)
        """
        if self.image_center is None:
            return
        
        center_x, center_y = self.image_center
        obj_x, obj_y = object_center
        
        # Calculate pixel errors
        error_x = obj_x - center_x
        error_y = obj_y - center_y
        
        # Get frame dimensions
        frame_width, frame_height = self.image_center[0] * 2, self.image_center[1] * 2
        half_w = frame_width / 2.0 if frame_width > 0 else 1.0
        half_h = frame_height / 2.0 if frame_height > 0 else 1.0
        
        # Normalize errors
        normalized_error_x = error_x / half_w
        normalized_error_y = error_y / half_h
        
        self.last_normalized_error_x = normalized_error_x
        self.last_normalized_error_y = normalized_error_y
        
        # Apply PID
        pan_output = self.pan_pid.update(normalized_error_x)
        tilt_output = self.tilt_pid.update(normalized_error_y)
        
        # Apply exponential moving average smoothing to reduce jitter
        self.smoothed_pan_output = (
            self.smoothing_factor * pan_output + 
            (1 - self.smoothing_factor) * self.smoothed_pan_output
        )
        self.smoothed_tilt_output = (
            self.smoothing_factor * tilt_output + 
            (1 - self.smoothing_factor) * self.smoothed_tilt_output
        )
        
        self.last_pan_output = self.smoothed_pan_output
        self.last_tilt_output = self.smoothed_tilt_output
        
        # Limit step size for smooth motion
        pan_delta = np.clip(self.smoothed_pan_output, -self.max_step_size, self.max_step_size)
        tilt_delta = np.clip(self.smoothed_tilt_output, -self.max_step_size, self.max_step_size)
        
        # Update positions
        self.pan_position = np.clip(self.pan_position + pan_delta, -1.0, 1.0)
        self.tilt_position = np.clip(self.tilt_position + tilt_delta, -1.0, 1.0)
        
        # Send to servos (only if not in mock mode)
        if not self.mock_servos:
            success_pan = self.pan_servo.set_position_normalized(self.pan_position)
            success_tilt = self.tilt_servo.set_position_normalized(self.tilt_position)
            
            if not success_pan:
                logger.warning(f"Failed to set pan servo position: {self.pan_position:.3f}")
            if not success_tilt:
                logger.warning(f"Failed to set tilt servo position: {self.tilt_position:.3f}")
    
    def get_positions(self) -> Tuple[float, float]:
        """Get current servo positions."""
        return (self.pan_position, self.tilt_position)
    
    def disconnect(self):
        """Disconnect servos."""
        if not self.mock_servos and hasattr(self, 'pan_servo') and self.pan_servo:
            if hasattr(self, 'tilt_servo') and self.tilt_servo:
                self.tilt_servo.disconnect()   # disables torque, does not own port
            self.pan_servo.disconnect()        # disables torque, then closes port
