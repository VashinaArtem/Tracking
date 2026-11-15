#!/usr/bin/env python3

import logging
import argparse
import time
from feetech_servo import FeetechServo
from calibration import CalibrationManager, ServoCalibration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calibrate_servo(servo: FeetechServo, motor_name: str) -> ServoCalibration:
    logger.info(f"\n=== Calibrating {motor_name} (servo ID: {servo.servo_id}) ===")
    
    if not servo.connect():
        raise RuntimeError(f"Failed to connect to servo {servo.servo_id}")
        
    try:
        logger.info("Disabling torque for safe calibration...")
        servo.disable_torque()
        
        input(f"\nMove {motor_name} to its MINIMUM safe position, then press ENTER...")
        min_pos = servo.get_position()
        if min_pos is None:
            raise RuntimeError("Failed to read servo position")
        logger.info(f"Recorded minimum position: {min_pos}")
        
        input(f"\nMove {motor_name} to its CENTER position, then press ENTER...")
        center_pos = servo.get_position()
        if center_pos is None:
            raise RuntimeError("Failed to read servo position")
        logger.info(f"Recorded center position: {center_pos}")
        
        input(f"\nMove {motor_name} to its MAXIMUM safe position, then press ENTER...")
        max_pos = servo.get_position()
        if max_pos is None:
            raise RuntimeError("Failed to read servo position")
        logger.info(f"Recorded maximum position: {max_pos}")
        
        if min_pos >= max_pos:
            raise ValueError(
                f"Invalid positions: min ({min_pos}) must be less than max ({max_pos})"
            )
        if not (min_pos <= center_pos <= max_pos):
            raise ValueError(
                f"Center position ({center_pos}) must be between min ({min_pos}) and max ({max_pos})"
            )
            
        calibration = ServoCalibration(
            servo_id=servo.servo_id,
            min_position=min_pos,
            max_position=max_pos,
            center_position=center_pos,
        )
        
        logger.info(f"\n{motor_name} calibration complete:")
        logger.info(f"  Min: {calibration.min_position}")
        logger.info(f"  Center: {calibration.center_position}")
        logger.info(f"  Max: {calibration.max_position}")
        
        return calibration
        
    finally:
        servo.enable_torque()
        servo.disconnect()


def main():
    parser = argparse.ArgumentParser(description='Calibrate Feetech servos')
    parser.add_argument(
        '--port',
        type=str,
        default='/dev/ttyUSB0',
        help='Serial port for servo communication (default: /dev/ttyUSB0)'
    )
    parser.add_argument(
        '--pan-id',
        type=int,
        default=1,
        help='Servo ID for pan joint (default: 1)'
    )
    parser.add_argument(
        '--tilt-id',
        type=int,
        default=2,
        help='Servo ID for tilt joint (default: 2)'
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        default='calibration.json',
        help='Path to calibration JSON file (default: calibration.json)'
    )
    
    args = parser.parse_args()
    
    calib_manager = CalibrationManager(args.calibration_file)
    
    logger.info("Starting calibration process...")
    logger.info("Make sure the robot is in a safe position and torque is disabled.")
    logger.info("You will be prompted to move each joint to min, center, and max positions.\n")
    
    input("Press ENTER when ready to begin calibration...")
    
    pan_servo = FeetechServo(args.pan_id, args.port)
    tilt_servo = FeetechServo(args.tilt_id, args.port)
    
    try:
        pan_calib = calibrate_servo(pan_servo, "pan")
        calib_manager.set_calibration("pan", pan_calib)
        
        tilt_calib = calibrate_servo(tilt_servo, "tilt")
        calib_manager.set_calibration("tilt", tilt_calib)
        
        if calib_manager.save():
            logger.info(f"\n✓ Calibration saved to {args.calibration_file}")
            logger.info("You can now run the face tracker with calibrated servos.")
        else:
            logger.error("\n✗ Failed to save calibration file")
            
    except KeyboardInterrupt:
        logger.info("\nCalibration cancelled by user")
    except Exception as e:
        logger.error(f"\nCalibration failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()

