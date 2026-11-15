#!/usr/bin/env python3

import logging
import argparse
from face_tracker_robot import FaceTrackerRobot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Face-following robot with Feetech servos')
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
        help='Servo ID for pan (rotation) joint (default: 1)'
    )
    parser.add_argument(
        '--tilt-id',
        type=int,
        default=2,
        help='Servo ID for tilt (flex) joint (default: 2)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    parser.add_argument(
        '--shape-predictor',
        type=str,
        default=None,
        help='Path to dlib shape predictor model (optional)'
    )
    parser.add_argument(
        '--mock-servos',
        action='store_true',
        help='Run in mock mode without actual servo hlklhllkk4444ardware (for testing camera/face detection)'
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        default='calibration.json',
        help='Path to calibration JSON file (default: calibration.json)'
    )
    parser.add_argument(
        '--camera-calibration-file',
        type=str,
        default='camera_calibration.json',
        help='Path to camera calibration JSON file for undistortion (default: camera_calibration.json)'
    )
    parser.add_argument(
        '--force-fisheye',
        action='store_true',
        help='Force fisheye undistortion mode even if calibration file does not specify'
    )
    parser.add_argument(
        '--upsample',
        type=int,
        default=1,
        help='Number of times to upsample image for face detection (0-2, default: 1, higher=better detection but slower)'
    )
    parser.add_argument(
        '--show-preprocessed',
        action='store_true',
        help='Show preprocessed image window for debugging'
    )
    parser.add_argument(
        '--heavy-denoising',
        action='store_true',
        help='Use heavy denoising (slower but better for very noisy images)'
    )
    parser.add_argument(
        '--use-cnn',
        action='store_true',
        help='Use CNN model for face detection (WARNING: Very slow, not recommended. HOG is now used by default for speed)'
    )
    parser.add_argument(
        '--detection-scale',
        type=float,
        default=0.5,
        help='Scale factor for face detection (0.3-1.0, default: 0.5). Lower = faster but less accurate'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1 = every frame). Higher = faster but less smooth'
    )
    parser.add_argument(
        '--disable-color-correction',
        action='store_true',
        help='Disable automatic color correction and white balance'
    )
    parser.add_argument(
        '--green-tint-reduction',
        type=float,
        default=0.3,
        help='Strength of green tint reduction (0.0-1.0, default: 0.3)'
    )
    parser.add_argument(
        '--white-balance-strength',
        type=float,
        default=0.7,
        help='Strength of white balance correction (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--rotate-180',
        action='store_true',
        help='Rotate camera image 180 degrees (upside down)'
    )
    parser.add_argument(
        '--invert-pan',
        action='store_true',
        help='Invert pan control direction (applied after rotation logic)'
    )
    parser.add_argument(
        '--invert-tilt',
        action='store_true',
        help='Invert tilt control direction (applied after rotation logic)'
    )
    parser.add_argument(
        '--min-face-size',
        type=int,
        default=15,
        help='Minimum face size (short side in pixels) to consider (default: 15)'
    )
    parser.add_argument(
        '--max-face-size',
        type=int,
        default=2000,
        help='Maximum face size (short side in pixels) to consider (default: 2000)'
    )
    parser.add_argument(
        '--show-all-detections',
        action='store_true',
        help='Draw all detected faces (selected face in green, others in yellow)'
    )
    parser.add_argument(
        '--no-detection-fallback',
        action='store_true',
        help='Disable adaptive fallback (higher scale/upsample) when no face is found'
    )
    parser.add_argument(
        '--max-fallback-upsamples',
        type=int,
        default=2,
        help='Max extra upsample attempts during fallback (0-2, default: 2)'
    )
    
    args = parser.parse_args()
    
    robot = FaceTrackerRobot(
        pan_servo_id=args.pan_id,
        tilt_servo_id=args.tilt_id,
        port=args.port,
        camera_index=args.camera,
        shape_predictor_path=args.shape_predictor,
        mock_servos=args.mock_servos,
        calibration_file=args.calibration_file,
        use_cnn=args.use_cnn,
        camera_calibration_file=args.camera_calibration_file,
        force_fisheye=args.force_fisheye
    )
    
    robot.upsample_count = max(0, min(2, args.upsample))
    robot.show_preprocessed = args.show_preprocessed
    robot.use_heavy_denoising = args.heavy_denoising
    robot.detection_scale = max(0.2, min(1.0, args.detection_scale))
    robot.frame_skip_interval = max(1, args.frame_skip)
    robot.enable_color_correction = not args.disable_color_correction
    robot.green_tint_reduction = max(0.0, min(1.0, args.green_tint_reduction))
    robot.white_balance_strength = max(0.0, min(1.0, args.white_balance_strength))
    robot.rotate_180 = args.rotate_180
    robot.invert_pan = args.invert_pan
    robot.invert_tilt = args.invert_tilt
    robot.min_face_size = max(1, args.min_face_size)
    robot.max_face_size = max(robot.min_face_size, args.max_face_size)
    robot.show_all_detections = args.show_all_detections
    robot.enable_detection_fallback = not args.no_detection_fallback
    robot.max_fallback_upsamples = max(0, min(2, args.max_fallback_upsamples))
    try:
        robot.run()
    finally:
        robot.disconnect()

if __name__ == '__main__':
    main()

