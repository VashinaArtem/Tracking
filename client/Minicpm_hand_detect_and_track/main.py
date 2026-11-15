"""
Main entry point for MiniCPM-V Hand Tracking Client.

Handles command-line arguments and starts the tracking system.
"""
import argparse
import sys

from config.settings import TrackingConfig
from hand_tracking_client import HandTrackingClient


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MiniCPM-V Hand Tracking Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server settings
    parser.add_argument(
        "--server", 
        type=str, 
        default="http://localhost:8000",
        help="Server URL"
    )
    
    # Camera settings
    parser.add_argument(
        "--camera", 
        type=int, 
        default=1,
        help="Camera device ID (0 for laptop webcam, 1 for robot camera)"
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera width in pixels"
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera height in pixels"
    )
    
    # Image processing
    parser.add_argument(
        "--fps", 
        type=float, 
        default=1.5,
        help="Target FPS for sending frames to server (1-2 recommended)"
    )
    parser.add_argument(
        "--quality", 
        type=int, 
        default=85,
        help="JPEG quality (0-100, higher = better quality)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Max image dimension in pixels (384=fast, 512=balanced, 768=quality)"
    )
    
    # Servo settings
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbserial-0001",
        help="Serial port for servos (macOS: /dev/tty.usbserial-*, Windows: COM3, Linux: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--pan-id",
        type=int,
        default=1,
        help="Servo ID for pan joint"
    )
    parser.add_argument(
        "--tilt-id",
        type=int,
        default=2,
        help="Servo ID for tilt joint"
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default="calibration.json",
        help="Path to servo calibration file"
    )
    parser.add_argument(
        "--mock-servos",
        action="store_true",
        help="Run without servo hardware (for testing)"
    )
    
    # Tracking settings
    parser.add_argument(
        "--tracker",
        type=str,
        default="KCF",
        choices=["KCF", "CSRT"],
        help="OpenCV tracker type (KCF=balanced, CSRT=accurate but slower)"
    )
    parser.add_argument(
        "--llm-check-interval",
        type=float,
        default=2.0,
        help="Seconds between LLM re-verification checks"
    )
    
    # Voice settings
    parser.add_argument(
        "--voice",
        action="store_true",
        default=True,
        help="Enable voice control with Whisper"
    )
    parser.add_argument(
        "--no-voice",
        dest="voice",
        action="store_false",
        help="Disable voice control"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    
    # Camera calibration
    parser.add_argument(
        "--camera-calibration",
        type=str,
        default=None,
        help="Path to camera calibration JSON file (for fisheye/lens distortion correction)"
    )
    
    # UI settings
    parser.add_argument(
        "--clean-ui",
        action="store_true",
        help="Show only bbox on frame, hide all text overlays"
    )
    
    args = parser.parse_args()
    
    # Build configuration from arguments
    config = TrackingConfig(
        server_url=args.server,
        camera_id=args.camera,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        target_fps=args.fps,
        jpeg_quality=args.quality,
        image_max_size=args.image_size,
        pan_servo_id=args.pan_id,
        tilt_servo_id=args.tilt_id,
        port=args.port,
        calibration_file=args.calibration_file,
        mock_servos=args.mock_servos,
        tracker_type=args.tracker,
        llm_check_interval=args.llm_check_interval,
        enable_voice=args.voice,
        whisper_model=args.whisper_model,
        camera_calibration_file=args.camera_calibration,
        clean_ui=args.clean_ui
    )
    
    try:
        client = HandTrackingClient(config)
        client.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
