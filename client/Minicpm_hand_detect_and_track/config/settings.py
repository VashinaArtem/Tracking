"""
Configuration settings for hand tracking system.

Defines dataclasses for system configuration.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackingConfig:
    """Main configuration for the tracking system."""
    
    # Server settings
    server_url: str = "http://localhost:8000"
    
    # Camera settings
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    
    # Image processing
    target_fps: float = 1.5
    jpeg_quality: int = 85
    image_max_size: int = 512
    
    # Servo settings
    pan_servo_id: int = 1
    tilt_servo_id: int = 2
    port: str = "COM3"  # Windows default, use /dev/ttyUSB0 on Linux
    calibration_file: str = "calibration.json"
    mock_servos: bool = False
    
    # Tracking settings
    tracker_type: str = "KCF"  # KCF or CSRT
    llm_check_interval: float = 2.0  # Seconds between LLM re-verification

    # Pipeline / threading settings
    frame_queue_size: int = 30  # bounded queue for no_drop + backpressure
    ui_queue_size: int = 1      # latest-only display packet queue
    llm_queue_size: int = 1     # size-1 for tasks/results
    ui_poll_timeout_ms: int = 50  # UI loop poll timeout
    
    # Voice settings
    enable_voice: bool = True
    whisper_model: str = "base"  # tiny, base, small, medium, large
    
    # Camera calibration (for lens distortion correction)
    camera_calibration_file: Optional[str] = None  # Path to camera_calibration.json

    # UI settings
    clean_ui: bool = False  # Show only bbox, hide all text overlays


@dataclass
class PIDConfig:
    """PID controller configuration."""
    
    kp: float = 0.02     # Уменьшено с 0.15 - меньше агрессивность
    ki: float = 0.0      # Интегральная составляющая (пока 0)
    kd: float = 0.0     # Добавлено демпфирование для устранения колебаний
    min_output: float = -1.0
    max_output: float = 1.0
    integral_limit: float = 0.15
    deadband: float = 0.015  # Увеличено с 0.003 - игнорируем малые ошибки


@dataclass
class ServoConfig:
    """Servo control configuration."""
    
    max_step_size: float = 0.02  # Уменьшено с 0.05 - более плавное движение
    pan_pid: PIDConfig = field(default_factory=PIDConfig)
    tilt_pid: PIDConfig = field(default_factory=PIDConfig)


@dataclass
class VoiceConfig:
    """Voice control configuration."""
    
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # VAD supports 10, 20, or 30 ms
    vad_aggressiveness: int = 2  # 0-3
    silence_threshold: int = 20  # Silent chunks before wake word processing (20 × 30ms = 600ms)
    min_speech_chunks: int = 10  # Minimum speech chunks to process in IDLE
    language: str = "en"  # Language for Whisper transcription
    wake_word: str = "atlas"
    command_silence_threshold: int = 100  # Silent chunks before command processing (100 × 30ms = 3000ms)
    max_active_chunks: int = 1000         # Max buffer size in ACTIVE state (~30 sec)
