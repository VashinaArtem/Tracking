"""
MiniCPM-V Hand Tracking Client
Integrated system combining voice-controlled object detection with servo tracking

Features:
- Voice control via Whisper for object prompts
- MiniCPM-V server for object detection
- Pan/tilt servo control with PID tracking
- Real-time video display with bounding boxes
"""

import io
import time
import threading
import logging
from collections import deque
from typing import Optional, List, Tuple
import cv2
import numpy as np
import requests
from PIL import Image

import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

from feetech_servo import FeetechServo
from calibration import CalibrationManager
from pid_controller import PIDController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandTrackingClient:
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        camera_id: int = 0,
        pan_servo_id: int = 1,
        tilt_servo_id: int = 2,
        port: str = "COM3",  # Windows default, use /dev/ttyUSB0 on Linux
        target_fps: float = 1.5,
        jpeg_quality: int = 85,
        image_max_size: int = 512,
        enable_voice: bool = True,
        whisper_model: str = "base",
        mock_servos: bool = False,
        calibration_file: str = "calibration.json"
    ):
        """
        Initialize the hand tracking client.
        
        Args:
            server_url: URL of the tracking server
            camera_id: Camera device ID (usually 0 for default webcam)
            pan_servo_id: Servo ID for pan (horizontal) movement
            tilt_servo_id: Servo ID for tilt (vertical) movement
            port: Serial port for servo communication (COM3 on Windows, /dev/ttyUSB0 on Linux)
            target_fps: Target FPS for sending frames to server (1-2 recommended)
            jpeg_quality: JPEG compression quality (0-100)
            image_max_size: Max image dimension in pixels (smaller = faster)
            enable_voice: Enable voice control with Whisper
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            mock_servos: Run without servo hardware (for testing)
            calibration_file: Path to servo calibration file
        """
        print("=" * 60)
        print("MiniCPM-V Hand Tracking Client")
        print("=" * 60)
        
        self.server_url = server_url.rstrip('/')
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_request_time = 0
        self.jpeg_quality = jpeg_quality
        self.image_max_size = image_max_size
        self.mock_servos = mock_servos
        
        # Control flag (must be set early for threads)
        self.running = True
        
        # Current prompt and detection results
        self.current_prompt = ""
        self.current_bbox = None
        self.last_inference_time = 0.0
        self.prompt_lock = threading.Lock()

        # Shared state for parallel pipeline
        # - latest frame: produced by capture thread, consumed by UI + detector
        # - latest bbox: produced by detector, consumed by UI + control loop
        # - prompt_id: increments on any prompt change to drop stale detections
        self._prompt_id = 0
        self._frame_lock = threading.Lock()
        self._bbox_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0  # time.monotonic()
        self._latest_frame_id: int = 0
        self._latest_frame_shape: Optional[Tuple[int, ...]] = None
        self._latest_bbox: Optional[List[int]] = None
        self._latest_bbox_ts: float = 0.0  # time.monotonic()
        self._bbox_frame_id: int = 0

        # Thread handles (started in run)
        self.capture_thread: Optional[threading.Thread] = None
        self.detector_thread: Optional[threading.Thread] = None
        self.control_thread: Optional[threading.Thread] = None

        # Control-loop parameters
        self.control_hz: float = 50.0
        self.bbox_ttl_s: float = 1.0
        self._detector_offline_backoff_s: float = 1.0
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.inference_times = deque(maxlen=10)
        self.network_times = deque(maxlen=10)
        
        # Connection status
        self.server_online = False
        self.last_error = ""

        # Servo and tracking state
        self.image_center = None
        self.pan_position = 0.0
        self.tilt_position = 0.0
        
        # PID controllers for servo tracking
        self.pan_pid = PIDController(
            kp=0.15, ki=0.00, kd=0.0,
            min_output=-1.0, max_output=1.0,
            integral_limit=0.15, deadband=0.003
        )
        self.tilt_pid = PIDController(
            kp=0.15, ki=0.00, kd=0.0,
            min_output=-1.0, max_output=1.0,
            integral_limit=0.15, deadband=0.003
        )
        
        self.max_step_size = 0.05
        self.last_normalized_error_x = 0.0
        self.last_normalized_error_y = 0.0
        self.last_pan_output = 0.0
        self.last_tilt_output = 0.0

        # Initialize servos
        self._init_servos(pan_servo_id, tilt_servo_id, port, calibration_file)

        # Voice control setup
        self.enable_voice = enable_voice
        self.whisper_model = whisper_model
        
        # Initialize voice state (always, to avoid attribute errors)
        self.is_listening = False
        self.audio_buffer = []
        self.silence_chunks = 0
        self.speech_chunks = 0
        self.silence_threshold = 20  # Number of silent chunks before processing
        self.min_speech_chunks = 10  # Minimum speech chunks to process
        
        if self.enable_voice:
            self._init_voice()
        
        # Check server
        print(f"\n[1/3] Checking server connection...")
        print(f"Server URL: {self.server_url}")
        if not self._check_server():
            print(f"WARNING: Cannot connect to server at {self.server_url}")
            print(f"Error: {self.last_error}")
            print(f"Make sure the server is running")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Server not accessible")
        else:
            print("Server is online and ready")
        
        # Initialize camera
        print(f"\n[2/3] Initializing camera (ID: {camera_id})...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera resolution to 640x480 (better than Andrew_camera's 320x240)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {actual_width}x{actual_height}")
        print(f"Image resize: max {self.image_max_size}px (reduces tokens for faster inference)")
        
        self.image_center = (actual_width // 2, actual_height // 2)
        
        # Color correction parameters (based on screenshot settings)
        self.color_params = {
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
        print(f"Color correction initialized with default parameters")
        
        # Start prompt input thread
        print("\n[3/3] Starting prompt input thread...")
        self.prompt_thread = threading.Thread(target=self._prompt_input_loop, daemon=True)
        self.prompt_thread.start()
        print("Ready!")
        
        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        if self.enable_voice:
            print("- VOICE: Just speak naturally! Say 'clear' or 'quit'")
        print("- TEXT: Type your prompt in the console and press Enter")
        print("- Example: 'hand', 'person', 'cup', 'phone'")
        print("- Press 'q' in the video window to quit")
        print("- Press 'c' in the video window to clear current prompt")
        print("- Press 's' to show server statistics")
        print("=" * 60 + "\n")

    def _init_servos(self, pan_id: int, tilt_id: int, port: str, calibration_file: str):
        """Initialize servo control system."""
        if self.mock_servos:
            logger.info("Running in MOCK SERVO mode (no hardware)")
            self.pan_servo = None
            self.tilt_servo = None
            return
        
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

    def _init_voice(self):
        """Initialize voice control system."""
        print("\n[VOICE] Initializing Whisper...")
        try:
            self.whisper = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
            self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3 (2 = moderate)
            
            self.voice_thread = threading.Thread(target=self._voice_input_loop, daemon=True)
            self.voice_thread.start()
            print("[VOICE] Voice control initialized")
        except Exception as e:
            print(f"[VOICE] Failed to initialize voice control: {e}")
            print("[VOICE] Continuing with text input only...")
            self.enable_voice = False
    
    def _check_server(self) -> bool:
        """Check if server is accessible."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                self.server_online = True
                return True
            else:
                self.last_error = f"Server returned status {response.status_code}"
                self.server_online = False
                return False
        except requests.exceptions.ConnectionError:
            self.last_error = "Connection refused - is the server running?"
            self.server_online = False
            return False
        except requests.exceptions.Timeout:
            self.last_error = "Connection timeout"
            self.server_online = False
            return False
        except Exception as e:
            self.last_error = str(e)
            self.server_online = False
            return False
    
    def _prompt_input_loop(self):
        """Background thread for receiving prompt input from console."""
        print("\n[CONSOLE] Ready for prompt input. Type and press Enter:\n")
        while self.running:
            try:
                user_input = input("> ").strip()
                if user_input:
                    with self.prompt_lock:
                        self.current_prompt = user_input
                        self._prompt_id += 1
                        self.current_bbox = None  # Back-compat for overlay/drawing
                    with self._bbox_lock:
                        self._latest_bbox = None
                    print(f"[CONSOLE] Prompt updated: '{user_input}'")
            except EOFError:
                break
            except Exception as e:
                print(f"[CONSOLE ERROR] {e}")
    
    def _voice_input_loop(self):
        """Background thread for voice input using VAD + Whisper"""
        if not self.enable_voice:
            return
        
        # Audio parameters (VAD requires 16kHz, 16-bit mono)
        sample_rate = 16000
        frame_duration_ms = 30  # VAD supports 10, 20, or 30 ms
        frame_size = int(sample_rate * frame_duration_ms / 1000)  # 480 samples
        
        def audio_callback(indata, frames, time_info, status):
            """Called by sounddevice for each audio chunk"""
            if status:
                print(f"[VOICE] Audio status: {status}")
            
            # Convert float32 to int16 for VAD
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            
            # VAD requires exactly frame_size samples
            if len(audio_int16) == frame_size:
                try:
                    # Check if this frame contains speech
                    is_speech = self.vad.is_speech(audio_int16.tobytes(), sample_rate)
                    
                    if is_speech:
                        self.speech_chunks += 1
                        self.silence_chunks = 0
                        
                        if not self.is_listening:
                            self.is_listening = True
                            print("[VOICE] Speech detected, listening...")
                        
                        # Buffer audio while speech is detected
                        self.audio_buffer.append(audio_int16)
                    
                    else:  # Silence
                        if self.is_listening:
                            self.silence_chunks += 1
                            # Continue buffering during short pauses
                            self.audio_buffer.append(audio_int16)
                            
                            # If enough silence, process the buffered audio
                            if self.silence_chunks >= self.silence_threshold:
                                if self.speech_chunks >= self.min_speech_chunks:
                                    # Process in separate thread to avoid blocking audio
                                    audio_to_process = np.concatenate(self.audio_buffer)
                                    threading.Thread(
                                        target=self._process_voice_buffer,
                                        args=(audio_to_process, sample_rate),
                                        daemon=True
                                    ).start()
                                
                                # Reset state
                                self.is_listening = False
                                self.audio_buffer = []
                                self.silence_chunks = 0
                                self.speech_chunks = 0
                
                except Exception as e:
                    print(f"[VOICE ERROR] VAD processing: {e}")
        
        # Start audio stream
        try:
            # Get default input device info
            try:
                default_device = sd.query_devices(kind='input')
                print(f"[VOICE] Using microphone: {default_device['name']}")
            except:
                pass
            
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                blocksize=frame_size,
                callback=audio_callback
            ):
                print("[VOICE] Listening... Speak naturally!")
                # Keep thread alive while running
                while self.running:
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"[VOICE ERROR] Failed to start audio stream: {e}")
            print("[VOICE] Voice control disabled. Text input still available.")
    
    def _process_voice_buffer(self, audio_data: np.ndarray, sample_rate: int):
        """Process buffered audio with Whisper (runs in separate thread)"""
        try:
            print("[VOICE] Transcribing...")
            start_time = time.time()
            
            # Convert int16 to float32 for Whisper
            audio_float32 = audio_data.astype(np.float32) / 32767.0
            
            # Transcribe with Whisper
            segments, info = self.whisper.transcribe(
                audio_float32,
                language="en",  # Set to None for auto-detect
                beam_size=5,
                vad_filter=True,  # Additional VAD filtering
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Extract text from segments
            transcription = " ".join([segment.text.strip() for segment in segments])
            transcription = transcription.strip()
            
            elapsed = time.time() - start_time
            
            if transcription:
                # Check for voice commands
                transcription_lower = transcription.lower()
                
                if "clear" in transcription_lower or "stop" in transcription_lower:
                    with self.prompt_lock:
                        self.current_prompt = ""
                        self._prompt_id += 1
                        self.current_bbox = None
                    with self._bbox_lock:
                        self._latest_bbox = None
                    print(f"[VOICE] Cleared prompt ({elapsed:.2f}s)")
                
                elif "quit" in transcription_lower or "exit" in transcription_lower:
                    print(f"[VOICE] Quit command received ({elapsed:.2f}s)")
                    self.running = False
                
                else:
                    # Update tracking prompt
                    with self.prompt_lock:
                        self.current_prompt = transcription
                        self._prompt_id += 1
                        self.current_bbox = None  # Reset bbox when prompt changes
                    with self._bbox_lock:
                        self._latest_bbox = None
                    print(f"[VOICE] Transcribed: '{transcription}' ({elapsed:.2f}s)")
            else:
                print(f"[VOICE] No speech detected ({elapsed:.2f}s)")
        
        except Exception as e:
            print(f"[VOICE ERROR] Transcription failed: {e}")
    
    def _frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        """Convert OpenCV frame to JPEG bytes with smart resizing to reduce image tokens."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to reduce image tokens (faster inference!)
        h, w = frame_rgb.shape[:2]
        if max(h, w) > self.image_max_size:
            scale = self.image_max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Make dimensions multiple of 16 (optimal for vision encoders)
            new_w = (new_w // 16) * 16
            new_h = (new_h // 16) * 16
            if new_w > 0 and new_h > 0:
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Compress to JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=self.jpeg_quality)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _send_frame_to_server(self, frame: np.ndarray, prompt: str) -> Optional[List[int]]:
        """
        Send frame to server for detection.
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            prompt: Text description of object to detect
        
        Returns:
            bbox: [x1, y1, x2, y2] or None if detection failed
        """
        try:
            network_start = time.time()
            
            # Convert frame to JPEG
            jpeg_bytes = self._frame_to_jpeg_bytes(frame)
            
            # Prepare request
            files = {'image': ('frame.jpg', jpeg_bytes, 'image/jpeg')}
            data = {'prompt': prompt}
            
            # Send request
            response = requests.post(
                f"{self.server_url}/detect",
                files=files,
                data=data,
                timeout=120
            )
            
            network_time = time.time() - network_start
            self.network_times.append(network_time)
            
            if response.status_code == 200:
                result = response.json()
                
                if result['success']:
                    bbox = result['bbox']
                    self.last_inference_time = result['inference_time']
                    self.inference_times.append(self.last_inference_time)
                    
                    print(f"[DETECTION] Found: {bbox} | "
                          f"Inference: {self.last_inference_time:.2f}s | "
                          f"Network: {network_time:.2f}s")
                    
                    self.server_online = True
                    return bbox
                else:
                    print(f"[DETECTION] Not found | "
                          f"Inference: {result['inference_time']:.2f}s | "
                          f"Network: {network_time:.2f}s")
                    self.server_online = True
                    return None
            else:
                print(f"[ERROR] Server returned status {response.status_code}")
                self.server_online = False
                return None
        
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to server at {self.server_url}")
            self.server_online = False
            return None
        except requests.exceptions.Timeout:
            print(f"[ERROR] Request timeout - server may be overloaded")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            return None
    
    def _calculate_bbox_center(self, bbox: List[int], frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Convert bbox [x1,y1,x2,y2] (0-1000) to pixel center coordinates."""
        h, w = frame_shape[:2]
        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)
    
    def _update_servo_positions(self, object_center: Tuple[int, int]):
        """Update servo positions to track object center."""
        if self.image_center is None:
            return
        
        if self.mock_servos:
            # In mock mode, just update positions without hardware
            center_x, center_y = self.image_center
            obj_x, obj_y = object_center
            
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
            
            self.last_pan_output = pan_output
            self.last_tilt_output = tilt_output
            
            # Update virtual positions
            pan_delta = np.clip(pan_output, -self.max_step_size, self.max_step_size)
            tilt_delta = np.clip(tilt_output, -self.max_step_size, self.max_step_size)
            
            self.pan_position = np.clip(self.pan_position + pan_delta, -1.0, 1.0)
            self.tilt_position = np.clip(self.tilt_position + tilt_delta, -1.0, 1.0)
            return
        
        # Real servo control
        center_x, center_y = self.image_center
        obj_x, obj_y = object_center
        
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
        
        self.last_pan_output = pan_output
        self.last_tilt_output = tilt_output
        
        # Limit step size for smooth motion
        pan_delta = np.clip(pan_output, -self.max_step_size, self.max_step_size)
        tilt_delta = np.clip(tilt_output, -self.max_step_size, self.max_step_size)
        
        # Update positions
        self.pan_position = np.clip(self.pan_position + pan_delta, -1.0, 1.0)
        self.tilt_position = np.clip(self.tilt_position + tilt_delta, -1.0, 1.0)
        
        # Send to servos
        success_pan = self.pan_servo.set_position_normalized(self.pan_position)
        success_tilt = self.tilt_servo.set_position_normalized(self.tilt_position)
        
        if not success_pan:
            logger.warning(f"Failed to set pan servo position: {self.pan_position:.3f}")
        if not success_tilt:
            logger.warning(f"Failed to set tilt servo position: {self.tilt_position:.3f}")
    
    def _draw_bbox_on_frame(self, frame: np.ndarray, bbox: List[int], 
                           color=(0, 0, 255), thickness=3) -> np.ndarray:
        """Draw bounding box on OpenCV frame."""
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        with self.prompt_lock:
            label = self.current_prompt
        label_size, _ = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            2
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (center_x, center_y), 12, (0, 0, 255), 2)
        
        return frame
    
    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current prompt
        with self.prompt_lock:
            prompt_text = self.current_prompt if self.current_prompt else "No prompt set"
        cv2.putText(frame, f"Prompt: {prompt_text}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            cv2.putText(frame, f"Display FPS: {avg_fps:.1f}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Inference time
        if self.inference_times:
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            cv2.putText(frame, f"Inference: {avg_inference:.2f}s", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Network time
        if self.network_times:
            avg_network = sum(self.network_times) / len(self.network_times)
            cv2.putText(frame, f"Network: {avg_network:.2f}s", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Servo positions
        cv2.putText(frame, f"Pan: {self.pan_position:.3f}, Tilt: {self.tilt_position:.3f}", 
                   (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # PID outputs
        cv2.putText(frame, f"PID: Pan={self.last_pan_output:.3f} Tilt={self.last_tilt_output:.3f}", 
                   (20, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Voice status
        if self.enable_voice:
            voice_color = (0, 255, 255) if self.is_listening else (128, 128, 128)
            voice_text = "Voice: LISTENING" if self.is_listening else "Voice: Ready"
            cv2.putText(frame, voice_text, (20, 215),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Server status
        status_color = (0, 255, 0) if self.server_online else (0, 0, 255)
        status_text = "Server: ONLINE" if self.server_online else "Server: OFFLINE"
        cv2.putText(frame, status_text, (w - 200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Image center crosshair
        if self.image_center:
            cv2.circle(frame, self.image_center, 15, (255, 0, 0), 2)
            cv2.circle(frame, self.image_center, 3, (255, 0, 0), -1)
        
        # Controls
        cv2.putText(frame, "q:quit | c:clear | s:stats", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def _show_server_stats(self):
        """Fetch and display server statistics."""
        try:
            response = requests.get(f"{self.server_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print("\n" + "=" * 60)
                print("SERVER STATISTICS")
                print("=" * 60)
                for key, value in stats.items():
                    print(f"{key:.<30} {value}")
                print("=" * 60 + "\n")
            else:
                print(f"[ERROR] Cannot fetch stats: status {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Cannot fetch stats: {e}")
    
    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction to the image based on current parameters.
        
        Args:
            image: Input image
            
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
        brightness = (self.color_params['brightness'] - 50) * 2  # -100 to +100
        contrast = self.color_params['contrast'] / 50.0  # 0 to 2
        
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return image
    
    def _apply_exposure(self, image: np.ndarray) -> np.ndarray:
        """Apply exposure adjustment."""
        exposure = self.color_params['exposure'] / 50.0  # 0 to 2
        
        # Try to set camera exposure
        if self.color_params['exposure'] == 50:
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
        hue_shift = self.color_params['hue']
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Apply saturation
        saturation = self.color_params['saturation'] / 50.0  # 0 to 2
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Convert back to BGR
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image
    
    def _apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction."""
        gamma = self.color_params['gamma'] / 50.0  # 0.02 to 6
        
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
        black_point = self.color_params['black_point']
        white_point = max(self.color_params['white_point'], black_point + 1)
        
        # Apply levels
        image = image.astype(np.float32)
        image = np.clip((image - black_point) * (255.0 / (white_point - black_point)), 0, 255)
        image = image.astype(np.uint8)
        return image
    
    def _apply_sharpness(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpness adjustment."""
        sharpness = (self.color_params['sharpness'] - 50) / 50.0  # -1 to 1
        
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
        temperature = (self.color_params['temperature'] - 50) / 50.0  # -1 to 1
        
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

    def _capture_loop(self) -> None:
        """Continuously capture frames and publish the latest snapshot."""
        consecutive_failures = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures % 50 == 0:
                        print("[CAPTURE] Warning: failed to read frame repeatedly")
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                # Apply color correction once so UI + detector see same image
                frame = self._apply_color_correction(frame)

                now = time.monotonic()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_ts = now
                    self._latest_frame_id += 1
                    self._latest_frame_shape = frame.shape
            except Exception as e:
                print(f"[CAPTURE] Error: {e}")
                time.sleep(0.05)

    def _detector_loop(self) -> None:
        """Send latest frame snapshots to server at a fixed rate; publish latest bbox."""
        next_time = time.monotonic()
        while self.running:
            now = time.monotonic()
            sleep_s = next_time - now
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.05))
                continue

            # If we're running behind, resync to avoid tight loops
            if now - next_time > self.frame_interval * 2:
                next_time = now

            # Snapshot prompt + prompt_id
            with self.prompt_lock:
                prompt = self.current_prompt
                prompt_id = self._prompt_id

            if not prompt:
                next_time = now + self.frame_interval
                time.sleep(0.05)
                continue

            # Snapshot latest frame
            with self._frame_lock:
                frame_ref = self._latest_frame
                frame_id = self._latest_frame_id

            if frame_ref is None:
                next_time = now + self.frame_interval
                time.sleep(0.01)
                continue

            frame_copy = frame_ref.copy()

            bbox = self._send_frame_to_server(frame_copy, prompt)

            # Drop stale responses if prompt changed while request was in flight
            with self.prompt_lock:
                if prompt_id != self._prompt_id:
                    next_time = time.monotonic() + self.frame_interval
                    continue

            if bbox:
                t_bbox = time.monotonic()
                with self._bbox_lock:
                    self._latest_bbox = bbox
                    self._latest_bbox_ts = t_bbox
                    self._bbox_frame_id = frame_id
                # Back-compat for existing drawing path
                self.current_bbox = bbox

            # Backoff a bit when server is offline to reduce spam
            if not self.server_online:
                next_time = time.monotonic() + max(self.frame_interval, self._detector_offline_backoff_s)
            else:
                next_time = time.monotonic() + self.frame_interval

    def _control_loop(self) -> None:
        """Fixed-rate control loop for PID/servo updates based on latest bbox."""
        dt = 1.0 / max(self.control_hz, 1.0)
        next_time = time.monotonic()
        while self.running:
            now = time.monotonic()
            sleep_s = next_time - now
            if sleep_s > 0:
                time.sleep(min(sleep_s, dt))
                continue

            # Snapshot bbox + timestamp
            with self._bbox_lock:
                bbox = self._latest_bbox
                bbox_ts = self._latest_bbox_ts

            if bbox is None or (now - bbox_ts) > self.bbox_ttl_s:
                next_time = now + dt
                continue

            # Snapshot frame shape for coordinate conversion
            with self._frame_lock:
                frame_shape = self._latest_frame_shape

            if frame_shape is None:
                next_time = now + dt
                continue

            try:
                object_center = self._calculate_bbox_center(bbox, frame_shape)
                self._update_servo_positions(object_center)
            except Exception as e:
                logger.error(f"Control loop error: {e}")

            next_time = now + dt
    
    def run(self):
        """Main loop for tracking client."""
        print("\n[CLIENT] Starting main loop...\n")

        # Start pipeline threads (capture/detect/control). Keep UI in main thread.
        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
        if self.detector_thread is None or not self.detector_thread.is_alive():
            self.detector_thread = threading.Thread(target=self._detector_loop, daemon=True)
            self.detector_thread.start()
        if self.control_thread is None or not self.control_thread.is_alive():
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

        last_fps_time = time.monotonic()
        try:
            while self.running:
                # Snapshot latest frame reference
                with self._frame_lock:
                    frame_ref = self._latest_frame

                if frame_ref is None:
                    time.sleep(0.005)
                    continue

                display_frame = frame_ref.copy()

                # Calculate display FPS (render cadence)
                now = time.monotonic()
                dt = now - last_fps_time
                if dt > 0:
                    self.fps_history.append(1.0 / dt)
                last_fps_time = now

                # Draw bbox if available and not stale
                with self._bbox_lock:
                    bbox = self._latest_bbox
                    bbox_ts = self._latest_bbox_ts
                if bbox is not None and (now - bbox_ts) <= self.bbox_ttl_s:
                    display_frame = self._draw_bbox_on_frame(display_frame, bbox)
                
                # Draw info overlay
                display_frame = self._draw_info_overlay(display_frame)
                
                # Display frame
                cv2.imshow('Hand Tracking Client', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[CLIENT] Quit requested")
                    break
                elif key == ord('c'):
                    with self.prompt_lock:
                        self.current_prompt = ""
                        self._prompt_id += 1
                        self.current_bbox = None
                    with self._bbox_lock:
                        self._latest_bbox = None
                    print("\n[CLIENT] Prompt cleared")
                elif key == ord('s'):
                    self._show_server_stats()
        
        except KeyboardInterrupt:
            print("\n[CLIENT] Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[CLEANUP] Releasing resources...")
        self.running = False

        # Best-effort joins (threads are daemons, but we try to stop cleanly)
        for t in (self.capture_thread, self.detector_thread, self.control_thread):
            try:
                if t is not None and t.is_alive():
                    t.join(timeout=1.0)
            except Exception:
                pass

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if not self.mock_servos and hasattr(self, 'pan_servo'):
            self.pan_servo.disconnect()
            self.tilt_servo.disconnect()
        
        print("Cleanup complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MiniCPM-V Hand Tracking Client")
    parser.add_argument(
        "--server", 
        type=str, 
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=1,
        help="Camera device ID (default: 1 for robot camera, 0 for laptop webcam)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port for servos (Windows: COM3, Linux: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--pan-id",
        type=int,
        default=1,
        help="Servo ID for pan joint (default: 1)"
    )
    parser.add_argument(
        "--tilt-id",
        type=int,
        default=2,
        help="Servo ID for tilt joint (default: 2)"
    )
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
    parser.add_argument(
        "--voice",
        action="store_true",
        default=True,
        help="Enable voice control with Whisper (default: True)"
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
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--mock-servos",
        action="store_true",
        help="Run without servo hardware (for testing)"
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default="calibration.json",
        help="Path to servo calibration file (default: calibration.json)"
    )
    
    args = parser.parse_args()
    
    try:
        client = HandTrackingClient(
            server_url=args.server,
            camera_id=args.camera,
            pan_servo_id=args.pan_id,
            tilt_servo_id=args.tilt_id,
            port=args.port,
            target_fps=args.fps,
            jpeg_quality=args.quality,
            image_max_size=args.image_size,
            enable_voice=args.voice,
            whisper_model=args.whisper_model,
            mock_servos=args.mock_servos,
            calibration_file=args.calibration_file
        )
        client.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
