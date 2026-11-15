"""
MiniCPM-V Hand Tracking Client (Modular Version)

Integrated system combining voice-controlled object detection with servo tracking.

Features:
- Voice control via Whisper for object prompts
- MiniCPM-V server for object detection
- Pan/tilt servo control with PID tracking
- Real-time video display with bounding boxes
- Hybrid tracking (LLM + classical tracker)
"""
import time
import threading
import logging
from queue import Queue, Empty
from typing import Optional
import cv2

from config.settings import TrackingConfig, ServoConfig, VoiceConfig
from network.server_client import ServerClient
from servo.servo_controller import ServoController
from tracking.hybrid_tracker import HybridTracker
from voice.voice_controller import VoiceController
from ui.visualization import Visualizer
from utils.metrics import PerformanceMetrics

from pipeline.capture_worker import CameraCaptureWorker
from pipeline.processing_worker import ProcessingWorker
from pipeline.llm_worker import LLMWorker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandTrackingClient:
    """Main orchestrator for the hand tracking system."""
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize the hand tracking client.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        if config is None:
            config = TrackingConfig()
        
        self.config = config
        self.running = True
        self.stop_event = threading.Event()
        
        print("=" * 60)
        print("MiniCPM-V Hand Tracking Client (Modular)")
        print("=" * 60)
        
        # Current state
        self.current_prompt = ""
        self.prompt_lock = threading.Lock()
        
        # Initialize metrics
        self.metrics = PerformanceMetrics()
        
        # Initialize server client
        print(f"\n[1/5] Initializing server client...")
        print(f"Server URL: {config.server_url}")
        self.server_client = ServerClient(
            server_url=config.server_url,
            jpeg_quality=config.jpeg_quality,
            image_max_size=config.image_max_size
        )
        
        if not self.server_client.check_health():
            print(f"WARNING: Cannot connect to server")
            print(f"Error: {self.server_client.last_error}")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Server not accessible")
        else:
            print("Server is online and ready")

        # Image center placeholder (will be corrected from real frame size in ProcessingWorker)
        self.image_center = (config.camera_width // 2, config.camera_height // 2)
        
        # Initialize servo controller
        print(f"\n[2/5] Initializing servo controller...")
        servo_config = ServoConfig()
        self.servo_controller = ServoController(
            pan_servo_id=config.pan_servo_id,
            tilt_servo_id=config.tilt_servo_id,
            port=config.port,
            calibration_file=config.calibration_file,
            image_center=self.image_center,
            pan_pid_config=vars(servo_config.pan_pid),
            tilt_pid_config=vars(servo_config.tilt_pid),
            max_step_size=servo_config.max_step_size,
            mock_servos=config.mock_servos
        )
        
        # Initialize hybrid tracker
        print(f"\n[3/5] Initializing hybrid tracker...")
        self.hybrid_tracker = HybridTracker(
            server_client=self.server_client,
            tracker_type=config.tracker_type,
            llm_check_interval=config.llm_check_interval
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(clean_ui=config.clean_ui)
        
        # Initialize voice controller
        self.voice_controller = None
        if config.enable_voice:
            voice_config = VoiceConfig()
            self.voice_controller = VoiceController(
                whisper_model=config.whisper_model,
                sample_rate=voice_config.sample_rate,
                frame_duration_ms=voice_config.frame_duration_ms,
                vad_aggressiveness=voice_config.vad_aggressiveness,
                silence_threshold=voice_config.silence_threshold,
                min_speech_chunks=voice_config.min_speech_chunks,
                language=voice_config.language,
                wake_word=voice_config.wake_word,
                command_silence_threshold=voice_config.command_silence_threshold,
                max_active_chunks=voice_config.max_active_chunks,
                on_command_callback=self._handle_voice_command,
            )
            self.voice_controller.start()
        
        # Start console prompt thread
        print("\n[SETUP] Starting prompt input thread...")
        self.prompt_thread = threading.Thread(target=self._prompt_input_loop, daemon=True)
        self.prompt_thread.start()

        # Queues and workers
        self.frame_queue: Queue = Queue(maxsize=config.frame_queue_size)
        self.ui_queue: Queue = Queue(maxsize=config.ui_queue_size)
        self.llm_task_queue: Queue = Queue(maxsize=config.llm_queue_size)
        self.llm_result_queue: Queue = Queue(maxsize=config.llm_queue_size)
        self.llm_slot = threading.BoundedSemaphore(1)

        print("\n[4/5] Starting pipeline workers...")
        self.capture_worker = CameraCaptureWorker(
            stop_event=self.stop_event,
            frame_queue=self.frame_queue,
            camera_id=config.camera_id,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            camera_calibration_file=config.camera_calibration_file,
        )
        self.processing_worker = ProcessingWorker(
            stop_event=self.stop_event,
            frame_queue=self.frame_queue,
            ui_queue=self.ui_queue,
            llm_task_queue=self.llm_task_queue,
            llm_result_queue=self.llm_result_queue,
            llm_slot=self.llm_slot,
            prompt_lock=self.prompt_lock,
            get_prompt=lambda: self.current_prompt,
            hybrid_tracker=self.hybrid_tracker,
            servo_controller=self.servo_controller,
            metrics=self.metrics,
            server_client_online_getter=lambda: self.server_client.online,
            voice_state_getter=lambda: (self.voice_controller.voice_state if self.voice_controller else "idle"),
        )
        self.llm_worker = LLMWorker(
            stop_event=self.stop_event,
            server_client=self.server_client,
            task_queue=self.llm_task_queue,
            result_queue=self.llm_result_queue,
            llm_slot=self.llm_slot,
        )
        self.capture_worker.start()
        self.processing_worker.start()
        self.llm_worker.start()

        print("\n[5/5] Pipeline is running.")
        
        print("\nReady!")
        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        if config.enable_voice:
            print("- VOICE: Just speak naturally! Say 'clear' or 'quit'")
        print("- TEXT: Type your prompt in the console and press Enter")
        print("- Example: 'hand', 'person', 'cup', 'phone'")
        print("- Press 'q' in the video window to quit")
        print("- Press 'c' in the video window to clear current prompt")
        print("- Press 's' to show server statistics")
        print("=" * 60 + "\n")
    
    def _handle_voice_command(self, transcription: str):
        """Handle voice command from voice controller."""
        transcription_lower = transcription.lower()
        
        if "clear" in transcription_lower or "stop" in transcription_lower:
            with self.prompt_lock:
                self.current_prompt = ""
            print(f"[VOICE] ✓ Tracking stopped")
        
        elif "quit" in transcription_lower or "exit" in transcription_lower:
            print(f"[VOICE] Quit command received")
            self.running = False
        
        else:
            # Update tracking prompt
            with self.prompt_lock:
                self.current_prompt = transcription
    
    def _prompt_input_loop(self):
        """Background thread for receiving prompt input from console."""
        print("\n[CONSOLE] Ready for prompt input. Type and press Enter:\n")
        while self.running:
            try:
                user_input = input("> ").strip()
                
                with self.prompt_lock:
                    self.current_prompt = user_input
                    if user_input:
                        print(f"[CONSOLE] Prompt updated: '{user_input}'")
                    else:
                        print(f"[CONSOLE] ✓ Tracking stopped")
            except EOFError:
                break
            except Exception as e:
                print(f"[CONSOLE ERROR] {e}")
    
    def run(self):
        """UI loop for tracking client (imshow/waitKey in main thread)."""
        print("\n[CLIENT] Starting UI loop...\n")

        last_display = None
        ui_timeout_s = max(1, int(self.config.ui_poll_timeout_ms)) / 1000.0

        try:
            while self.running:
                current_time = time.time()

                try:
                    last_display = self.ui_queue.get(timeout=ui_timeout_s)
                except Empty:
                    pass

                if last_display is not None:
                    prompt = last_display.prompt
                    bbox = last_display.bbox

                    display_frame = last_display.frame.copy()
                    if bbox:
                        display_frame = self.visualizer.draw_bbox(display_frame, bbox, label=prompt)

                    display_frame = self.visualizer.draw_info_overlay(
                        frame=display_frame,
                        prompt=prompt,
                        avg_fps=last_display.avg_fps,
                        avg_inference=last_display.avg_inference,
                        avg_network=last_display.avg_network,
                        is_tracking=last_display.is_tracking,
                        tracker_type=last_display.tracker_type,
                        time_to_next_check=last_display.time_to_next_check,
                        pan_position=last_display.pan_position,
                        tilt_position=last_display.tilt_position,
                        pan_output=last_display.pan_output,
                        tilt_output=last_display.tilt_output,
                        voice_state=last_display.voice_state,
                        server_online=last_display.server_online,
                        image_center=last_display.image_center,
                        bbox_age_ms=last_display.bbox_age_ms,
                    )

                    cv2.imshow('Hand Tracking Client', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[CLIENT] Quit requested")
                    break
                elif key == ord('c'):
                    with self.prompt_lock:
                        self.current_prompt = ""
                    print("\n[CLIENT] Prompt cleared")
                elif key == ord('s'):
                    stats = self.server_client.get_stats()
                    if stats:
                        self.visualizer.display_server_stats(stats)
        
        except KeyboardInterrupt:
            print("\n[CLIENT] Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[CLEANUP] Releasing resources...")
        self.running = False
        self.stop_event.set()
        
        if self.voice_controller:
            self.voice_controller.stop()

        # Join workers
        for t in (getattr(self, "capture_worker", None),
                  getattr(self, "processing_worker", None),
                  getattr(self, "llm_worker", None)):
            try:
                if t is not None:
                    t.join(timeout=1.0)
            except Exception:
                pass
        
        cv2.destroyAllWindows()
        
        self.servo_controller.disconnect()
        
        print("Cleanup complete")
