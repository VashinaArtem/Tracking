"""
MiniCPM-V-4.5 Tracking Client
Client-side video capture and display for real-time object tracking

This client captures video from webcam, sends frames to the tracking server,
and displays bounding boxes in real-time.
"""

import io
import time
import threading
from collections import deque
from typing import Optional, List
import cv2
import numpy as np
import requests
from PIL import Image

import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


class TrackingClient:
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        camera_id: int = 0,
        target_fps: float = 1.5,
        jpeg_quality: int = 85,
        image_max_size: int = 512,
        enable_voice=True,
        whisper_model="base"
    ):
        """
        Initialize the tracking client.
        
        Args:
            server_url: URL of the tracking server (e.g., http://192.168.1.100:8000)
            camera_id: Camera device ID (usually 0 for default webcam)
            target_fps: Target FPS for sending frames to server (1-2 recommended)
            jpeg_quality: JPEG compression quality (0-100, higher = better quality but larger size)
            image_max_size: Max image dimension in pixels (smaller = fewer tokens = faster inference)
            enable_voice: Enable voice control with Whisper
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        print("=" * 60)
        print("MiniCPM-V-4.5 Tracking Client")
        print("=" * 60)
        
        self.server_url = server_url.rstrip('/')
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_request_time = 0
        self.jpeg_quality = jpeg_quality
        self.image_max_size = image_max_size
        
        # Control flag (must be set early for threads)
        self.running = True
        
        # Current prompt and detection results
        self.current_prompt = ""
        self.current_bbox = None
        self.last_inference_time = 0.0
        self.prompt_lock = threading.Lock()
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.inference_times = deque(maxlen=10)
        self.network_times = deque(maxlen=10)
        
        # Connection status
        self.server_online = False
        self.last_error = ""

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
            print("\n[VOICE] Initializing Whisper...")
            try:
                self.whisper = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
                self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3 (2 = moderate)
                
                self.voice_thread = threading.Thread(target=self._voice_input_loop, daemon=True)
                self.voice_thread.start()
                print("[VOICE] ✓ Voice control initialized")
            except Exception as e:
                print(f"[VOICE] ✗ Failed to initialize voice control: {e}")
                print("[VOICE] Continuing with text input only...")
                self.enable_voice = False
        
        # Check server
        print(f"\n[1/3] Checking server connection...")
        print(f"Server URL: {self.server_url}")
        if not self._check_server():
            print(f"⚠ WARNING: Cannot connect to server at {self.server_url}")
            print(f"Error: {self.last_error}")
            print(f"Make sure the server is running and accessible")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Server not accessible")
        else:
            print("✓ Server is online and ready")
        
        # Initialize camera
        print(f"\n[2/3] Initializing camera (ID: {camera_id})...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera resolution to 720p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera initialized: {actual_width}x{actual_height}")
        print(f"✓ Image resize: max {self.image_max_size}px (reduces tokens for faster inference)")
        
        # Start prompt input thread
        print("\n[3/3] Starting prompt input thread...")
        self.prompt_thread = threading.Thread(target=self._prompt_input_loop, daemon=True)
        self.prompt_thread.start()
        print("✓ Ready!")
        
        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        if self.enable_voice:
            print("- VOICE: Just speak naturally! Say 'clear' or 'quit'")
        print("- TEXT: Type your prompt in the console and press Enter")
        print("- Example: 'person', 'red cup', 'laptop computer'")
        print("- Press 'q' in the video window to quit")
        print("- Press 'c' in the video window to clear current prompt")
        print("- Press 's' to show server statistics")
        print("=" * 60 + "\n")

    
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
                        self.current_bbox = None  # Reset bbox when prompt changes
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
                            print("[VOICE] 🎤 Speech detected, listening...")
                        
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
                print("[VOICE] 🎤 Listening... Speak naturally!")
                # Keep thread alive while running
                while self.running:
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"[VOICE ERROR] Failed to start audio stream: {e}")
            print("[VOICE] Voice control disabled. Text input still available.")
            print("\n[VOICE] Troubleshooting tips:")
            print("  - Check if microphone is connected")
            print("  - Run: python -c \"import sounddevice as sd; print(sd.query_devices())\"")
            print("  - Try a different microphone or audio device\n")
    
    def _process_voice_buffer(self, audio_data: np.ndarray, sample_rate: int):
        """Process buffered audio with Whisper (runs in separate thread)"""
        try:
            print("[VOICE] ⏳ Transcribing...")
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
                        self.current_bbox = None
                    print(f"[VOICE] ✓ Cleared prompt ({elapsed:.2f}s)")
                
                elif "quit" in transcription_lower or "exit" in transcription_lower:
                    print(f"[VOICE] ✓ Quit command received ({elapsed:.2f}s)")
                    self.running = False
                
                else:
                    # Update tracking prompt
                    with self.prompt_lock:
                        self.current_prompt = transcription
                        self.current_bbox = None  # Reset bbox when prompt changes
                    print(f"[VOICE] ✓ Transcribed: '{transcription}' ({elapsed:.2f}s)")
            else:
                print(f"[VOICE] ⚠ No speech detected ({elapsed:.2f}s)")
        
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
                timeout=120  # 60 second timeout (1 minute)
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
    
    def _draw_bbox_on_frame(self, frame: np.ndarray, bbox: List[int], 
                           color=(0, 0, 255), thickness=3) -> np.ndarray:
        """
        Draw bounding box on OpenCV frame.
        
        Args:
            frame: OpenCV frame (numpy array)
            bbox: [x1, y1, x2, y2] normalized to 0-1000
            color: BGR color tuple
            thickness: Line thickness
        """
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(
            self.current_prompt, 
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
            self.current_prompt, 
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        return frame
    
    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 180), (0, 0, 0), -1)
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
        
        # Voice status
        if self.enable_voice:
            voice_color = (0, 255, 255) if self.is_listening else (128, 128, 128)
            voice_text = "Voice: LISTENING" if self.is_listening else "Voice: Ready"
            cv2.putText(frame, voice_text, (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Server status
        status_color = (0, 255, 0) if self.server_online else (0, 0, 255)
        status_text = "Server: ONLINE" if self.server_online else "Server: OFFLINE"
        cv2.putText(frame, status_text, (w - 200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
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
    
    def run(self):
        """Main loop for tracking client."""
        print("\n[CLIENT] Starting main loop...\n")
        
        frame_count = 0
        last_fps_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Calculate display FPS
                time_delta = current_time - last_fps_time
                if time_delta > 0:
                    self.fps_history.append(1.0 / time_delta)
                last_fps_time = current_time
                
                # Send frame to server if enough time has passed
                with self.prompt_lock:
                    has_prompt = bool(self.current_prompt)
                    prompt = self.current_prompt
                
                if has_prompt and self.server_online and \
                   (current_time - self.last_request_time) >= self.frame_interval:
                    self.current_bbox = self._send_frame_to_server(frame, prompt)
                    self.last_request_time = current_time
                
                # Draw bbox if available
                display_frame = frame.copy()
                if self.current_bbox is not None:
                    display_frame = self._draw_bbox_on_frame(display_frame, self.current_bbox)
                
                # Draw info overlay
                display_frame = self._draw_info_overlay(display_frame)
                
                # Display frame
                cv2.imshow('MiniCPM-V Tracking Client', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[CLIENT] Quit requested")
                    break
                elif key == ord('c'):
                    with self.prompt_lock:
                        self.current_prompt = ""
                        self.current_bbox = None
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
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.5 Tracking Client")
    parser.add_argument(
        "--server", 
        type=str, 
        default="http://localhost:8000",
        help="Server URL (e.g., http://192.168.1.100:8000)"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera device ID (0 for default webcam)"
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
        help="Whisper model size (default: base, tiny=fastest, large=most accurate)"
    )
    
    args = parser.parse_args()
    
    try:
        client = TrackingClient(
            server_url=args.server,
            camera_id=args.camera,
            target_fps=args.fps,
            jpeg_quality=args.quality,
            image_max_size=args.image_size,
            enable_voice=args.voice,
            whisper_model=args.whisper_model
        )
        client.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
