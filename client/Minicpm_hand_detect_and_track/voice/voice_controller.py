"""
Voice control using VAD (Voice Activity Detection) and Whisper.

Wake word "Atlas" activates recording; a 3-second silence ends the command.
States: idle → processing_wake → active → processing_cmd → idle
"""
import time
import logging
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class VoiceController:
    """Voice control system using VAD + Whisper with wake word support."""

    def __init__(
        self,
        whisper_model: str = "base",
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_threshold: int = 20,
        min_speech_chunks: int = 10,
        language: str = "en",
        wake_word: str = "atlas",
        command_silence_threshold: int = 100,
        max_active_chunks: int = 1000,
        on_command_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize voice controller.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            sample_rate: Audio sample rate (16000 Hz required by VAD)
            frame_duration_ms: VAD frame duration (10, 20, or 30 ms)
            vad_aggressiveness: VAD sensitivity (0-3, 2=moderate)
            silence_threshold: Silent chunks before wake-word processing (IDLE phase)
            min_speech_chunks: Minimum speech chunks to run Whisper in IDLE phase
            language: Language for Whisper transcription
            wake_word: Activation word (lowercase); detected in first 3 transcribed words
            command_silence_threshold: Silent chunks before command processing (ACTIVE phase)
            max_active_chunks: Hard cap on ACTIVE buffer size to prevent unbounded growth
            on_command_callback: Called with the transcribed command string
        """
        self.whisper_model_name = whisper_model
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_threshold = silence_threshold
        self.min_speech_chunks = min_speech_chunks
        self.language = language
        self.wake_word = wake_word.lower()
        self.command_silence_threshold = command_silence_threshold
        self.max_active_chunks = max_active_chunks
        self.on_command_callback = on_command_callback

        # Public state (read by UI thread via property)
        self._voice_state: str = "idle"   # "idle" | "active" | "processing"
        self._state_lock = threading.Lock()

        # Internal VAD buffering flag (only used inside audio_callback thread)
        self._is_buffering: bool = False

        # Audio accumulation
        self.audio_buffer: list = []
        self.silence_chunks: int = 0
        self.speech_chunks: int = 0

        # Whisper / VAD handles (set in start())
        self.running: bool = False
        self.whisper: Optional[WhisperModel] = None
        self.vad: Optional[webrtcvad.Vad] = None
        self.voice_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def voice_state(self) -> str:
        """Current voice state: 'idle' | 'active' | 'processing'."""
        return self._voice_state

    def start(self) -> bool:
        """Load models and start the background audio thread."""
        try:
            print("\n[VOICE] Initializing Whisper...")
            self.whisper = WhisperModel(
                self.whisper_model_name,
                device="cpu",
                compute_type="int8",
            )
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)

            self.running = True
            self.voice_thread = threading.Thread(
                target=self._voice_input_loop, daemon=True
            )
            self.voice_thread.start()
            print(f"[VOICE] Ready — say '{self.wake_word.capitalize()}' to activate")
            return True
        except Exception as e:
            print(f"[VOICE] Failed to initialize: {e}")
            print("[VOICE] Continuing with text input only...")
            return False

    def stop(self) -> None:
        """Stop the audio thread."""
        self.running = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_buffer(self) -> None:
        """Clear all VAD accumulation state. Called only from the audio thread."""
        self._is_buffering = False
        self.audio_buffer = []
        self.silence_chunks = 0
        self.speech_chunks = 0

    def _trigger_processing(self, phase: str) -> None:
        """
        Decide whether to send the current buffer to Whisper and transition state.

        phase: "wake_word" (IDLE) or "command" (ACTIVE)

        Called exclusively from audio_callback (sounddevice thread).
        """
        has_enough_speech = (
            phase == "command"                             # ACTIVE: Atlas confirmed, always send
            or self.speech_chunks >= self.min_speech_chunks  # IDLE: filter short noises
        )

        if has_enough_speech and len(self.audio_buffer) > 0:
            audio_to_process = np.concatenate(self.audio_buffer)
            with self._state_lock:
                self._voice_state = "processing"
            threading.Thread(
                target=self._process_voice_buffer,
                args=(audio_to_process, phase),
                daemon=True,
            ).start()
        else:
            # Not enough speech (IDLE noise) or empty buffer (ACTIVE timeout without speech)
            with self._state_lock:
                self._voice_state = "idle"

        self._reset_buffer()

    # ------------------------------------------------------------------
    # Audio input loop (runs in self.voice_thread)
    # ------------------------------------------------------------------

    def _voice_input_loop(self) -> None:
        """Background thread: open microphone stream and process frames via VAD."""

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio status: {status}")

            # Read current state atomically
            with self._state_lock:
                state = self._voice_state

            # While Whisper is running, drop incoming audio entirely
            if state == "processing":
                return

            # Hard cap: force-process if ACTIVE buffer is too large
            if state == "active" and len(self.audio_buffer) >= self.max_active_chunks:
                self._trigger_processing("command")
                return

            # Choose silence threshold based on current phase
            threshold = (
                self.command_silence_threshold  # 3 s in ACTIVE
                if state == "active"
                else self.silence_threshold     # 600 ms in IDLE
            )

            # Convert to int16 for webrtcvad
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            if len(audio_int16) != self.frame_size:
                return

            try:
                is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
            except Exception as e:
                logger.error(f"VAD processing error: {e}")
                return

            if is_speech:
                self.speech_chunks += 1
                self.silence_chunks = 0
                if not self._is_buffering:
                    self._is_buffering = True
                self.audio_buffer.append(audio_int16)

            else:  # Silence frame
                if self._is_buffering:
                    # Normal case: silence after/between speech — keep buffering
                    self.silence_chunks += 1
                    self.audio_buffer.append(audio_int16)
                elif state == "active":
                    # Pre-speech ACTIVE: count silence for timeout but don't fill buffer
                    self.silence_chunks += 1

            # Fire processing when enough silence has accumulated
            if self.silence_chunks >= threshold and (
                self._is_buffering or state == "active"
            ):
                phase = "command" if state == "active" else "wake_word"
                self._trigger_processing(phase)

        try:
            try:
                default_device = sd.query_devices(kind="input")
                print(f"[VOICE] Using microphone: {default_device['name']}")
            except Exception:
                pass

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                callback=audio_callback,
            ):
                print("[VOICE] Listening for wake word...")
                while self.running:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            print("[VOICE] Voice control disabled. Text input still available.")

    # ------------------------------------------------------------------
    # Whisper processing (runs in a daemon thread per utterance)
    # ------------------------------------------------------------------

    def _process_voice_buffer(self, audio_data: np.ndarray, phase: str) -> None:
        """Transcribe buffered audio and dispatch to the appropriate handler."""
        try:
            print("[VOICE] Transcribing...")
            start_time = time.time()

            audio_float32 = audio_data.astype(np.float32) / 32767.0

            segments, _info = self.whisper.transcribe(
                audio_float32,
                language=self.language if self.language != "auto" else None,
                beam_size=5,
                vad_filter=(phase != "wake_word"),  # disable for short wake-word clips
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            transcription = " ".join(s.text.strip() for s in segments).strip()
            elapsed = time.time() - start_time
            print(f"[VOICE] Transcribed ({phase}): '{transcription}' ({elapsed:.2f}s)")

            if phase == "wake_word":
                self._handle_wake_word_check(transcription)
            else:
                self._handle_command(transcription)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
        finally:
            # Guarantee state recovery if a handler raised or transcription was empty
            with self._state_lock:
                if self._voice_state == "processing":
                    self._voice_state = "idle"

    def _handle_wake_word_check(self, transcription: str) -> None:
        """Check whether the wake word appears in the first 3 transcribed words."""
        text_lower = transcription.lower()
        words = text_lower.split()
        wake_found = any(w.strip(".,!?-") == self.wake_word for w in words[:3])

        if not wake_found:
            with self._state_lock:
                self._voice_state = "idle"
            return

        # Extract any command that followed the wake word in the same utterance
        idx = text_lower.find(self.wake_word)
        after = transcription[idx + len(self.wake_word):]
        command = after.lstrip(" ,.:!?-–—").rstrip(" .!?")

        if command:
            print(f"[VOICE] Atlas + command: '{command}'")
            with self._state_lock:
                self._voice_state = "idle"
            if self.on_command_callback:
                self.on_command_callback(command)
        else:
            print("[VOICE] Atlas detected — waiting for command...")
            with self._state_lock:
                self._voice_state = "active"

    def _handle_command(self, transcription: str) -> None:
        """Pass the transcribed command to the callback and return to idle."""
        with self._state_lock:
            self._voice_state = "idle"
        if transcription and self.on_command_callback:
            self.on_command_callback(transcription)
