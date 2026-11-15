"""
Processing worker.

Consumes frames sequentially from a bounded queue (no_drop),
updates tracker/servos/metrics, and publishes the latest DisplayPacket to UI.
"""

from __future__ import annotations

import threading
import time
import logging
from queue import Queue, Empty, Full
from typing import Optional

from pipeline.messages import FramePacket, LLMTask, LLMResult, DisplayPacket
from tracking.hybrid_tracker import HybridTracker
from servo.servo_controller import ServoController
from utils.metrics import PerformanceMetrics
from utils.coordinates import calculate_bbox_center

logger = logging.getLogger(__name__)


def _put_latest(q: Queue, item) -> None:
    """Put to a size-1 queue, overwriting existing value if full."""
    try:
        q.put(item, block=False)
        return
    except Full:
        try:
            q.get(block=False)
        except Empty:
            pass
        q.put(item, block=False)


def _drain_latest(q: Queue):
    """Non-blocking drain; return the most recent item or None."""
    latest = None
    while True:
        try:
            latest = q.get(block=False)
        except Empty:
            return latest


class ProcessingWorker(threading.Thread):
    def __init__(
        self,
        *,
        stop_event: threading.Event,
        frame_queue: Queue,
        ui_queue: Queue,
        llm_task_queue: Queue,
        llm_result_queue: Queue,
        llm_slot: threading.BoundedSemaphore,
        prompt_lock: threading.Lock,
        get_prompt: callable,
        hybrid_tracker: HybridTracker,
        servo_controller: ServoController,
        metrics: PerformanceMetrics,
        server_client_online_getter: callable,
        voice_state_getter: callable,
    ):
        super().__init__(daemon=True)
        self._stop_event = stop_event
        self._frame_queue = frame_queue
        self._ui_queue = ui_queue
        self._llm_task_queue = llm_task_queue
        self._llm_result_queue = llm_result_queue
        self._llm_slot = llm_slot

        self._prompt_lock = prompt_lock
        self._get_prompt = get_prompt

        self._hybrid_tracker = hybrid_tracker
        self._servo_controller = servo_controller
        self._metrics = metrics

        self._server_client_online_getter = server_client_online_getter
        self._voice_state_getter = voice_state_getter

        self._last_proc_ts: Optional[float] = None
        self._last_bbox_ts: Optional[float] = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pkt: FramePacket = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            now = time.time()
            if self._last_proc_ts is not None:
                dt = now - self._last_proc_ts
                if dt > 0:
                    self._metrics.add_fps_sample(1.0 / dt)
            self._last_proc_ts = now

            frame = pkt.frame
            image_center = (frame.shape[1] // 2, frame.shape[0] // 2)

            # Update servo center to actual frame size (camera may ignore requested resolution)
            self._servo_controller.image_center = image_center

            with self._prompt_lock:
                prompt = self._get_prompt()

            # Let tracker react to prompt changes (reset on change).
            self._hybrid_tracker.set_prompt(prompt)

            # Apply latest LLM result (if any)
            llm_result: Optional[LLMResult] = _drain_latest(self._llm_result_queue)
            if llm_result is not None:
                # Only apply result for current prompt
                if prompt and llm_result.prompt == prompt:
                    init_frame = llm_result.frame if llm_result.frame is not None else frame
                    applied_bbox = self._hybrid_tracker.apply_llm_result(init_frame, llm_result, now)
                    if applied_bbox is not None:
                        self._last_bbox_ts = now
                        self._metrics.add_inference_time(llm_result.inference_time)
                        self._metrics.add_network_time(llm_result.network_time)

            bbox = None
            used_tracker = False

            server_online = bool(self._server_client_online_getter())

            if prompt and server_online:
                # Tracker step
                bbox = self._hybrid_tracker.update_tracker(frame)
                used_tracker = bbox is not None
                if bbox is not None:
                    self._last_bbox_ts = now

                # Decide if we should request LLM (non-blocking)
                if self._hybrid_tracker.should_request_llm(now):
                    acquired = self._llm_slot.acquire(blocking=False)
                    if acquired:
                        task = LLMTask(frame=frame, prompt=prompt, ts=pkt.ts, frame_id=pkt.frame_id)
                        try:
                            self._llm_task_queue.put(task, block=False)
                            self._hybrid_tracker.mark_llm_requested(now)
                        except Full:
                            # Should not happen often; release slot if enqueue failed.
                            self._llm_slot.release()
            elif prompt and not server_online:
                # Tracker-only mode (if tracker active)
                bbox = self._hybrid_tracker.update_tracker(frame)
                used_tracker = bbox is not None
                if bbox is not None:
                    self._last_bbox_ts = now

            # Servo update
            if bbox is not None:
                cx, cy = calculate_bbox_center(bbox, frame.shape)
                self._servo_controller.update_position((cx, cy))

            pan_pos, tilt_pos = self._servo_controller.get_positions()
            bbox_age_ms = None
            if self._last_bbox_ts is not None:
                bbox_age_ms = (now - self._last_bbox_ts) * 1000.0

            display = DisplayPacket(
                frame=frame,
                bbox=bbox,
                prompt=prompt,
                avg_fps=self._metrics.get_average_fps(),
                avg_inference=self._metrics.get_average_inference_time(),
                avg_network=self._metrics.get_average_network_time(),
                is_tracking=self._hybrid_tracker.is_tracking(),
                tracker_type=self._hybrid_tracker.get_tracker_type(),
                time_to_next_check=self._hybrid_tracker.get_time_to_next_llm_check(now),
                pan_position=pan_pos,
                tilt_position=tilt_pos,
                pan_output=self._servo_controller.last_pan_output,
                tilt_output=self._servo_controller.last_tilt_output,
                voice_state=str(self._voice_state_getter()),
                server_online=server_online,
                image_center=image_center,
                bbox_age_ms=bbox_age_ms,
            )
            _put_latest(self._ui_queue, display)

