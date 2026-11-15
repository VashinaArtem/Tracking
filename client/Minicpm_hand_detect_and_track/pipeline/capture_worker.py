"""
Camera capture worker.

Owns cv2.VideoCapture and performs lightweight per-frame preprocessing
(undistortion + color correction) before pushing frames to a bounded queue.
"""

from __future__ import annotations

import threading
import time
import logging
from queue import Queue
from typing import Optional

import cv2

from config.color_params import ColorCorrectionParams
from image.camera_calibration import CameraCalibrator
from image.color_correction import ColorCorrector
from pipeline.messages import FramePacket

logger = logging.getLogger(__name__)


class CameraCaptureWorker(threading.Thread):
    def __init__(
        self,
        *,
        stop_event: threading.Event,
        frame_queue: Queue,
        camera_id: int,
        camera_width: int,
        camera_height: int,
        camera_calibration_file: Optional[str],
    ):
        super().__init__(daemon=True)
        self._stop_event = stop_event
        self._frame_queue = frame_queue

        self._camera_id = camera_id
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_calibration_file = camera_calibration_file

        self._cap: Optional[cv2.VideoCapture] = None

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
        return cap

    def run(self) -> None:
        frame_id = 0
        try:
            self._cap = self._open_camera()

            # Init calibration/color correction in the same thread as capture (cap owner).
            camera_calibrator = None
            if self._camera_calibration_file:
                camera_calibrator = CameraCalibrator(self._camera_calibration_file)
                if not camera_calibrator.is_enabled():
                    logger.warning("Camera calibration failed to load; disabling undistortion")
                    camera_calibrator = None

            color_params = ColorCorrectionParams()
            color_corrector = ColorCorrector(color_params.to_dict(), self._cap)

            while not self._stop_event.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    time.sleep(0.01)
                    continue

                # Preprocess
                if camera_calibrator:
                    frame = camera_calibrator.undistort(frame)
                frame = color_corrector.apply(frame)

                pkt = FramePacket(frame=frame, ts=time.time(), frame_id=frame_id)

                # no_drop + backpressure: bounded queue with blocking put()
                self._frame_queue.put(pkt, block=True)
                frame_id += 1

        except Exception as e:
            logger.exception("CameraCaptureWorker crashed: %s", e)
        finally:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass

