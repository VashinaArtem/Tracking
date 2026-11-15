"""
OpenCV frame viewer — must be run on the main thread (macOS requirement).

Usage
-----
    q = queue.Queue(maxsize=256)
    viewer = Viewer(video_queue=q)
    viewer.run()          # blocks until user presses 'q'

Keyboard shortcuts
------------------
    q   quit
    s   save a PNG snapshot of the last displayed frame
"""

import queue
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from stream.decoder import Decoder

_WINDOW_TITLE   = "DJI RavenEye Stream"
_WAIT_MS        = 1      # cv2.waitKey delay — keeps GUI responsive
_SNAPSHOT_DIR   = Path("snapshots")


class Viewer:
    """
    Consumes encoded H.264 byte chunks from *video_queue*, decodes them
    with an internal :class:`~stream.decoder.Decoder`, and shows the
    resulting BGR frames via ``cv2.imshow``.

    Parameters
    ----------
    video_queue : bounded ``queue.Queue`` fed by the receiver thread.
    """

    def __init__(self, video_queue: queue.Queue) -> None:
        self.queue   = video_queue
        self._decoder = Decoder()

    def run(self) -> None:
        """
        Enter the display loop.  Returns when the user presses **q** or
        closes the window.  Always calls ``cv2.destroyAllWindows()`` on exit.
        """
        cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        last_frame: Optional[np.ndarray] = None
        snapshot_idx = 0
        print(f"[Viewer] Window open — press 'q' to quit, 's' to snapshot.")

        try:
            while True:
                # --- drain one encoded chunk from the queue ---------------
                try:
                    encoded = self.queue.get_nowait()
                    frames = self._decoder.decode(encoded)
                    for frame in frames:
                        last_frame = frame
                        cv2.imshow(_WINDOW_TITLE, frame)
                except queue.Empty:
                    pass

                # --- process GUI events (must be called every iteration) --
                key = cv2.waitKey(_WAIT_MS) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("s"):
                    if last_frame is not None:
                        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                        filename = _SNAPSHOT_DIR / f"snapshot_{snapshot_idx:04d}.png"
                        cv2.imwrite(str(filename), last_frame)
                        print(f"[Viewer] Snapshot saved → {filename}")
                        snapshot_idx += 1
                    else:
                        print("[Viewer] No frame yet to snapshot.")

                # Exit if the window was closed via the OS close button.
                if cv2.getWindowProperty(_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            cv2.destroyAllWindows()
