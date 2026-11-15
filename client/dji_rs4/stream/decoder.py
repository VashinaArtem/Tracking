"""
H.264 decoder wrapping PyAV.

Usage
-----
    dec = Decoder()
    frames = dec.decode(raw_h264_bytes)   # list of numpy BGR arrays
    for frame in frames:
        cv2.imshow("preview", frame)

Notes
-----
* Uses the two-step PyAV API:
      av.CodecContext.parse(chunk) → [av.Packet, ...]
      av.CodecContext.decode(packet) → [av.VideoFrame, ...]
  parse() accumulates data internally and emits complete packets aligned on
  NAL unit boundaries, which is essential for a fragmented UDP stream.

* The first several calls will return an empty list — this is expected.
  The H.264 codec cannot produce frames until it has received SPS (NAL type 7)
  and PPS (NAL type 8) units, which arrive at the very start of the stream.

* Occasional malformed or non-video bytes from the device are handled by
  catching av.AVError so a single bad chunk does not crash the viewer loop.
"""

from typing import List

import av
import numpy as np


class Decoder:
    def __init__(self) -> None:
        self._ctx = av.CodecContext.create("h264", "r")

    def decode(self, raw_bytes: bytes) -> List[np.ndarray]:
        """
        Feed *raw_bytes* (a DUML payload chunk) into the H.264 decoder.

        Returns a list of BGR numpy arrays (one per decoded frame).
        Returns an empty list while the codec is still accumulating SPS/PPS,
        or when the chunk does not yet complete a full frame.
        """
        frames: List[np.ndarray] = []
        try:
            packets = self._ctx.parse(raw_bytes)
            for pkt in packets:
                for video_frame in self._ctx.decode(pkt):
                    bgr = video_frame.to_ndarray(format="bgr24")
                    frames.append(bgr)
        except av.AVError:
            pass  # malformed / non-video bytes — keep stream alive
        return frames
