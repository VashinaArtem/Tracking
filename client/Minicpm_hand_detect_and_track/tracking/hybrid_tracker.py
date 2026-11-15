"""
Hybrid tracking system combining LLM detection with classical tracking.

Uses LLM (MiniCPM-V) for periodic verification and OpenCV tracker for fast tracking.
"""
import time
import logging
from typing import Optional, List
import numpy as np

from tracking.tracker_manager import TrackerManager
from network.server_client import ServerClient
from pipeline.messages import LLMResult

logger = logging.getLogger(__name__)


class HybridTracker:
    """
    Hybrid tracking system using LLM + classical tracker.
    
    Strategy:
    - Use LLM detection periodically for accuracy (every N seconds)
    - Use classical tracker (KCF/CSRT) between LLM checks for speed
    - Automatically reinitialize tracker when LLM finds object
    """
    
    def __init__(
        self,
        server_client: ServerClient,
        tracker_type: str = "KCF",
        llm_check_interval: float = 2.0
    ):
        """
        Initialize hybrid tracker.
        
        Args:
            server_client: Server client for LLM detection
            tracker_type: Type of OpenCV tracker ("KCF" or "CSRT")
            llm_check_interval: Seconds between LLM re-verification checks
        """
        self.server_client = server_client
        self.tracker_manager = TrackerManager(tracker_type)
        self.llm_check_interval = llm_check_interval
        
        # Async mode: we track request timestamps (not blocking on detect).
        self.last_llm_request = 0.0
        self.current_prompt = ""
    
    def _on_prompt(self, prompt: str) -> None:
        if prompt != self.current_prompt:
            self.current_prompt = prompt
            self.tracker_manager.reset()
            self.last_llm_request = 0.0  # Force LLM request on prompt change

    def set_prompt(self, prompt: str) -> None:
        """Update prompt and reset state if changed."""
        self._on_prompt(prompt)

    def should_request_llm(self, current_time: float) -> bool:
        """
        Decide whether to request an async LLM detection.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if we should enqueue an LLM request, False otherwise
        """
        time_since_req = current_time - self.last_llm_request
        
        return (
            not self.tracker_manager.is_active() or  # No tracker active
            time_since_req >= self.llm_check_interval  # Time for periodic check
        )
    
    def mark_llm_requested(self, current_time: float) -> None:
        """Mark that an LLM request has been enqueued."""
        self.last_llm_request = current_time

    def update_tracker(self, frame: np.ndarray) -> Optional[List[int]]:
        """
        Update classical tracker for the given frame.

        Returns:
            Normalized bbox [x1,y1,x2,y2] or None if not tracking.
        """
        success, bbox = self.tracker_manager.update(frame)
        return bbox if success else None

    def apply_llm_result(
        self,
        frame_for_init: np.ndarray,
        result: LLMResult,
        current_time: Optional[float] = None,
    ) -> Optional[List[int]]:
        """
        Apply an async LLMResult as a correction (re-init tracker).

        Returns:
            bbox if applied successfully, else None.
        """
        if current_time is None:
            current_time = time.time()

        if not result.ok or not result.bbox:
            if not self.tracker_manager.is_active():
                # Retry sooner if we have no tracker.
                self.last_llm_request = current_time - (self.llm_check_interval * 0.5)
            return None

        bbox = result.bbox
        self.tracker_manager.init_with_bbox(frame_for_init, bbox)
        return bbox
    
    def reset(self):
        """Reset tracker state."""
        self.tracker_manager.reset()
        self.last_llm_request = 0.0
        self.current_prompt = ""
    
    def get_tracker_type(self) -> str:
        """Get current tracker type."""
        return self.tracker_manager.tracker_type
    
    def is_tracking(self) -> bool:
        """Check if actively tracking an object."""
        return self.tracker_manager.is_active()
    
    def get_time_to_next_llm_check(self, current_time: float) -> float:
        """Get seconds until next LLM check."""
        time_since_last = current_time - self.last_llm_request
        return max(0, self.llm_check_interval - time_since_last)
