"""
Visualization utilities for hand tracking display.

Provides functions for drawing bounding boxes, overlays, and status information.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple


class Visualizer:
    """Handles all visualization for the tracking system."""
    
    def __init__(self, clean_ui: bool = False):
        """Initialize visualizer.
        
        Args:
            clean_ui: When True, only the bounding box is drawn — all text overlays are suppressed.
        """
        self.clean_ui = clean_ui
    
    def draw_bbox(
        self, 
        frame: np.ndarray, 
        bbox: List[int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame (BGR)
            bbox: Normalized bbox [x1,y1,x2,y2] (0-1000)
            label: Label text to display
            color: Box color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Frame with drawn bbox
        """
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided and UI is not in clean mode
        if label and not self.clean_ui:
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
        cv2.circle(frame, (center_x, center_y), 8, color, -1)
        cv2.circle(frame, (center_x, center_y), 12, color, 2)
        
        return frame
    
    def draw_info_overlay(
        self,
        frame: np.ndarray,
        prompt: str,
        avg_fps: Optional[float],
        avg_inference: Optional[float],
        avg_network: Optional[float],
        is_tracking: bool,
        tracker_type: str,
        time_to_next_check: float,
        pan_position: float,
        tilt_position: float,
        pan_output: float,
        tilt_output: float,
        voice_state: str,
        server_online: bool,
        image_center: Optional[Tuple[int, int]],
        bbox_age_ms: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw information overlay on frame.
        
        Args:
            frame: Input frame
            prompt: Current tracking prompt
            avg_fps: Average FPS
            avg_inference: Average inference time
            avg_network: Average network time
            is_tracking: Whether actively tracking
            tracker_type: Type of tracker (KCF/CSRT)
            time_to_next_check: Seconds until next LLM check
            pan_position: Pan servo position
            tilt_position: Tilt servo position
            pan_output: Pan PID output
            tilt_output: Tilt PID output
            voice_state: Voice state string ("idle" | "active" | "processing")
            server_online: Server status
            image_center: Image center coordinates
            
        Returns:
            Frame with overlay
        """
        if self.clean_ui:
            return frame

        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current prompt
        prompt_text = prompt if prompt else "No prompt set"
        cv2.putText(frame, f"Prompt: {prompt_text}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display FPS
        if avg_fps is not None:
            cv2.putText(frame, f"Display FPS: {avg_fps:.1f}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Inference time
        if avg_inference is not None:
            cv2.putText(frame, f"Inference: {avg_inference:.2f}s", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Network time
        if avg_network is not None:
            cv2.putText(frame, f"Network: {avg_network:.2f}s", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tracking mode indicator
        if is_tracking:
            tracking_text = f"Mode: TRACKING ({tracker_type})"
            tracking_color = (0, 255, 0)  # Green
            tracking_text += f" | Next: {time_to_next_check:.1f}s"
        else:
            tracking_text = "Mode: LLM DETECTION"
            tracking_color = (255, 255, 0)  # Yellow
        
        cv2.putText(frame, tracking_text, (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
        
        # Servo positions
        cv2.putText(frame, f"Pan: {pan_position:.3f}, Tilt: {tilt_position:.3f}", 
                   (20, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # PID outputs
        cv2.putText(frame, f"PID: Pan={pan_output:.3f} Tilt={tilt_output:.3f}", 
                   (20, 215),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Voice status
        _voice_display = {
            "idle":       ((128, 128, 128), "Voice: say 'Atlas'"),
            "active":     ((0, 255, 255),   "Voice: LISTENING..."),
            "processing": ((0, 215, 255),   "Voice: Processing..."),
        }
        voice_color, voice_text = _voice_display.get(voice_state, _voice_display["idle"])
        cv2.putText(frame, voice_text, (20, 245),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)

        # BBox age (debug / async pipeline)
        if bbox_age_ms is not None:
            cv2.putText(
                frame,
                f"BBox age: {bbox_age_ms:.0f}ms",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )
        
        # Server status
        status_color = (0, 255, 0) if server_online else (0, 0, 255)
        status_text = "Server: ONLINE" if server_online else "Server: OFFLINE"
        cv2.putText(frame, status_text, (w - 200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Image center crosshair
        if image_center:
            cv2.circle(frame, image_center, 15, (255, 0, 0), 2)
            cv2.circle(frame, image_center, 3, (255, 0, 0), -1)
        
        # Controls
        cv2.putText(frame, "q:quit | c:clear | s:stats", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def display_server_stats(self, stats: dict):
        """
        Print server statistics to console.
        
        Args:
            stats: Dictionary of server statistics
        """
        print("\n" + "=" * 60)
        print("SERVER STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key:.<30} {value}")
        print("=" * 60 + "\n")
