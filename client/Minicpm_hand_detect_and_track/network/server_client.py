"""
Network client for communicating with MiniCPM-V detection server.

Handles all network communication including health checks, detection requests,
and statistics retrieval.
"""
import io
import logging
from typing import Optional, List, Dict, Any
import requests
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ServerClient:
    """Client for MiniCPM-V object detection server."""
    
    def __init__(
        self,
        server_url: str,
        jpeg_quality: int = 85,
        image_max_size: int = 512,
        timeout: int = 120
    ):
        """
        Initialize server client.
        
        Args:
            server_url: Base URL of the detection server
            jpeg_quality: JPEG compression quality (0-100)
            image_max_size: Maximum dimension for image resizing
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.jpeg_quality = jpeg_quality
        self.image_max_size = image_max_size
        self.timeout = timeout
        
        self.online = False
        self.last_error = ""
    
    def check_health(self) -> bool:
        """
        Check if server is accessible and responding.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                self.online = True
                return True
            else:
                self.last_error = f"Server returned status {response.status_code}"
                self.online = False
                return False
        except requests.exceptions.ConnectionError:
            self.last_error = "Connection refused - is the server running?"
            self.online = False
            return False
        except requests.exceptions.Timeout:
            self.last_error = "Connection timeout"
            self.online = False
            return False
        except Exception as e:
            self.last_error = str(e)
            self.online = False
            return False
    
    def _frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        """
        Convert OpenCV frame to JPEG bytes with smart resizing.
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            
        Returns:
            JPEG encoded bytes
        """
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
    
    def detect(
        self, 
        frame: np.ndarray, 
        prompt: str
    ) -> tuple[Optional[List[int]], float, float]:
        """
        Send frame to server for object detection.
        
        Args:
            frame: OpenCV frame (BGR numpy array)
            prompt: Text description of object to detect
            
        Returns:
            Tuple of (bbox, inference_time, network_time) where:
            - bbox: [x1, y1, x2, y2] in 0-1000 range, or None if not detected
            - inference_time: Server inference time in seconds
            - network_time: Network round-trip time in seconds
        """
        import time
        
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
                timeout=self.timeout
            )
            
            network_time = time.time() - network_start
            
            if response.status_code == 200:
                result = response.json()
                
                if result['success']:
                    bbox = result['bbox']
                    inference_time = result['inference_time']
                    
                    logger.info(f"Detection successful: {bbox} | "
                              f"Inference: {inference_time:.2f}s | "
                              f"Network: {network_time:.2f}s")
                    
                    self.online = True
                    return bbox, inference_time, network_time
                else:
                    logger.info(f"Object not detected | "
                              f"Inference: {result['inference_time']:.2f}s | "
                              f"Network: {network_time:.2f}s")
                    self.online = True
                    return None, result['inference_time'], network_time
            else:
                logger.error(f"Server returned status {response.status_code}")
                self.online = False
                return None, 0.0, network_time
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to server at {self.server_url}")
            self.online = False
            return None, 0.0, 0.0
        except requests.exceptions.Timeout:
            logger.error("Request timeout - server may be overloaded")
            return None, 0.0, 0.0
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None, 0.0, 0.0
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Fetch server statistics.
        
        Returns:
            Dictionary of server statistics, or None if request failed
        """
        try:
            response = requests.get(f"{self.server_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Cannot fetch stats: status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Cannot fetch stats: {e}")
            return None
