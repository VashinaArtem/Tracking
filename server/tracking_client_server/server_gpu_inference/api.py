"""
FastAPI application and endpoints
"""

import io
from typing import Optional
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from .models import DetectionResponse
from .tracking_engine import TrackingServerGPU
from .utils import ensure_rgb


class TrackingAPI:
    """FastAPI application for tracking server."""
    
    def __init__(self):
        """Initialize API without tracking server."""
        self.tracking_server: Optional[TrackingServerGPU] = None
        self.app = self._create_app()
    
    def set_tracking_server(self, server: TrackingServerGPU) -> None:
        """Set the tracking server instance."""
        self.tracking_server = server
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="MiniCPM-V-4.5 Tracking Server (GPU)",
            description="Fast inference with GPU acceleration using Transformers",
            version="1.0.0"
        )
        
        # Register routes
        app.get("/")(self.root)
        app.get("/health")(self.health)
        app.get("/stats")(self.stats)
        app.post("/detect", response_model=DetectionResponse)(self.detect)
        
        return app
    
    async def root(self):
        """Root endpoint with server info."""
        return {
            "status": "online",
            "message": "MiniCPM-V-4.5 GPU Server",
            "model": self.tracking_server.model_name if self.tracking_server else "Not loaded",
            "num_gpus": self.tracking_server.num_gpus if self.tracking_server else 0,
            "dtype": "bfloat16",
        }
    
    async def health(self):
        """Health check endpoint."""
        model_loaded = (
            self.tracking_server is not None
            and self.tracking_server.model is not None
        )
        return {
            "status": "healthy" if model_loaded else "starting",
            "model_loaded": model_loaded,
            "num_gpus": self.tracking_server.num_gpus if self.tracking_server else 0,
            "device": "CUDA",
            "dtype": "bfloat16",
        }
    
    async def stats(self):
        """Server statistics endpoint."""
        if self.tracking_server is None:
            raise HTTPException(status_code=503, detail="Server not initialized")
        return self.tracking_server.get_stats()
    
    async def detect(
        self,
        image: UploadFile = File(...),
        prompt: str = Form(...)
    ) -> DetectionResponse:
        """
        Object detection endpoint.
        
        Args:
            image: Image file to process
            prompt: Text description of object to detect
            
        Returns:
            DetectionResponse with bbox coordinates or error
        """
        if self.tracking_server is None:
            raise HTTPException(status_code=503, detail="Server not initialized")
        
        try:
            # Read and process image
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = ensure_rgb(pil_image)
            
            # Run detection
            result = self.tracking_server.detect(pil_image, prompt)
            return result
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def create_app(tracking_server: TrackingServerGPU) -> FastAPI:
    """
    Create FastAPI app with tracking server.
    
    Args:
        tracking_server: Initialized tracking server instance
        
    Returns:
        Configured FastAPI application
    """
    api = TrackingAPI()
    api.set_tracking_server(tracking_server)
    return api.app
