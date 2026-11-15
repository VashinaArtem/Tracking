"""
MiniCPM-V-4.5 Tracking Server
Server-side inference for real-time object tracking

This server receives images and prompts from clients, runs MiniCPM-V-4.5 inference,
and returns bounding box coordinates.
"""

import re
import io
import time
from typing import Optional, List
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoModel, AutoTokenizer


# Response model
class DetectionResponse(BaseModel):
    success: bool
    bbox: Optional[List[int]] = None
    error: Optional[str] = None
    inference_time: float
    prompt: str


class TrackingServer:
    def __init__(self, model_path: str = 'openbmb/MiniCPM-V-4_5', use_gpu: bool = True):
        """
        Initialize the tracking server.
        
        Args:
            model_path: Path to MiniCPM-V model (HuggingFace or local)
            use_gpu: Use GPU if available
        """
        print("=" * 60)
        print("MiniCPM-V-4.5 Tracking Server")
        print("=" * 60)
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        print(f"\n[1/2] Loading MiniCPM-V-4.5 model...")
        print(f"Model path: {model_path}")
        print(f"Device: {'CUDA' if self.use_gpu else 'CPU'}")
        
        self.model, self.tokenizer = self._setup_model(model_path)
        
        print("✓ Model loaded successfully")
        print("\n[2/2] Initializing FastAPI server...")
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.successful_detections = 0
        
        print("✓ Server initialized")
        print("\n" + "=" * 60)
        print("Server ready to accept connections!")
        print("=" * 60 + "\n")
    
    def _setup_model(self, model_path: str):
        """Load MiniCPM-V model and tokenizer."""
        if self.use_gpu:
            dtype = torch.bfloat16
            device = 'cuda'
        else:
            dtype = torch.float32
            device = 'cpu'
        
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model = model.to(dtype=dtype)
        
        if self.use_gpu:
            model = model.cuda()
        
        model = model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def _extract_bbox_from_response(self, response: str) -> Optional[List[int]]:
        """
        Extract bounding box coordinates from model response.
        Expected format: <box>x1 y1 x2 y2</box>
        """
        match = re.search(r"<box>([\d\s]+)</box>", response)
        if match:
            bbox_str = match.group(1)
            bbox = list(map(int, bbox_str.strip().split()))
            if len(bbox) == 4:
                return bbox
        return None
    
    def detect(self, image: Image.Image, prompt: str) -> DetectionResponse:
        """
        Run object detection on image with given prompt.
        
        Args:
            image: PIL Image
            prompt: Text description of object to detect
        
        Returns:
            DetectionResponse with bbox coordinates or error
        """
        start_time = time.time()
        
        try:
            # Prepare question in grounding format
            question = f"Please provide the bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>"
            
            msgs = [
                {'role': 'user', 'content': [question, image]},
            ]
            
            # Run inference
            with torch.inference_mode():
                response = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=True,
                    max_new_tokens=1024,
                    max_inp_length=8192,
                    use_image_id=True,
                )
            
            inference_time = time.time() - start_time
            
            # Extract bbox
            bbox = self._extract_bbox_from_response(response)
            
            # Update statistics
            self.total_requests += 1
            self.total_inference_time += inference_time
            
            if bbox:
                self.successful_detections += 1
                print(f"[DETECTION] Success - Prompt: '{prompt}' | BBox: {bbox} | Time: {inference_time:.2f}s")
                return DetectionResponse(
                    success=True,
                    bbox=bbox,
                    inference_time=inference_time,
                    prompt=prompt
                )
            else:
                print(f"[DETECTION] Not found - Prompt: '{prompt}' | Response: {response[:100]}... | Time: {inference_time:.2f}s")
                return DetectionResponse(
                    success=False,
                    error="Object not found in image",
                    inference_time=inference_time,
                    prompt=prompt
                )
        
        except Exception as e:
            inference_time = time.time() - start_time
            print(f"[ERROR] {str(e)}")
            return DetectionResponse(
                success=False,
                error=str(e),
                inference_time=inference_time,
                prompt=prompt
            )
    
    def get_stats(self) -> dict:
        """Get server statistics."""
        avg_time = (self.total_inference_time / self.total_requests) if self.total_requests > 0 else 0
        success_rate = (self.successful_detections / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_detections": self.successful_detections,
            "success_rate": f"{success_rate:.1f}%",
            "avg_inference_time": f"{avg_time:.2f}s",
            "total_inference_time": f"{self.total_inference_time:.2f}s",
            "device": "CUDA" if self.use_gpu else "CPU"
        }


# Initialize server
print("Initializing tracking server...")
tracking_server = TrackingServer(
    model_path='openbmb/MiniCPM-V-4_5',
    use_gpu=False  # Set to True for GPU inference (if GPU available and not busy)
)

# FastAPI app
app = FastAPI(
    title="MiniCPM-V-4.5 Tracking Server",
    description="Real-time object detection server using MiniCPM-V-4.5",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "MiniCPM-V-4.5 Tracking Server is running",
        "endpoints": {
            "/detect": "POST - Detect objects in image",
            "/stats": "GET - Server statistics",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True}


@app.get("/stats")
async def stats():
    """Get server statistics."""
    return tracking_server.get_stats()


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(..., description="Image file to process"),
    prompt: str = Form(..., description="Text description of object to detect")
):
    """
    Detect object in image based on text prompt.
    
    Args:
        image: Image file (JPEG, PNG, etc.)
        prompt: Text description of object to detect
    
    Returns:
        DetectionResponse with bbox coordinates [x1, y1, x2, y2] (normalized 0-1000)
    """
    try:
        # Read image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Run detection
        result = tracking_server.detect(pil_image, prompt)
        
        return result
    
    except Exception as e:
        print(f"[ERROR] Failed to process request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.5 Tracking Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Access the API at: http://{args.host}:{args.port}")
    print(f"API docs available at: http://{args.host}:{args.port}/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
