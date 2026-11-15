"""
MiniCPM-V-4.5 Tracking Server (GGUF + llama.cpp version)
Optimized for CPU inference with INT4 quantization
"""

import re
import io
import time
import base64
import subprocess
import tempfile
import os
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# Response model
class DetectionResponse(BaseModel):
    success: bool
    bbox: Optional[List[int]] = None
    error: Optional[str] = None
    inference_time: float
    prompt: str


class TrackingServerGGUF:
    def __init__(
        self, 
        model_path: str = "/home/jovyan/shares/SR006.nfs2/vashina/models/minicpm-v4_5-gguf/ggml-model-Q4_K_M.gguf",
        mmproj_path: str = "/home/jovyan/shares/SR006.nfs2/vashina/models/minicpm-v4_5-gguf/mmproj-model-f16.gguf",
        llama_cli_path: str = "/home/jovyan/shares/SR006.nfs2/vashina/llama.cpp/build/bin/llama-mtmd-cli"
    ):
        """
        Initialize the tracking server with GGUF model.
        """
        print("=" * 60)
        print("MiniCPM-V-4.5 Tracking Server (GGUF INT4)")
        print("=" * 60)
        
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.llama_cli_path = llama_cli_path
        
        # Verify files exist
        for path, name in [(model_path, "Language model"), 
                           (mmproj_path, "Vision model"),
                           (llama_cli_path, "llama-mtmd-cli")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")
            print(f"✓ {name}: {path}")
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.successful_detections = 0
        
        print("\n" + "=" * 60)
        print("Server ready! Using INT4 quantization for fast CPU inference")
        print("=" * 60 + "\n")
    
    def _extract_bbox_from_response(self, response: str) -> Optional[List[int]]:
        """Extract bounding box from model response."""
        match = re.search(r"<box>([\d\s]+)</box>", response)
        if match:
            bbox_str = match.group(1)
            bbox = list(map(int, bbox_str.strip().split()))
            if len(bbox) == 4:
                return bbox
        return None
    
    def detect(self, image: Image.Image, prompt: str) -> DetectionResponse:
        """
        Run object detection using llama.cpp CLI.
        """
        start_time = time.time()
        
        try:
            # Save image to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp, format='JPEG', quality=90)
                tmp_image_path = tmp.name
            
            # Prepare grounding question
            question = f"Please provide the bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>"
            
            # Build llama.cpp command
            cmd = [
                self.llama_cli_path,
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "-c", "4096",
                "--temp", "0.7",
                "--top-p", "0.8",
                "--top-k", "100",
                "--repeat-penalty", "1.05",
                "--image", tmp_image_path,
                "-p", question,
                "-n", "100",  # Max tokens (for bbox we need ~20)
                "--no-display-prompt"
            ]
            
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Clean up temp file
            os.unlink(tmp_image_path)
            
            inference_time = time.time() - start_time
            
            # Parse output
            response = result.stdout.strip()
            
            if result.returncode != 0:
                print(f"[ERROR] llama.cpp error: {result.stderr}")
                return DetectionResponse(
                    success=False,
                    error=f"Inference failed: {result.stderr[:200]}",
                    inference_time=inference_time,
                    prompt=prompt
                )
            
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
                print(f"[DETECTION] Not found - Response: {response[:100]}... | Time: {inference_time:.2f}s")
                return DetectionResponse(
                    success=False,
                    error="Object not found in image",
                    inference_time=inference_time,
                    prompt=prompt
                )
        
        except subprocess.TimeoutExpired:
            return DetectionResponse(
                success=False,
                error="Inference timeout",
                inference_time=time.time() - start_time,
                prompt=prompt
            )
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return DetectionResponse(
                success=False,
                error=str(e),
                inference_time=time.time() - start_time,
                prompt=prompt
            )
    
    def get_stats(self) -> dict:
        avg_time = (self.total_inference_time / self.total_requests) if self.total_requests > 0 else 0
        success_rate = (self.successful_detections / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_detections": self.successful_detections,
            "success_rate": f"{success_rate:.1f}%",
            "avg_inference_time": f"{avg_time:.2f}s",
            "device": "CPU (INT4 GGUF)"
        }


# ============ FastAPI App ============

# Paths - ИЗМЕНИТЕ ЭТИ ПУТИ НА ВАШИ
MODEL_PATH = "/home/jovyan/shares/SR006.nfs2/vashina/models/minicpm-v4_5-gguf/ggml-model-Q4_K_M.gguf"
MMPROJ_PATH = "/home/jovyan/shares/SR006.nfs2/vashina/models/minicpm-v4_5-gguf/mmproj-model-f16.gguf"
LLAMA_CLI_PATH = "/home/jovyan/shares/SR006.nfs2/vashina/llama.cpp/build/bin/llama-mtmd-cli"

print("Initializing GGUF tracking server...")
tracking_server = TrackingServerGGUF(
    model_path=MODEL_PATH,
    mmproj_path=MMPROJ_PATH,
    llama_cli_path=LLAMA_CLI_PATH
)

app = FastAPI(
    title="MiniCPM-V-4.5 Tracking Server (GGUF)",
    description="Fast CPU inference with INT4 quantization",
    version="2.0.0"
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "MiniCPM-V-4.5 GGUF Server",
        "quantization": "INT4 (Q4_K_M)"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True, "quantization": "INT4"}


@app.get("/stats")
async def stats():
    return tracking_server.get_stats()


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        result = tracking_server.detect(pil_image, prompt)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()