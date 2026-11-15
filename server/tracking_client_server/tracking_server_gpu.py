"""
MiniCPM-V-4.5 Tracking Server (GPU + Transformers version)
Optimized for GPU inference with bfloat16 precision using Transformers
"""

import re
import io
import time
import torch
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import uvicorn


# Response model
class DetectionResponse(BaseModel):
    success: bool
    bbox: Optional[List[int]] = None
    error: Optional[str] = None
    inference_time: float
    prompt: str


class TrackingServerGPU:
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-2_6",
        num_gpus: int = 1,
        max_image_size: int = 768,
    ):
        """
        Initialize the tracking server with GPU model.
        
        Args:
            model_name: HuggingFace model name
            num_gpus: Number of GPUs to use (1-8)
            max_image_size: Maximum image dimension in pixels
        """
        print("=" * 80)
        print("MiniCPM-V-4.5 Tracking Server (GPU + Transformers)")
        print("=" * 80)
        print()
        
        # GPU availability check
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This server requires GPU support.")
        
        available_gpus = torch.cuda.device_count()
        print(f"[GPU] Available GPUs: {available_gpus}")
        
        # Print GPU info
        for i in range(available_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"[GPU {i}] {props.name} | Memory: {memory_gb:.1f} GB")
        
        print()
        
        # Validate num_gpus
        if num_gpus < 1 or num_gpus > 8:
            raise ValueError(f"num_gpus must be between 1 and 8, got {num_gpus}")
        
        if num_gpus > available_gpus:
            print(f"[WARNING] Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
            num_gpus = available_gpus
        
        self.num_gpus = num_gpus
        self.max_image_size = max_image_size
        self.model_name = model_name
        
        print(f"[GPU] Using {self.num_gpus} GPU(s) for inference")
        print()
        
        # Load model
        print(f"[LOADING] Model: {model_name}")
        print(f"[LOADING] Device: cuda")
        print(f"[LOADING] Dtype: torch.bfloat16")
        print(f"[LOADING] Num GPUs: {self.num_gpus}")
        print()
        
        from transformers import AutoModel, AutoTokenizer
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with multi-GPU support
        if self.num_gpus == 1:
            # Single GPU - simple loading
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0"
            )
        else:
            # Multi-GPU - use device_map="auto" or specify GPU mapping
            device_map = {}
            # Distribute model layers across GPUs
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={i: "79GiB" for i in range(self.num_gpus)}  # Adjust based on your GPU memory
            )
        
        self.model.eval()
        
        print(f"[LOADED] Model loaded successfully on {self.num_gpus} GPU(s)")
        print()
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.successful_detections = 0
        
        print("=" * 80)
        print("Server ready! Using GPU inference with Transformers")
        print("=" * 80)
        print()
    
    def _extract_bbox_from_response(self, response: str) -> Optional[List[int]]:
        """Extract bounding box from model response."""
        # Try <box> tags first
        match = re.search(r"<box>(.*?)</box>", response, flags=re.DOTALL)
        if match:
            bbox_str = match.group(1)
            nums = re.findall(r"[-+]?\d*\.?\d+", bbox_str)
            bbox = [int(round(float(x))) for x in nums]
            if len(bbox) == 4:
                return bbox
        
        # Fallback: extract any 4 consecutive numbers from response
        nums = re.findall(r"\d+", response)
        if len(nums) >= 4:
            bbox = [int(nums[i]) for i in range(4)]
            return bbox
        
        return None
    
    def detect(self, image: Image.Image, prompt: str) -> DetectionResponse:
        """
        Run object detection using GPU model.
        """
        start_time = time.time()
        
        try:
            # Resize image if needed
            if max(image.size) > self.max_image_size:
                image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            
            # Prepare grounding question
            question = f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>. Return format: <box>x1 y1 x2 y2</box>'
            
            # Prepare messages for chat
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Generate response
            with torch.no_grad():
                response = self.model.chat(
                    image=image,
                    msgs=[{"role": "user", "content": question}],
                    tokenizer=self.tokenizer,
                    sampling=False,  # Greedy decoding for deterministic results
                    max_new_tokens=64,  # Short output for bbox
                )
            
            inference_time = time.time() - start_time
            
            if response:
                print(f"[DEBUG] Model response ({len(response)} chars): {response[:300]}")
            else:
                print("[ERROR] Empty response from model")
                return DetectionResponse(
                    success=False,
                    error="Empty response from model",
                    inference_time=inference_time,
                    prompt=prompt,
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
                print(f"[DETECTION] Not found - Prompt: '{prompt}' | Response: {response[:200]}... | Time: {inference_time:.2f}s")
                return DetectionResponse(
                    success=False,
                    error=f"Object not found in image. Model response: {response[:200]}",
                    inference_time=inference_time,
                    prompt=prompt
                )
        
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
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
            "device": f"CUDA (GPU x{self.num_gpus})",
            "dtype": "bfloat16"
        }


# ============ FastAPI App ============

# Global variable for tracking server (will be initialized in main)
tracking_server = None

app = FastAPI(
    title="MiniCPM-V-4.5 Tracking Server (GPU)",
    description="Fast inference with GPU acceleration using Transformers",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "MiniCPM-V-4.5 GPU Server",
        "model": tracking_server.model_name if tracking_server else "Not loaded",
        "num_gpus": tracking_server.num_gpus if tracking_server else 0,
        "dtype": "bfloat16",
    }


@app.get("/health")
async def health():
    model_loaded = tracking_server is not None and tracking_server.model is not None
    return {
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "num_gpus": tracking_server.num_gpus if tracking_server else 0,
        "device": "CUDA",
        "dtype": "bfloat16",
    }


@app.get("/stats")
async def stats():
    if tracking_server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return tracking_server.get_stats()


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    if tracking_server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
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
    global tracking_server
    
    import argparse
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.5 GPU Tracking Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (1-8)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Max image dimension in pixels (affects token count & speed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/MiniCPM-V-2_6",
        help="Model name from HuggingFace"
    )
    args = parser.parse_args()
    
    # Initialize tracking server
    print("Initializing GPU tracking server...")
    tracking_server = TrackingServerGPU(
        model_name=args.model,
        num_gpus=args.num_gpus,
        max_image_size=args.image_size,
    )
    
    # Run FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
