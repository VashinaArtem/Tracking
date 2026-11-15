"""
Core tracking engine with GPU-accelerated inference
"""

import re
import time
import torch
from typing import Optional
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from .models import DetectionResponse
from .config import ServerConfig
from .bbox_parser import BBoxParser
from .gpu_manager import GPUManager
from .utils import (
    resize_image_if_needed,
    format_grounding_question,
    print_section_header,
    print_detection_log
)


class TrackingServerGPU:
    """GPU-accelerated tracking server engine."""
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the tracking server with GPU model.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.bbox_parser = BBoxParser()
        
        print_section_header("MiniCPM-V-4.5 Tracking Server (GPU + Transformers)")
        
        # GPU availability check
        if not GPUManager.check_cuda_available():
            raise RuntimeError("CUDA is not available. This server requires GPU support.")
        
        # Print GPU info
        GPUManager.print_gpu_info()
        
        # Validate and adjust GPU count
        available_gpus = GPUManager.get_gpu_count()
        self.num_gpus = GPUManager.validate_gpu_count(
            config.num_gpus,
            available_gpus
        )
        
        print(f"[GPU] Using {self.num_gpus} GPU(s) for inference")
        print()
        
        # Load model
        self._load_model()
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.successful_detections = 0
        
        print_section_header("Server ready! Using GPU inference with Transformers")
    
    def _load_model(self) -> None:
        """Load model and tokenizer."""
        print(f"[LOADING] Model: {self.config.model_name}")
        print(f"[LOADING] Device: cuda")
        print(f"[LOADING] Dtype: torch.bfloat16")
        print(f"[LOADING] Num GPUs: {self.num_gpus}")
        print()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Load processor (required by MiniCPM-V-4.5 chat for thinking mode)
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Load model based on GPU configuration
        if self.num_gpus == 1:
            self._load_single_gpu()
        else:
            self._load_multi_gpu()
        
        self.model.eval()
        
        print(f"[LOADED] Model loaded successfully on {self.num_gpus} GPU(s)")
        print()
    
    def _load_single_gpu(self) -> None:
        """Load model on single GPU."""
        print("[LOADING] Using single GPU mode (cuda:0)...")
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to('cuda:0')
        self.use_data_parallel = False
    
    def _load_multi_gpu(self) -> None:
        """Load model with multi-GPU support."""
        try:
            import accelerate
            print("[LOADING] Using accelerate for multi-GPU distribution...")
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={i: "79GiB" for i in range(self.num_gpus)}
            )
            self.use_data_parallel = False
        except ImportError:
            print("[WARNING] accelerate not installed. Using DataParallel for multi-GPU (less efficient).")
            print("[INFO] For better multi-GPU support, install: pip install accelerate")
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to('cuda:0')
            
            if self.num_gpus > 1:
                device_ids = list(range(self.num_gpus))
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
                self.use_data_parallel = True
                print(f"[GPU] Using DataParallel on GPUs: {device_ids}")
            else:
                self.use_data_parallel = False
    
    def detect(self, image: Image.Image, prompt: str) -> DetectionResponse:
        """
        Run object detection using GPU model.
        
        Args:
            image: PIL Image
            prompt: Text description of object to detect
            
        Returns:
            DetectionResponse with bbox coordinates or error
        """
        start_time = time.time()

        max_new_tokens = (
            self.config.max_new_tokens_thinking
            if self.config.enable_thinking
            else self.config.max_new_tokens
        )
        
        try:
            # Resize image if needed
            image = resize_image_if_needed(image, self.config.max_image_size)
            
            # Prepare grounding question
            question = format_grounding_question(prompt)
            
            # Get the actual model (unwrap DataParallel if needed)
            model = self.model.module if self.use_data_parallel else self.model
            
            # Generate response
            with torch.no_grad():
                response = model.chat(
                    image=image,
                    msgs=[{"role": "user", "content": question}],
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    sampling=self.config.use_sampling,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=self.config.enable_thinking,
                )

            inference_time = time.time() - start_time
            # Handle empty response
            if not response:
                print("[ERROR] Empty response from model")
                return DetectionResponse(
                    success=False,
                    error="Empty response from model",
                    inference_time=inference_time,
                    prompt=prompt,
                )

            # print(f"[DEBUG] Model response ({len(response)} chars): {response[:300]}")
            print(f"[DEBUG] Response (type={type(response).__name__}, len={len(response)}): {repr(response)}")


            # Strip <think>...</think> blocks before parsing to prevent
            # the fallback number extractor from picking up reasoning content
            
            response_for_parsing = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            # Extract bbox
            bbox = self.bbox_parser.extract_bbox(response_for_parsing)
            
            # Update statistics
            self._update_statistics(inference_time, bbox is not None)
            
            # Return result
            if bbox:
                print_detection_log(True, prompt, bbox, inference_time)
                return DetectionResponse(
                    success=True,
                    bbox=bbox,
                    inference_time=inference_time,
                    prompt=prompt
                )
            else:
                print_detection_log(False, prompt, inference_time=inference_time, response_preview=response_for_parsing)
                return DetectionResponse(
                    success=False,
                    error=f"Object not found in image. Model response: {response_for_parsing[:200]}",
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
    
    def _update_statistics(self, inference_time: float, success: bool) -> None:
        """Update server statistics."""
        self.total_requests += 1
        self.total_inference_time += inference_time
        if success:
            self.successful_detections += 1
    
    def get_stats(self) -> dict:
        """Get server statistics."""
        avg_time = (
            self.total_inference_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        success_rate = (
            self.successful_detections / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        multi_gpu_mode = "single"
        if self.num_gpus > 1:
            multi_gpu_mode = "DataParallel" if self.use_data_parallel else "accelerate"
        
        return {
            "total_requests": self.total_requests,
            "successful_detections": self.successful_detections,
            "success_rate": f"{success_rate:.1f}%",
            "avg_inference_time": f"{avg_time:.2f}s",
            "device": f"CUDA (GPU x{self.num_gpus})",
            "dtype": "bfloat16",
            "multi_gpu_mode": multi_gpu_mode
        }
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name
