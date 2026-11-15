"""
GPU management and device utilities
"""

import torch
from typing import Tuple, Optional


class GPUManager:
    """Manager for GPU availability and configuration."""
    
    @staticmethod
    def check_cuda_available() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_gpu_count() -> int:
        """Get number of available GPUs."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    
    @staticmethod
    def print_gpu_info() -> None:
        """Print information about available GPUs."""
        if not torch.cuda.is_available():
            print("[GPU] CUDA is not available")
            return
        
        gpu_count = torch.cuda.device_count()
        print(f"[GPU] Available GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"[GPU {i}] {props.name} | Memory: {memory_gb:.1f} GB")
        print()
    
    @staticmethod
    def validate_gpu_count(requested_gpus: int, available_gpus: int) -> int:
        """
        Validate and adjust requested GPU count.
        
        Args:
            requested_gpus: Number of GPUs requested
            available_gpus: Number of GPUs available
            
        Returns:
            Actual number of GPUs to use
        """
        if requested_gpus > available_gpus:
            print(
                f"[WARNING] Requested {requested_gpus} GPUs but only "
                f"{available_gpus} available. Using {available_gpus}."
            )
            return available_gpus
        return requested_gpus
    
    @staticmethod
    def get_device_and_dtype(use_gpu: bool = True) -> Tuple[str, torch.dtype]:
        """
        Get device and dtype for model.
        
        Args:
            use_gpu: Whether to use GPU if available
            
        Returns:
            Tuple of (device, dtype)
        """
        if use_gpu and torch.cuda.is_available():
            return "cuda", torch.bfloat16
        return "cpu", torch.float32
    
    @staticmethod
    def clear_cache() -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
