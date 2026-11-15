"""
Configuration management for tracking server
"""

from dataclasses import dataclass
from typing import Optional


# Constants
DEFAULT_MODEL = "openbmb/MiniCPM-V-4_5"
DEFAULT_MAX_IMAGE_SIZE = 768
DEFAULT_NUM_GPUS = 1
MIN_GPUS = 1
MAX_GPUS = 8
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_THINKING = False
DEFAULT_MAX_NEW_TOKENS_THINKING = 4096

@dataclass
class ServerConfig:
    """Server configuration."""
    
    # Model settings
    model_name: str = DEFAULT_MODEL
    num_gpus: int = DEFAULT_NUM_GPUS
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE
    
    # Server settings
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # Reasoning
    enable_thinking: bool = DEFAULT_THINKING

    # Inference settings
    max_new_tokens: int = 64
    use_sampling: bool = False
    max_new_tokens_thinking: int = DEFAULT_MAX_NEW_TOKENS_THINKING
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_gpus < MIN_GPUS or self.num_gpus > MAX_GPUS:
            raise ValueError(
                f"num_gpus must be between {MIN_GPUS} and {MAX_GPUS}, got {self.num_gpus}"
            )
        
        if self.max_image_size <= 0:
            raise ValueError(f"max_image_size must be positive, got {self.max_image_size}")
        
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")


def create_config_from_args(args) -> ServerConfig:
    """Create ServerConfig from command line arguments."""
    config = ServerConfig(
        model_name=args.model,
        num_gpus=args.num_gpus,
        max_image_size=args.image_size,
        host=args.host,
        port=args.port,
        enable_thinking=getattr(args, 'enable_thinking', False),
    )
    config.validate()
    return config
