"""
MiniCPM-V-4.5 Tracking Server (GPU + Transformers version)
Entry point for GPU-accelerated tracking server

Modular implementation with separated concerns:
- models.py: Data models
- config.py: Configuration management
- bbox_parser.py: Bounding box extraction
- gpu_manager.py: GPU utilities
- tracking_engine.py: Core inference engine
- api.py: FastAPI endpoints
- utils.py: Helper functions
"""

import argparse
import uvicorn

from .config import ServerConfig, create_config_from_args, DEFAULT_MODEL
from .tracking_engine import TrackingServerGPU
from .api import create_app


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiniCPM-V-4.5 GPU Tracking Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (1-8). For multi-GPU, install accelerate for best performance."
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
        default=DEFAULT_MODEL,
        help="Model name from HuggingFace"
    )
    
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable model reasoning/thinking mode. Off by default for faster bbox-only inference."
    )

    return parser.parse_args()


def main():
    """Main entry point for the tracking server."""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Initialize tracking server
    print("Initializing GPU tracking server...")
    print()
    tracking_server = TrackingServerGPU(config)
    
    # Create FastAPI app
    app = create_app(tracking_server)
    
    # Run server
    print(f"Starting FastAPI server on {config.host}:{config.port}...")
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
