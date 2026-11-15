"""
Allow running the package as a module:
python -m server_gpu_inference
"""

from .tracking_server_gpu import main

if __name__ == "__main__":
    main()
