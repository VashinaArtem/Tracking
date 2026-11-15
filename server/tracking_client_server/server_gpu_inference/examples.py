"""
Examples of using the modular tracking server components
"""

from PIL import Image
from io import BytesIO
import requests


def example_1_basic_usage():
    """Example 1: Basic usage with default config."""
    from server_gpu_inference import ServerConfig, TrackingServerGPU
    
    # Create config
    config = ServerConfig(
        model_name="openbmb/MiniCPM-V-4_5",
        num_gpus=1,
        max_image_size=768
    )
    
    # Initialize server
    server = TrackingServerGPU(config)
    
    # Load test image
    image = Image.open("test_image.jpg")
    
    # Detect object
    result = server.detect(image, "red car")
    
    if result.success:
        print(f"Found object at: {result.bbox}")
        print(f"Inference time: {result.inference_time:.2f}s")
    else:
        print(f"Detection failed: {result.error}")


def example_2_custom_config():
    """Example 2: Custom configuration."""
    from server_gpu_inference.config import ServerConfig
    from server_gpu_inference import TrackingServerGPU
    
    # Custom config
    config = ServerConfig(
        model_name="openbmb/MiniCPM-V-4_5",
        num_gpus=2,  # Multi-GPU
        max_image_size=1024,  # Larger images
        max_new_tokens=128,  # More tokens for response
        use_sampling=True  # Use sampling instead of greedy
    )
    
    config.validate()
    
    server = TrackingServerGPU(config)
    # ... use server


def example_3_bbox_parser():
    """Example 3: Using BBoxParser standalone."""
    from server_gpu_inference.bbox_parser import BBoxParser
    
    parser = BBoxParser()
    
    # Test different formats
    responses = [
        "<box>100 200 300 400</box>",
        "The object is at coordinates 150 250 350 450",
        "<box>50.5 60.2 250.8 300.1</box>",
    ]
    
    for response in responses:
        bbox = parser.extract_bbox(response)
        if bbox:
            print(f"Extracted: {bbox}")
            is_valid = parser.validate_bbox(bbox)
            print(f"Valid: {is_valid}")
        else:
            print(f"Could not extract bbox from: {response}")


def example_4_gpu_manager():
    """Example 4: Using GPUManager utilities."""
    from server_gpu_inference.gpu_manager import GPUManager
    
    # Check CUDA availability
    if GPUManager.check_cuda_available():
        print("CUDA is available!")
        
        # Print GPU info
        GPUManager.print_gpu_info()
        
        # Get GPU count
        gpu_count = GPUManager.get_gpu_count()
        print(f"Available GPUs: {gpu_count}")
        
        # Validate GPU request
        requested = 4
        actual = GPUManager.validate_gpu_count(requested, gpu_count)
        print(f"Requested {requested}, using {actual} GPUs")
        
        # Get device and dtype
        device, dtype = GPUManager.get_device_and_dtype(use_gpu=True)
        print(f"Device: {device}, Dtype: {dtype}")
    else:
        print("CUDA not available")


def example_5_image_utils():
    """Example 5: Using image utilities."""
    from server_gpu_inference.utils import (
        resize_image_if_needed,
        ensure_rgb,
        format_grounding_question
    )
    from PIL import Image
    
    # Load image
    image = Image.open("test_image.jpg")
    
    # Resize if needed
    image = resize_image_if_needed(image, max_size=768)
    print(f"Image size after resize: {image.size}")
    
    # Ensure RGB
    image = ensure_rgb(image)
    print(f"Image mode: {image.mode}")
    
    # Format question
    prompt = "red car"
    question = format_grounding_question(prompt)
    print(f"Question: {question}")


def example_6_fastapi_app():
    """Example 6: Create FastAPI app programmatically."""
    from server_gpu_inference import ServerConfig, TrackingServerGPU, create_app
    import uvicorn
    
    # Setup
    config = ServerConfig()
    server = TrackingServerGPU(config)
    app = create_app(server)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)


def example_7_api_client():
    """Example 7: Client to test the API."""
    import requests
    from PIL import Image
    from io import BytesIO
    
    # Server URL
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    print("Health:", response.json())
    
    # Test detection endpoint
    with open("test_image.jpg", "rb") as f:
        files = {"image": f}
        data = {"prompt": "red car"}
        response = requests.post(f"{base_url}/detect", files=files, data=data)
    
    result = response.json()
    if result["success"]:
        print(f"Detected at: {result['bbox']}")
    else:
        print(f"Failed: {result['error']}")
    
    # Get stats
    response = requests.get(f"{base_url}/stats")
    print("Stats:", response.json())


def example_8_batch_detection():
    """Example 8: Batch detection (multiple images)."""
    from server_gpu_inference import ServerConfig, TrackingServerGPU
    from PIL import Image
    from pathlib import Path
    
    config = ServerConfig()
    server = TrackingServerGPU(config)
    
    # Multiple images
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    prompts = ["person", "car", "dog"]
    
    results = []
    for img_path, prompt in zip(image_paths, prompts):
        if Path(img_path).exists():
            image = Image.open(img_path)
            result = server.detect(image, prompt)
            results.append({
                "image": img_path,
                "prompt": prompt,
                "bbox": result.bbox if result.success else None,
                "time": result.inference_time
            })
    
    # Print results
    for res in results:
        print(f"{res['image']}: {res['bbox']} ({res['time']:.2f}s)")
    
    # Print statistics
    print("\nServer stats:", server.get_stats())


def example_9_error_handling():
    """Example 9: Error handling and validation."""
    from server_gpu_inference.config import ServerConfig
    from server_gpu_inference.bbox_parser import BBoxParser
    
    # Test config validation
    try:
        config = ServerConfig(num_gpus=10)  # Too many
        config.validate()
    except ValueError as e:
        print(f"Config error: {e}")
    
    # Test bbox validation
    parser = BBoxParser()
    
    invalid_boxes = [
        [300, 400, 100, 200],  # x2 < x1
        [-10, 20, 30, 40],     # negative coordinate
        [10, 20, 30],          # wrong length
    ]
    
    for bbox in invalid_boxes:
        is_valid = parser.validate_bbox(bbox)
        print(f"{bbox} valid: {is_valid}")


if __name__ == "__main__":
    print("Running examples...")
    print("\n=== Example 3: BBox Parser ===")
    example_3_bbox_parser()
    
    print("\n=== Example 4: GPU Manager ===")
    example_4_gpu_manager()
    
    print("\n=== Example 5: Image Utils ===")
    # example_5_image_utils()  # Needs test image
    
    print("\n=== Example 9: Error Handling ===")
    example_9_error_handling()
