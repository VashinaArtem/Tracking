"""
Test script for tracking_server.py
Tests the server with a local image file
"""

import requests
import sys
from pathlib import Path


def test_server(server_url: str, image_path: str, prompt: str):
    """
    Test the tracking server with an image file.
    
    Args:
        server_url: URL of the server (e.g., http://localhost:8000)
        image_path: Path to image file
        prompt: Text description of object to detect
    """
    print("=" * 60)
    print("Testing MiniCPM-V Tracking Server")
    print("=" * 60)
    print(f"\nServer: {server_url}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-" * 60)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"[ERROR] Image not found: {image_path}")
        return False
    
    # Check server health
    print("\n[1/3] Checking server health...")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is healthy")
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {server_url}")
        print("   Make sure the server is running!")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Send detection request
    print("\n[2/3] Sending detection request...")
    try:
        with open(image_path, 'rb') as f:
            files = {'image': ('test.jpg', f, 'image/jpeg')}
            data = {'prompt': prompt}
            
            response = requests.post(
                f"{server_url}/detect",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            print("✓ Request successful")
        else:
            print(f"✗ Server returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Parse response
    print("\n[3/3] Parsing response...")
    try:
        result = response.json()
        
        print("\nResults:")
        print("-" * 60)
        print(f"Success: {result['success']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Inference time: {result['inference_time']:.2f}s")
        
        if result['success']:
            print(f"BBox: {result['bbox']}")
            print("\nBounding box coordinates (normalized 0-1000):")
            print(f"  Top-left: ({result['bbox'][0]}, {result['bbox'][1]})")
            print(f"  Bottom-right: ({result['bbox'][2]}, {result['bbox'][3]})")
            print("\n✓ Object detected successfully!")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            print("\n✗ Object not found")
        
        print("-" * 60)
        return result['success']
    
    except Exception as e:
        print(f"✗ Error parsing response: {e}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MiniCPM-V Tracking Server")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of object to detect"
    )
    
    args = parser.parse_args()
    
    success = test_server(args.server, args.image, args.prompt)
    
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED ✓")
    else:
        print("TEST FAILED ✗")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
