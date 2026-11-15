"""
Utility functions for image processing and helpers
"""

from PIL import Image
from typing import Tuple


def resize_image_if_needed(
    image: Image.Image,
    max_size: int
) -> Image.Image:
    """
    Resize image if it exceeds maximum dimension.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension in pixels
        
    Returns:
        Resized image or original if already small enough
    """
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure image is in RGB mode.
    
    Args:
        image: PIL Image
        
    Returns:
        RGB image
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def format_grounding_question(prompt: str) -> str:
    """
    Format user prompt into grounding question for the model.
    
    Args:
        prompt: User's object description
        
    Returns:
        Formatted question for model
    """
    return (
        f'Please provide the bounding box coordinate of the region this sentence '
        f'describes: <ref>{prompt}</ref>. Return format: <box>x1 y1 x2 y2</box>'
    )


def print_section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def print_detection_log(
    success: bool,
    prompt: str,
    bbox: list = None,
    inference_time: float = 0.0,
    response_preview: str = ""
) -> None:
    """
    Print formatted detection log message.
    
    Args:
        success: Whether detection was successful
        prompt: Detection prompt
        bbox: Bounding box coordinates (if successful)
        inference_time: Time taken for inference
        response_preview: Preview of model response (if failed)
    """
    if success:
        print(
            f"[DETECTION] Success - Prompt: '{prompt}' | "
            f"BBox: {bbox} | Time: {inference_time:.2f}s"
        )
    else:
        print(
            f"[DETECTION] Not found - Prompt: '{prompt}' | "
            f"Response: {response_preview[:200]}... | Time: {inference_time:.2f}s"
        )
