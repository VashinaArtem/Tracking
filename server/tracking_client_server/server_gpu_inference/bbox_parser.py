"""
Bounding box extraction from model responses
"""

import re
from typing import Optional, List


class BBoxParser:
    """Parser for extracting bounding boxes from model responses."""
    
    @staticmethod
    def extract_bbox(response: str) -> Optional[List[int]]:
        """
        Extract bounding box coordinates from model response.
        
        Tries multiple strategies:
        1. Extract from <box>x1 y1 x2 y2</box> tags
        2. Fallback: extract any 4 consecutive numbers
        
        Args:
            response: Model response text
            
        Returns:
            List of 4 integers [x1, y1, x2, y2] or None if not found
        """
        # Strategy 1: Try <box> tags first
        bbox = BBoxParser._extract_from_box_tags(response)
        if bbox:
            return bbox
        
        # Strategy 2: Fallback - extract any 4 consecutive numbers
        bbox = BBoxParser._extract_consecutive_numbers(response)
        if bbox:
            return bbox
        
        return None
    
    @staticmethod
    def _extract_from_box_tags(response: str) -> Optional[List[int]]:
        """Extract bbox from <box>...</box> tags."""
        match = re.search(r"<box>(.*?)</box>", response, flags=re.DOTALL)
        if match:
            bbox_str = match.group(1)
            nums = re.findall(r"[-+]?\d*\.?\d+", bbox_str)
            if len(nums) >= 4:
                bbox = [int(round(float(nums[i]))) for i in range(4)]
                return bbox
        return None
    
    @staticmethod
    def _extract_consecutive_numbers(response: str) -> Optional[List[int]]:
        """Extract first 4 consecutive numbers from response."""
        nums = re.findall(r"\d+", response)
        if len(nums) >= 4:
            bbox = [int(nums[i]) for i in range(4)]
            return bbox
        return None
    
    @staticmethod
    def validate_bbox(bbox: List[int], min_val: int = 0, max_val: int = 1000) -> bool:
        """
        Validate bounding box coordinates.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            min_val: Minimum valid coordinate value
            max_val: Maximum valid coordinate value
            
        Returns:
            True if valid, False otherwise
        """
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Check if coordinates are within bounds
        if not all(min_val <= coord <= max_val for coord in bbox):
            return False
        
        # Check if box has positive area
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True
