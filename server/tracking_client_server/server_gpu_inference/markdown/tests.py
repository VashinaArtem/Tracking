"""
Unit tests for modular tracking server components
Run: python -m server_gpu_inference.tests
"""

import sys
from typing import List


class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def add_pass(self):
        self.passed += 1
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
    
    def print_summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\n❌ Failed tests:")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("✅ All tests passed!")
        print("=" * 60)


def test_bbox_parser(result: TestResult):
    """Test BBoxParser functionality."""
    print("\n📦 Testing BBoxParser...")
    
    from .bbox_parser import BBoxParser
    
    parser = BBoxParser()
    
    # Test 1: Extract from box tags
    try:
        bbox = parser.extract_bbox("<box>100 200 300 400</box>")
        assert bbox == [100, 200, 300, 400], f"Expected [100, 200, 300, 400], got {bbox}"
        print("  ✓ Extract from <box> tags")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_parser_box_tags", str(e))
        print(f"  ✗ Extract from <box> tags: {e}")
    
    # Test 2: Extract consecutive numbers
    try:
        bbox = parser.extract_bbox("The coordinates are 150 250 350 450")
        assert bbox == [150, 250, 350, 450], f"Expected [150, 250, 350, 450], got {bbox}"
        print("  ✓ Extract consecutive numbers")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_parser_consecutive", str(e))
        print(f"  ✗ Extract consecutive numbers: {e}")
    
    # Test 3: Extract floats and round
    try:
        bbox = parser.extract_bbox("<box>10.5 20.7 30.2 40.9</box>")
        assert bbox == [10, 21, 30, 41] or bbox == [11, 21, 30, 41], f"Got {bbox}"
        print("  ✓ Extract and round floats")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_parser_floats", str(e))
        print(f"  ✗ Extract and round floats: {e}")
    
    # Test 4: Validate valid bbox
    try:
        assert parser.validate_bbox([10, 20, 30, 40]) == True
        print("  ✓ Validate valid bbox")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_validate_valid", str(e))
        print(f"  ✗ Validate valid bbox: {e}")
    
    # Test 5: Validate invalid bbox (reversed coords)
    try:
        assert parser.validate_bbox([30, 40, 10, 20]) == False
        print("  ✓ Reject invalid bbox (reversed)")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_validate_invalid", str(e))
        print(f"  ✗ Reject invalid bbox: {e}")
    
    # Test 6: Validate bbox out of bounds
    try:
        assert parser.validate_bbox([-10, 20, 30, 40]) == False
        print("  ✓ Reject out of bounds bbox")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_validate_bounds", str(e))
        print(f"  ✗ Reject out of bounds bbox: {e}")
    
    # Test 7: Return None for no bbox
    try:
        bbox = parser.extract_bbox("No coordinates here")
        assert bbox is None, f"Expected None, got {bbox}"
        print("  ✓ Return None when no bbox found")
        result.add_pass()
    except AssertionError as e:
        result.add_fail("bbox_parser_none", str(e))
        print(f"  ✗ Return None when no bbox found: {e}")


def test_config(result: TestResult):
    """Test ServerConfig."""
    print("\n⚙️  Testing ServerConfig...")
    
    from .config import ServerConfig
    
    # Test 1: Default config
    try:
        config = ServerConfig()
        config.validate()
        print("  ✓ Default config is valid")
        result.add_pass()
    except Exception as e:
        result.add_fail("config_default", str(e))
        print(f"  ✗ Default config: {e}")
    
    # Test 2: Custom valid config
    try:
        config = ServerConfig(num_gpus=2, max_image_size=1024)
        config.validate()
        assert config.num_gpus == 2
        assert config.max_image_size == 1024
        print("  ✓ Custom valid config")
        result.add_pass()
    except Exception as e:
        result.add_fail("config_custom", str(e))
        print(f"  ✗ Custom valid config: {e}")
    
    # Test 3: Invalid num_gpus (too high)
    try:
        config = ServerConfig(num_gpus=10)
        config.validate()
        result.add_fail("config_invalid_gpus", "Should have raised ValueError")
        print("  ✗ Should reject num_gpus > 8")
    except ValueError:
        print("  ✓ Reject invalid num_gpus (>8)")
        result.add_pass()
    
    # Test 4: Invalid num_gpus (too low)
    try:
        config = ServerConfig(num_gpus=0)
        config.validate()
        result.add_fail("config_invalid_gpus_low", "Should have raised ValueError")
        print("  ✗ Should reject num_gpus < 1")
    except ValueError:
        print("  ✓ Reject invalid num_gpus (<1)")
        result.add_pass()
    
    # Test 5: Invalid image size
    try:
        config = ServerConfig(max_image_size=-100)
        config.validate()
        result.add_fail("config_invalid_image_size", "Should have raised ValueError")
        print("  ✗ Should reject negative image size")
    except ValueError:
        print("  ✓ Reject negative image size")
        result.add_pass()
    
    # Test 6: Invalid port
    try:
        config = ServerConfig(port=99999)
        config.validate()
        result.add_fail("config_invalid_port", "Should have raised ValueError")
        print("  ✗ Should reject invalid port")
    except ValueError:
        print("  ✓ Reject invalid port")
        result.add_pass()


def test_gpu_manager(result: TestResult):
    """Test GPUManager utilities."""
    print("\n🖥️  Testing GPUManager...")
    
    from .gpu_manager import GPUManager
    
    # Test 1: Check CUDA availability (should not crash)
    try:
        cuda_available = GPUManager.check_cuda_available()
        assert isinstance(cuda_available, bool)
        print(f"  ✓ Check CUDA available: {cuda_available}")
        result.add_pass()
    except Exception as e:
        result.add_fail("gpu_check_cuda", str(e))
        print(f"  ✗ Check CUDA: {e}")
    
    # Test 2: Get GPU count
    try:
        gpu_count = GPUManager.get_gpu_count()
        assert isinstance(gpu_count, int)
        assert gpu_count >= 0
        print(f"  ✓ Get GPU count: {gpu_count}")
        result.add_pass()
    except Exception as e:
        result.add_fail("gpu_count", str(e))
        print(f"  ✗ Get GPU count: {e}")
    
    # Test 3: Validate GPU count (valid)
    try:
        actual = GPUManager.validate_gpu_count(2, 4)
        assert actual == 2
        print("  ✓ Validate GPU count (within range)")
        result.add_pass()
    except Exception as e:
        result.add_fail("gpu_validate_valid", str(e))
        print(f"  ✗ Validate GPU count: {e}")
    
    # Test 4: Validate GPU count (adjust down)
    try:
        actual = GPUManager.validate_gpu_count(8, 4)
        assert actual == 4
        print("  ✓ Adjust GPU count when exceeds available")
        result.add_pass()
    except Exception as e:
        result.add_fail("gpu_validate_adjust", str(e))
        print(f"  ✗ Adjust GPU count: {e}")
    
    # Test 5: Get device and dtype
    try:
        device, dtype = GPUManager.get_device_and_dtype(use_gpu=False)
        assert device == "cpu"
        print("  ✓ Get device and dtype (CPU)")
        result.add_pass()
    except Exception as e:
        result.add_fail("gpu_device_dtype", str(e))
        print(f"  ✗ Get device and dtype: {e}")


def test_utils(result: TestResult):
    """Test utility functions."""
    print("\n🔧 Testing Utils...")
    
    from .utils import format_grounding_question, ensure_rgb
    from PIL import Image
    
    # Test 1: Format grounding question
    try:
        question = format_grounding_question("red car")
        assert "<ref>red car</ref>" in question
        assert "<box>" in question
        print("  ✓ Format grounding question")
        result.add_pass()
    except Exception as e:
        result.add_fail("utils_format_question", str(e))
        print(f"  ✗ Format grounding question: {e}")
    
    # Test 2: Ensure RGB (already RGB)
    try:
        image = Image.new("RGB", (100, 100), color="red")
        rgb_image = ensure_rgb(image)
        assert rgb_image.mode == "RGB"
        print("  ✓ Ensure RGB (already RGB)")
        result.add_pass()
    except Exception as e:
        result.add_fail("utils_ensure_rgb", str(e))
        print(f"  ✗ Ensure RGB: {e}")
    
    # Test 3: Ensure RGB (convert from L)
    try:
        image = Image.new("L", (100, 100), color=128)
        rgb_image = ensure_rgb(image)
        assert rgb_image.mode == "RGB"
        print("  ✓ Ensure RGB (convert from L)")
        result.add_pass()
    except Exception as e:
        result.add_fail("utils_ensure_rgb_convert", str(e))
        print(f"  ✗ Ensure RGB convert: {e}")


def test_models(result: TestResult):
    """Test Pydantic models."""
    print("\n📋 Testing Models...")
    
    from .models import DetectionResponse, ServerStats
    
    # Test 1: DetectionResponse success
    try:
        response = DetectionResponse(
            success=True,
            bbox=[10, 20, 30, 40],
            inference_time=0.5,
            prompt="test"
        )
        assert response.success == True
        assert response.bbox == [10, 20, 30, 40]
        print("  ✓ DetectionResponse (success)")
        result.add_pass()
    except Exception as e:
        result.add_fail("models_detection_success", str(e))
        print(f"  ✗ DetectionResponse success: {e}")
    
    # Test 2: DetectionResponse failure
    try:
        response = DetectionResponse(
            success=False,
            error="Not found",
            inference_time=0.3,
            prompt="test"
        )
        assert response.success == False
        assert response.error == "Not found"
        assert response.bbox is None
        print("  ✓ DetectionResponse (failure)")
        result.add_pass()
    except Exception as e:
        result.add_fail("models_detection_failure", str(e))
        print(f"  ✗ DetectionResponse failure: {e}")
    
    # Test 3: ServerStats
    try:
        stats = ServerStats(
            total_requests=100,
            successful_detections=95,
            success_rate="95.0%",
            avg_inference_time="0.50s",
            device="CUDA",
            dtype="bfloat16",
            multi_gpu_mode="single"
        )
        assert stats.total_requests == 100
        assert stats.success_rate == "95.0%"
        print("  ✓ ServerStats")
        result.add_pass()
    except Exception as e:
        result.add_fail("models_server_stats", str(e))
        print(f"  ✗ ServerStats: {e}")


def test_imports(result: TestResult):
    """Test that all imports work."""
    print("\n📦 Testing Imports...")
    
    # Test 1: Main package imports
    try:
        from server_gpu_inference import (
            DetectionResponse,
            TrackingServerGPU,
            ServerConfig,
            create_app
        )
        print("  ✓ Main package imports")
        result.add_pass()
    except ImportError as e:
        result.add_fail("imports_main", str(e))
        print(f"  ✗ Main package imports: {e}")
    
    # Test 2: Individual module imports
    try:
        from .models import DetectionResponse, ServerStats
        from .config import ServerConfig
        from .bbox_parser import BBoxParser
        from .gpu_manager import GPUManager
        from .tracking_engine import TrackingServerGPU
        from .api import create_app
        from .utils import format_grounding_question
        print("  ✓ Individual module imports")
        result.add_pass()
    except ImportError as e:
        result.add_fail("imports_modules", str(e))
        print(f"  ✗ Individual module imports: {e}")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Running Modular Tracking Server Tests")
    print("=" * 60)
    
    result = TestResult()
    
    # Run test suites
    test_imports(result)
    test_bbox_parser(result)
    test_config(result)
    test_gpu_manager(result)
    test_utils(result)
    test_models(result)
    
    # Print summary
    result.print_summary()
    
    return result.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
