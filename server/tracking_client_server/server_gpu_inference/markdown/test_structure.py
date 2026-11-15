"""
Structure and import tests (no heavy dependencies required)
Runs without torch, transformers, etc.
"""

import os
import sys
from pathlib import Path


def test_file_structure():
    """Test that all required files exist."""
    print("\n📁 Testing file structure...")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        "__init__.py",
        "__main__.py",
        "tracking_server_gpu.py",
        "models.py",
        "config.py",
        "bbox_parser.py",
        "gpu_manager.py",
        "tracking_engine.py",
        "api.py",
        "utils.py",
        "tests.py",
        "examples.py",
        "requirements.txt",
        "README.md",
        "MIGRATION.md",
        "ARCHITECTURE.md",
        "SUMMARY.md",
    ]
    
    missing = []
    for file in required_files:
        file_path = base_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ All required files exist")
    return True


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\n🐍 Testing Python syntax...")
    
    base_dir = Path(__file__).parent
    py_files = list(base_dir.glob("*.py"))
    
    errors = []
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file.name, 'exec')
            print(f"  ✓ {py_file.name}")
        except SyntaxError as e:
            print(f"  ✗ {py_file.name}: {e}")
            errors.append((py_file.name, str(e)))
    
    if errors:
        print(f"\n❌ Syntax errors found")
        return False
    
    print("\n✅ All Python files have valid syntax")
    return True


def test_module_docstrings():
    """Test that modules have docstrings."""
    print("\n📝 Testing module docstrings...")
    
    base_dir = Path(__file__).parent
    modules = [
        "models.py",
        "config.py",
        "bbox_parser.py",
        "gpu_manager.py",
        "tracking_engine.py",
        "api.py",
        "utils.py",
    ]
    
    missing = []
    for module in modules:
        module_path = base_dir / module
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if file starts with docstring
            if content.strip().startswith('"""') or content.strip().startswith("'''"):
                print(f"  ✓ {module}")
            else:
                print(f"  ✗ {module} - missing docstring")
                missing.append(module)
    
    if missing:
        print(f"\n⚠️  Some modules missing docstrings")
        return True  # Warning, not failure
    
    print("\n✅ All modules have docstrings")
    return True


def test_imports_without_heavy_deps():
    """Test imports that don't require torch/transformers."""
    print("\n📦 Testing lightweight imports...")
    
    results = []
    
    # Test 1: Config imports
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from server_gpu_inference.config import ServerConfig
        config = ServerConfig()
        print("  ✓ config.ServerConfig")
        results.append(True)
    except Exception as e:
        print(f"  ✗ config.ServerConfig: {e}")
        results.append(False)
    
    # Test 2: BBoxParser imports
    try:
        from server_gpu_inference.bbox_parser import BBoxParser
        parser = BBoxParser()
        print("  ✓ bbox_parser.BBoxParser")
        results.append(True)
    except Exception as e:
        print(f"  ✗ bbox_parser.BBoxParser: {e}")
        results.append(False)
    
    # Test 3: Models imports
    try:
        from server_gpu_inference.models import DetectionResponse
        print("  ✓ models.DetectionResponse")
        results.append(True)
    except Exception as e:
        print(f"  ✗ models.DetectionResponse: {e}")
        results.append(False)
    
    if all(results):
        print("\n✅ All lightweight imports work")
        return True
    else:
        print("\n❌ Some imports failed")
        return False


def test_bbox_parser_logic():
    """Test BBoxParser logic without heavy deps."""
    print("\n🔍 Testing BBoxParser logic...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from server_gpu_inference.bbox_parser import BBoxParser
        
        parser = BBoxParser()
        
        # Test 1
        bbox = parser.extract_bbox("<box>100 200 300 400</box>")
        assert bbox == [100, 200, 300, 400], f"Test 1 failed: {bbox}"
        print("  ✓ Extract from <box> tags")
        
        # Test 2
        bbox = parser.extract_bbox("coords are 150 250 350 450")
        assert bbox == [150, 250, 350, 450], f"Test 2 failed: {bbox}"
        print("  ✓ Extract consecutive numbers")
        
        # Test 3
        bbox = parser.extract_bbox("no coords here")
        assert bbox is None, f"Test 3 failed: {bbox}"
        print("  ✓ Return None when no bbox")
        
        # Test 4
        valid = parser.validate_bbox([10, 20, 30, 40])
        assert valid == True, "Test 4 failed"
        print("  ✓ Validate valid bbox")
        
        # Test 5
        valid = parser.validate_bbox([30, 40, 10, 20])
        assert valid == False, "Test 5 failed"
        print("  ✓ Reject invalid bbox")
        
        print("\n✅ BBoxParser tests passed")
        return True
        
    except Exception as e:
        print(f"\n❌ BBoxParser tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """Test Config validation without heavy deps."""
    print("\n⚙️  Testing Config validation...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from server_gpu_inference.config import ServerConfig
        
        # Test 1: Valid config
        config = ServerConfig(num_gpus=2)
        config.validate()
        print("  ✓ Valid config passes")
        
        # Test 2: Invalid num_gpus (too high)
        try:
            config = ServerConfig(num_gpus=10)
            config.validate()
            print("  ✗ Should reject num_gpus > 8")
            return False
        except ValueError:
            print("  ✓ Reject num_gpus > 8")
        
        # Test 3: Invalid num_gpus (too low)
        try:
            config = ServerConfig(num_gpus=0)
            config.validate()
            print("  ✗ Should reject num_gpus < 1")
            return False
        except ValueError:
            print("  ✓ Reject num_gpus < 1")
        
        # Test 4: Invalid image size
        try:
            config = ServerConfig(max_image_size=-100)
            config.validate()
            print("  ✗ Should reject negative image size")
            return False
        except ValueError:
            print("  ✓ Reject negative image size")
        
        print("\n✅ Config validation tests passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Config tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_metrics():
    """Test code metrics and structure."""
    print("\n📊 Code metrics...")
    
    base_dir = Path(__file__).parent
    
    # Count lines in each module
    modules = {
        "models.py": 0,
        "config.py": 0,
        "bbox_parser.py": 0,
        "gpu_manager.py": 0,
        "tracking_engine.py": 0,
        "api.py": 0,
        "utils.py": 0,
        "tracking_server_gpu.py": 0,
    }
    
    total_lines = 0
    for module in modules:
        module_path = base_dir / module
        if module_path.exists():
            with open(module_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                modules[module] = lines
                total_lines += lines
    
    print(f"\n  Module sizes:")
    for module, lines in sorted(modules.items(), key=lambda x: -x[1]):
        print(f"    {module:30s} {lines:4d} lines")
    
    print(f"\n  Total: {total_lines} lines")
    print(f"  Average: {total_lines // len(modules)} lines per module")
    
    # Check that no module is too large
    max_lines = 300
    large_modules = [m for m, l in modules.items() if l > max_lines]
    if large_modules:
        print(f"\n  ⚠️  Large modules (>{max_lines} lines): {', '.join(large_modules)}")
    else:
        print(f"\n  ✅ All modules under {max_lines} lines")
    
    return True


def main():
    """Run all structure tests."""
    print("=" * 60)
    print("Server GPU Inference - Structure Tests")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Module Docstrings", test_module_docstrings),
        ("Lightweight Imports", test_imports_without_heavy_deps),
        ("BBoxParser Logic", test_bbox_parser_logic),
        ("Config Validation", test_config_validation),
        ("Code Metrics", test_code_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")
    
    print("\n" + "=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("=" * 60)
        return True
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
