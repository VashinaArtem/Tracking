"""
Test script to verify all module imports work correctly.
"""
import sys

def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing module imports...")
    
    tests = [
        ("config.settings", "TrackingConfig, PIDConfig, ServoConfig, VoiceConfig"),
        ("config.color_params", "ColorCorrectionParams"),
        ("network.server_client", "ServerClient"),
        ("voice.voice_controller", "VoiceController"),
        ("servo.servo_controller", "ServoController"),
        ("tracking.tracker_manager", "TrackerManager"),
        ("tracking.hybrid_tracker", "HybridTracker"),
        ("image.color_correction", "ColorCorrector"),
        ("utils.coordinates", "calculate_bbox_center, normalize_bbox_to_pixels, pixels_to_normalized_bbox"),
        ("utils.metrics", "PerformanceMetrics"),
        ("ui.visualization", "Visualizer"),
    ]
    
    failed = []
    
    for module_name, items in tests:
        try:
            print(f"  Importing {module_name}...", end=" ")
            exec(f"from {module_name} import {items}")
            print("[OK]")
        except Exception as e:
            print("[FAILED]")
            print(f"    Error: {e}")
            failed.append((module_name, e))
    
    print("\n" + "=" * 60)
    if failed:
        print(f"[ERROR] {len(failed)} module(s) failed to import:")
        for module_name, error in failed:
            print(f"  - {module_name}: {error}")
        sys.exit(1)
    else:
        print("[SUCCESS] All modules imported successfully!")
        print("=" * 60)

if __name__ == "__main__":
    test_imports()
