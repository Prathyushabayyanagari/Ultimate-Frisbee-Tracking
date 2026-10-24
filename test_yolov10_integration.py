#!/usr/bin/env python3
"""
Simple test to check YOLOv10 integration without full environment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test if YOLOv10 modules can be imported."""
    try:
        from yolov10.config import YOLOv10Config

        print("✓ YOLOv10Config import successful")

        config = YOLOv10Config(model_variant="yolov10s")
        print("✓ YOLOv10Config creation successful")

        config_dict = config.to_dict()
        print(f"✓ Config: {config_dict['model_variant']}")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_detection_adapter():
    """Test DetectionAdapter creation."""
    try:
        from yolov10.detection_adapter import DetectionAdapter

        adapter = DetectionAdapter(
            target_classes=["person", "sports ball"], confidence_threshold=0.25
        )
        print("✓ DetectionAdapter creation successful")

        info = adapter.get_info()
        print(f"✓ Adapter info: {len(info)} fields")

        return True
    except Exception as e:
        print(f"✗ DetectionAdapter failed: {e}")
        return False


def test_object_tracker_structure():
    """Test if object_tracker.py can be parsed."""
    try:
        # Try to compile object_tracker.py to check syntax
        import py_compile

        py_compile.compile("object_tracker.py", doraise=True)
        print("✓ object_tracker.py syntax check passed")
        return True
    except Exception as e:
        print(f"✗ object_tracker.py syntax error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing YOLOv10 Migration...")
    print("=" * 40)

    tests = [
        test_imports,
        test_detection_adapter,
        test_object_tracker_structure,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print("Test failed!")

    print(f"\n{'=' * 40}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! YOLOv10 integration looks good.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
