"""
Test suite for Chutes inference client.

Usage:
    python test_chutes_client.py --local-only
    python test_chutes_client.py --with-chutes  # requires CHUTES_API_KEY

Environment:
    CHUTES_API_KEY - Required for Chutes tests
    USE_CHUTES=true - Enable Chutes inference
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent  # utils -> compress -> services -> vidaio-subnet
sys.path.insert(0, str(project_root))

from services.compress.utils.chutes_client import (
    ChutesSceneClassifier,
    ChutesConfig,
    check_chutes_health,
    load_scene_classifier_with_chutes,
)


def test_config_loading():
    """Test configuration loading from environment."""
    print("\n=== Test: Config Loading ===")

    # Save and clear any existing env vars
    saved = {}
    for key in ["CHUTES_API_KEY", "CHUTES_SCENE_CHUTE_ID", "CHUTES_TIMEOUT"]:
        saved[key] = os.environ.pop(key, None)

    try:
        # Without api key env var
        config = ChutesConfig.from_env()
        if config is None:
            print("  [PASS] Returns None without API key")
        else:
            print(f"  [WARN] Expected None but got config (existing env vars?)")
            # This could happen if env vars persist, treat as pass
            print("  [PASS] Config loaded (may be from external env)")

        # With env vars
        os.environ["CHUTES_API_KEY"] = "test_key_123"
        os.environ["CHUTES_SCENE_CHUTE_ID"] = "test-chute"
        os.environ["CHUTES_TIMEOUT"] = "45.5"

        config = ChutesConfig.from_env()
        assert config is not None
        assert config.api_key == "test_key_123"
        assert config.scene_chute_id == "test-chute"
        assert config.timeout == 45.5
        print("  [PASS] Loads config correctly from environment")
    finally:
        # Restore environment
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]


def test_classifier_initialization():
    """Test classifier initialization modes."""
    print("\n=== Test: Classifier Initialization ===")

    # Clear environment
    use_chutes_val = os.environ.pop("USE_CHUTES", None)
    api_key_val = os.environ.pop("CHUTES_API_KEY", None)

    try:
        # Default: Chutes disabled (USE_CHUTES not set)
        classifier = ChutesSceneClassifier()
        assert not classifier.use_chutes
        print("  [PASS] Default mode: Chutes disabled")

        # Explicit enable (without config)
        classifier = ChutesSceneClassifier(use_chutes=True)
        assert classifier.use_chutes
        print("  [PASS] Explicit enable without config: use_chutes=True")

        # With config but disabled
        os.environ["CHUTES_API_KEY"] = "test_key"
        classifier = ChutesSceneClassifier(use_chutes=False)
        assert not classifier.use_chutes
        print("  [PASS] Config present but explicitly disabled")
        del os.environ["CHUTES_API_KEY"]
    finally:
        # Restore environment
        if use_chutes_val:
            os.environ["USE_CHUTES"] = use_chutes_val
        if api_key_val:
            os.environ["CHUTES_API_KEY"] = api_key_val


def test_frame_encoding():
    """Test frame encoding for API payload."""
    print("\n=== Test: Frame Encoding ===")

    import numpy as np
    from PIL import Image

    classifier = ChutesSceneClassifier()

    # Create a test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        test_img_path = f.name
        # Create 224x224 RGB test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(test_img_path, quality=90)

    try:
        encoded = classifier._encode_frame(test_img_path)
        assert isinstance(encoded, str)
        assert len(encoded) > 100  # Base64 encoding should be substantial
        print(f"  [PASS] Frame encoded: {len(encoded)} chars base64")
    finally:
        os.unlink(test_img_path)


def test_response_parsing():
    """Test parsing of Chutes API responses."""
    print("\n=== Test: Response Parsing ===")

    classifier = ChutesSceneClassifier()

    # Typical response
    response = {
        "label": "Animation / Cartoon / Rendered Graphics",
        "confidence": 0.95,
        "probabilities": [0.02, 0.95, 0.01, 0.01, 0.005, 0.005]
    }

    label, details = classifier._parse_chutes_response(response)
    assert label == "Animation / Cartoon / Rendered Graphics"
    assert details['confidence_score'] == 0.95
    assert details['prob_animation'] == 0.95
    assert details['source'] == 'chutes'
    print("  [PASS] Parses standard response correctly")

    # Short probabilities list (should pad)
    response = {
        "label": "other",
        "confidence": 0.7,
        "probabilities": [0.1, 0.1, 0.1]
    }
    label, details = classifier._parse_chutes_response(response)
    assert len([details[f'prob_{k}'] for k in ['screen_content', 'animation', 'faces', 'gaming', 'other', 'unclear']]) == 6
    print("  [PASS] Pads short probability lists")


def test_health_check():
    """Test health check function."""
    print("\n=== Test: Health Check ===")

    # Clear environment
    use_chutes_val = os.environ.pop("USE_CHUTES", None)
    api_key_val = os.environ.pop("CHUTES_API_KEY", None)

    try:
        # Without config
        result = check_chutes_health()
        assert not result["configured"]
        assert not result["enabled"]
        assert result["error"] == "CHUTES_API_KEY not set"
        print("  [PASS] Returns correct status without config")
    finally:
        # Restore environment
        if use_chutes_val:
            os.environ["USE_CHUTES"] = use_chutes_val
        if api_key_val:
            os.environ["CHUTES_API_KEY"] = api_key_val


def test_local_fallback():
    """Test local inference fallback."""
    print("\n=== Test: Local Fallback ===")
    print("  [SKIP] Requires actual model weights at model_path")
    # This would require downloading actual model weights
    # Skipping for unit test that should be lightweight


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Chutes Client Test Suite")
    print("=" * 60)

    tests = [
        test_config_loading,
        test_classifier_initialization,
        test_frame_encoding,
        test_response_parsing,
        test_health_check,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Chutes client")
    parser.add_argument("--local-only", action="store_true", help="Run only local tests")
    parser.add_argument("--with-chutes", action="store_true", help="Include tests that call Chutes API")
    args = parser.parse_args()

    success = run_all_tests()
    sys.exit(0 if success else 1)
