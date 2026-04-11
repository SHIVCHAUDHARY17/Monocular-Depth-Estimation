import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import torch
import cv2


# ── helpers ───────────────────────────────────────────────────────────────────

def make_fake_depth_output(h=240, w=320):
    """Simulate raw model output as a torch tensor."""
    return torch.rand(h, w) * 10.0  # random relative depth values


# ── mock estimator fixture ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_estimator():
    """
    Creates a DepthEstimator with model loading mocked out.
    This allows CI to test all logic without downloading weights.
    """
    with patch("torch.hub.load") as mock_hub:
        # Mock the model itself
        fake_model = MagicMock()
        fake_model.return_value = make_fake_depth_output()

        # Mock the transforms
        fake_transforms = MagicMock()
        fake_transform_fn = MagicMock()
        fake_transform_fn.return_value = torch.rand(1, 3, 256, 256)
        fake_transforms.small_transform = fake_transform_fn

        # torch.hub.load returns model on first call, transforms on second
        mock_hub.side_effect = [fake_model, fake_transforms]

        from src.depth_estimator import DepthEstimator
        estimator = DepthEstimator(model_name="midas_small", device="cpu")

        # Override predict to return a controlled depth map
        # so we can test downstream logic cleanly
        estimator._predict_midas = lambda frame: estimator._normalize(
            np.random.rand(frame.shape[0], frame.shape[1]).astype(np.float32) * 10
        )
        estimator.predict = estimator._predict_midas

        yield estimator


# ── normalize unit tests (no model needed) ────────────────────────────────────

def test_normalize_returns_uint8():
    """_normalize converts float array to uint8."""
    from src.depth_estimator import DepthEstimator
    with patch("torch.hub.load"):
        est = DepthEstimator.__new__(DepthEstimator)
        est.is_depth_anything = False
        raw = np.random.rand(240, 320).astype(np.float32) * 100
        result = est._normalize(raw)
        assert result.dtype == np.uint8


def test_normalize_range():
    """_normalize output stays within 0-255."""
    from src.depth_estimator import DepthEstimator
    est = DepthEstimator.__new__(DepthEstimator)
    est.is_depth_anything = False
    raw = np.random.rand(240, 320).astype(np.float32) * 100
    result = est._normalize(raw)
    assert result.min() >= 0
    assert result.max() <= 255


def test_normalize_uniform_input():
    """_normalize handles uniform input (no division by zero)."""
    from src.depth_estimator import DepthEstimator
    est = DepthEstimator.__new__(DepthEstimator)
    est.is_depth_anything = False
    raw = np.ones((240, 320), dtype=np.float32) * 5.0
    result = est._normalize(raw)
    assert result.dtype == np.uint8


# ── colorize unit tests (no model needed) ─────────────────────────────────────

def test_colorize_output_shape():
    """colorize returns 3-channel image."""
    from src.depth_estimator import DepthEstimator
    est = DepthEstimator.__new__(DepthEstimator)
    depth = np.zeros((240, 320), dtype=np.uint8)
    result = est.colorize(depth)
    assert result.shape == (240, 320, 3)


def test_colorize_output_dtype():
    """colorize returns uint8."""
    from src.depth_estimator import DepthEstimator
    est = DepthEstimator.__new__(DepthEstimator)
    depth = np.zeros((240, 320), dtype=np.uint8)
    result = est.colorize(depth)
    assert result.dtype == np.uint8


def test_colorize_different_colormaps():
    """colorize works with different valid OpenCV colormaps."""
    from src.depth_estimator import DepthEstimator