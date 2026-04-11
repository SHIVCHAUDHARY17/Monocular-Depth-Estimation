import numpy as np
import pytest
from src.depth_estimator import DepthEstimator


@pytest.fixture(scope="module")
def estimator():
    """Load MiDaS Small once for all tests in this module."""
    return DepthEstimator(model_name="midas_small", device="cpu")


def test_estimator_loads(estimator):
    """Model loads without error."""
    assert estimator is not None


def test_predict_returns_uint8(estimator):
    """predict() returns uint8 numpy array."""
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    result = estimator.predict(dummy)
    assert result.dtype == np.uint8


def test_predict_output_shape(estimator):
    """Output depth map matches input spatial dimensions."""
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    result = estimator.predict(dummy)
    assert result.shape == (240, 320)


def test_predict_value_range(estimator):
    """Output values are in valid 0-255 range."""
    dummy = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    result = estimator.predict(dummy)
    assert result.min() >= 0
    assert result.max() <= 255


def test_colorize_output_shape(estimator):
    """Colorized output is 3-channel BGR image."""
    dummy_depth = np.zeros((240, 320), dtype=np.uint8)
    result = estimator.colorize(dummy_depth)
    assert result.shape == (240, 320, 3)


def test_colorize_output_dtype(estimator):
    """Colorized output is uint8."""
    dummy_depth = np.zeros((240, 320), dtype=np.uint8)
    result = estimator.colorize(dummy_depth)
    assert result.dtype == np.uint8


def test_invalid_model_raises():
    """Invalid model name raises ValueError."""
    with pytest.raises(ValueError):
        DepthEstimator(model_name="invalid_model", device="cpu")