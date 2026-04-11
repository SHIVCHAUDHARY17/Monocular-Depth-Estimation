import numpy as np
import pytest
from src.annotator import draw_detections


def make_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_no_detections_returns_frame():
    """Empty detections returns a frame of same shape."""
    frame = make_frame()
    result = draw_detections(frame, [])
    assert result.shape == frame.shape


def test_single_detection_draws():
    """Single detection returns annotated frame without error."""
    frame = make_frame()
    detections = [{"bbox": (100, 100, 300, 300), "label": "car", "conf": 0.85}]
    result = draw_detections(frame, detections)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_detection_with_depth_map():
    """Detection with depth map attaches depth value without error."""
    frame = make_frame()
    depth = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
    detections = [{"bbox": (100, 100, 300, 300), "label": "truck", "conf": 0.75}]
    result = draw_detections(frame, detections, depth_map=depth)
    assert result.shape == frame.shape


def test_multiple_detections():
    """Multiple detections all drawn without error."""
    frame = make_frame()
    detections = [
        {"bbox": (50, 50, 200, 200), "label": "car", "conf": 0.9},
        {"bbox": (400, 100, 600, 400), "label": "bus", "conf": 0.7},
        {"bbox": (800, 200, 1000, 500), "label": "person", "conf": 0.6},
    ]
    result = draw_detections(frame, detections)
    assert result.shape == frame.shape


def test_unknown_class_uses_default_color():
    """Unknown class label does not crash."""
    frame = make_frame()
    detections = [{"bbox": (100, 100, 300, 300), "label": "spaceship", "conf": 0.5}]
    result = draw_detections(frame, detections)
    assert result.shape == frame.shape