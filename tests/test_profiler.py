import os
import pytest
from src.profiler import Profiler


def test_profiler_records_frames(tmp_path):
    """Profiler records correct number of frames."""
    log = str(tmp_path / "test_log.csv")
    p = Profiler(enabled=True, log_path=log)
    for i in range(5):
        p.start()
        p.stop(i)
    assert len(p.records) == 5


def test_profiler_saves_csv(tmp_path):
    """Profiler saves CSV file with correct headers."""
    log = str(tmp_path / "test_log.csv")
    p = Profiler(enabled=True, log_path=log)
    p.start()
    p.stop(0)
    p.save()
    assert os.path.exists(log)
    with open(log) as f:
        headers = f.readline().strip()
    assert "frame" in headers
    assert "inference_ms" in headers
    assert "fps" in headers


def test_profiler_disabled_records_nothing():
    """Disabled profiler records nothing."""
    p = Profiler(enabled=False, log_path="outputs/dummy.csv")
    p.start()
    p.stop(0)
    assert len(p.records) == 0


def test_profiler_fps_positive(tmp_path):
    """Recorded FPS values are positive."""
    log = str(tmp_path / "test_log.csv")
    p = Profiler(enabled=True, log_path=log)
    import time
    p.start()
    time.sleep(0.01)
    p.stop(0)
    assert p.records[0]["fps"] > 0


def test_profiler_inference_ms_positive(tmp_path):
    """Recorded inference_ms values are positive."""
    log = str(tmp_path / "test_log.csv")
    p = Profiler(enabled=True, log_path=log)
    import time
    p.start()
    time.sleep(0.01)
    p.stop(0)
    assert p.records[0]["inference_ms"] > 0