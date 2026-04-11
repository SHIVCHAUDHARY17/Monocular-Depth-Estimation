import time
import csv
import os


class Profiler:
    """
    Tracks per-frame inference time and saves a CSV log.
    """

    def __init__(self, enabled: bool = True, log_path: str = "outputs/profiler_log.csv"):
        self.enabled = enabled
        self.log_path = log_path
        self.records = []
        self._start = None

    def start(self):
        if self.enabled:
            self._start = time.perf_counter()

    def stop(self, frame_idx: int):
        if self.enabled and self._start is not None:
            elapsed = time.perf_counter() - self._start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.records.append({
                "frame": frame_idx,
                "inference_ms": round(elapsed * 1000, 2),
                "fps": round(fps, 2)
            })

    def summary(self):
        if not self.records:
            return
        avg_ms = sum(r["inference_ms"] for r in self.records) / len(self.records)
        avg_fps = sum(r["fps"] for r in self.records) / len(self.records)
        print(f"[Profiler] Frames processed : {len(self.records)}")
        print(f"[Profiler] Avg inference    : {avg_ms:.2f} ms")
        print(f"[Profiler] Avg FPS          : {avg_fps:.2f}")

    def save(self):
        if not self.enabled or not self.records:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "inference_ms", "fps"])
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[Profiler] Log saved to: {self.log_path}")