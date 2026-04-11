import cv2
import yaml
import time
import argparse
from src.exporter import export_midas_to_onnx
from src.onnx_estimator import ONNXDepthEstimator
from src.depth_estimator import DepthEstimator


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def benchmark_pytorch(config: dict, num_frames: int = 50) -> dict:
    """Benchmark PyTorch MiDaS Small inference."""
    print("\n[Benchmark] PyTorch MiDaS Small")

    estimator = DepthEstimator(
        model_name="midas_small",
        device=config["model"]["device"]
    )

    cap = cv2.VideoCapture(config["input"]["video_path"])
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    times = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frame = cv2.resize(frame, (resize_w, resize_h))
        t = time.perf_counter()
        estimator.predict(frame)
        times.append(time.perf_counter() - t)

    cap.release()
    avg_ms = sum(times) / len(times) * 1000
    avg_fps = 1000 / avg_ms
    print(f"[Benchmark] Avg: {avg_ms:.2f} ms | {avg_fps:.2f} FPS")
    return {"backend": "PyTorch", "avg_ms": round(avg_ms, 2), "avg_fps": round(avg_fps, 2)}


def benchmark_onnx(config: dict, onnx_path: str, num_frames: int = 50) -> dict:
    """Benchmark ONNX Runtime MiDaS Small inference."""
    print("\n[Benchmark] ONNX Runtime MiDaS Small")

    estimator = ONNXDepthEstimator(
        onnx_path=onnx_path,
        device=config["model"]["device"]
    )

    cap = cv2.VideoCapture(config["input"]["video_path"])
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    times = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frame = cv2.resize(frame, (resize_w, resize_h))
        t = time.perf_counter()
        estimator.predict(frame)
        times.append(time.perf_counter() - t)

    cap.release()
    avg_ms = sum(times) / len(times) * 1000
    avg_fps = 1000 / avg_ms
    print(f"[Benchmark] Avg: {avg_ms:.2f} ms | {avg_fps:.2f} FPS")
    return {"backend": "ONNX Runtime", "avg_ms": round(avg_ms, 2), "avg_fps": round(avg_fps, 2)}


def print_table(results: list):
    print("\n" + "="*50)
    print(f"{'Backend':<20} {'Avg Inference':>15} {'Avg FPS':>10}")
    print("="*50)
    for r in results:
        print(f"{r['backend']:<20} {str(r['avg_ms']) + ' ms':>15} {r['avg_fps']:>10}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export if .onnx already exists"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    onnx_path = "outputs/midas_small.onnx"

    # Step 1: Export
    if not args.skip_export:
        export_midas_to_onnx(
            model_name="midas_small",
            output_path=onnx_path,
            device=config["model"]["device"]
        )
    else:
        print(f"[run_onnx] Skipping export, using: {onnx_path}")

    # Step 2: Benchmark both backends
    results = []
    results.append(benchmark_pytorch(config, args.frames))
    results.append(benchmark_onnx(config, onnx_path, args.frames))

    # Step 3: Print comparison table
    print_table(results)