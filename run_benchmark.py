import cv2
import yaml
import time
import argparse
from src.depth_estimator import DepthEstimator


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def benchmark_model(model_name: str, config: dict, num_frames: int = 100):
    """
    Runs a model on num_frames from the video.
    Returns average inference time and FPS.
    """
    print(f"\n[Benchmark] Testing: {model_name}")
    print(f"[Benchmark] Frames : {num_frames}")

    estimator = DepthEstimator(
        model_name=model_name,
        device=config["model"]["device"]
    )

    video_path = config["input"]["video_path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    times = []
    frame_idx = 0

    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            # Loop video if shorter than num_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_resized = cv2.resize(frame, (resize_w, resize_h))

        start = time.perf_counter()
        depth = estimator.predict(frame_resized)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        frame_idx += 1

    cap.release()

    avg_ms = (sum(times) / len(times)) * 1000
    avg_fps = 1000 / avg_ms

    print(f"[Benchmark] Avg inference : {avg_ms:.2f} ms")
    print(f"[Benchmark] Avg FPS       : {avg_fps:.2f}")

    return {
        "model": model_name,
        "avg_ms": round(avg_ms, 2),
        "avg_fps": round(avg_fps, 2)
    }


def print_table(results: list):
    print("\n" + "="*55)
    print(f"{'Model':<20} {'Avg Inference':>15} {'Avg FPS':>10}")
    print("="*55)
    for r in results:
        print(f"{r['model']:<20} {str(r['avg_ms']) + ' ms':>15} {r['avg_fps']:>10}")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to benchmark per model"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    models = ["midas_small", "midas_large", "depth_anything"]
    results = []

    for model_name in models:
        result = benchmark_model(model_name, config, num_frames=args.frames)
        if result:
            results.append(result)

    print_table(results)