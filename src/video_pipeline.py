import cv2
import os
from src.depth_estimator import DepthEstimator
from src.profiler import Profiler


def run_video(config: dict):
    """
    Reads input video frame by frame.
    Runs depth estimation on each frame.
    Writes colorized depth map video to output path.
    """
    video_path = config["input"]["video_path"]
    out_path = config["output"]["video_out"]
    colormap_name = config["output"]["colormap"]

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[video_pipeline] Input  : {video_path}")
    print(f"[video_pipeline] Frames : {total_frames} @ {fps:.1f} FPS")
    print(f"[video_pipeline] Size   : {width}x{height}")

    # Read resize config — always resize before inference for performance
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    print(f"[video_pipeline] Processing at : {resize_w}x{resize_h}")

    # Set up output video writer at target resolution
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (resize_w, resize_h))

    # Set up depth estimator and profiler
    estimator = DepthEstimator(
        model_name=config["model"]["name"],
        device=config["model"]["device"]
    )

    profiler = Profiler(
        enabled=config["profiler"]["enabled"],
        log_path=config["profiler"]["log_path"]
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame before inference
        # Decouples pipeline performance from source video resolution
        frame_resized = cv2.resize(frame, (resize_w, resize_h))

        # Time the depth inference only
        profiler.start()
        depth = estimator.predict(frame_resized)
        profiler.stop(frame_idx)

        # Colorize depth map
        depth_colored = estimator.colorize(depth, colormap_name)

        # Write frame to output video
        writer.write(depth_colored)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[video_pipeline] Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"[video_pipeline] Done. Output saved to: {out_path}")

    profiler.summary()
    profiler.save()