import cv2
import os
from src.depth_estimator import DepthEstimator
from src.detector import Detector
from src.annotator import draw_detections
from src.profiler import Profiler


def run_overlay(config: dict):
    """
    Runs depth estimation + YOLO detection on each frame.
    Draws detections on top of the depth colormap.
    Outputs an annotated video showing depth context per vehicle.
    """
    video_path = config["input"]["video_path"]
    out_path = config["output"]["overlay_out"]
    colormap_name = config["output"]["colormap"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[overlay_pipeline] Input  : {video_path}")
    print(f"[overlay_pipeline] Frames : {total_frames} @ {fps:.1f} FPS")
    print(f"[overlay_pipeline] Output : {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (resize_w, resize_h))

    # Load depth estimator
    estimator = DepthEstimator(
        model_name=config["model"]["name"],
        device=config["model"]["device"]
    )

    # Load YOLO detector
    detector = Detector(
        weights=config["detector"]["weights"],
        confidence=config["detector"]["confidence"],
        device=config["model"]["device"]
    )

    profiler = Profiler(
        enabled=config["profiler"]["enabled"],
        log_path=config["profiler"]["log_path"].replace(".csv", "_overlay.csv")
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (resize_w, resize_h))

        profiler.start()

        # Step 1: depth estimation
        depth = estimator.predict(frame_resized)

        # Step 2: colorize depth map
        depth_colored = estimator.colorize(depth, colormap_name)

        # Step 3: YOLO detection on original resized frame
        detections = detector.detect(frame_resized)

        # Step 4: draw detections on depth colored frame
        # pass raw depth map so we can show per-box depth value
        annotated = draw_detections(depth_colored, detections, depth_map=depth)

        profiler.stop(frame_idx)

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[overlay_pipeline] Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"[overlay_pipeline] Done. Saved to: {out_path}")
    profiler.summary()
    profiler.save()