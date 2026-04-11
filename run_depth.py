import cv2
import yaml
import argparse
from src.depth_estimator import DepthEstimator
from src.video_pipeline import run_video
from src.overlay_pipeline import run_overlay


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_image(config: dict):
    estimator = DepthEstimator(
        model_name=config["model"]["name"],
        device=config["model"]["device"]
    )

    image_path = config["input"]["image_path"]
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    print(f"[run_depth] Running inference on: {image_path}")
    depth = estimator.predict(frame)
    depth_colored = estimator.colorize(depth, config["output"]["colormap"])

    out_path = config["output"]["image_out"]
    cv2.imwrite(out_path, depth_colored)
    print(f"[run_depth] Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["image", "video", "overlay"],
        default="image",
        help="image: single image | video: depth video | overlay: depth + detections"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "image":
        run_image(config)
    elif args.mode == "video":
        run_video(config)
    elif args.mode == "overlay":
        run_overlay(config)