import cv2
import numpy as np


# Color map per class — BGR format
CLASS_COLORS = {
    "car":        (0, 255, 0),
    "truck":      (0, 165, 255),
    "bus":        (0, 0, 255),
    "motorcycle": (255, 0, 255),
    "bicycle":    (255, 255, 0),
    "person":     (255, 0, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(
    frame: np.ndarray,
    detections: list,
    depth_map: np.ndarray = None
) -> np.ndarray:
    """
    Draws YOLO detections on a frame.

    Args:
        frame      : BGR image to draw on (depth colormap frame)
        detections : list of dicts with keys:
                     bbox (x1,y1,x2,y2), label (str), conf (float)
        depth_map  : optional uint8 depth array (H,W) — if provided,
                     shows avg depth value inside each bounding box

    Returns:
        Annotated BGR frame
    """
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Build label text
        text = f"{label} {conf:.2f}"

        # If depth map provided, compute mean depth in the box region
        if depth_map is not None:
            roi = depth_map[y1:y2, x1:x2]
            if roi.size > 0:
                mean_depth = int(roi.mean())
                text += f" d:{mean_depth}"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            output, text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    return output