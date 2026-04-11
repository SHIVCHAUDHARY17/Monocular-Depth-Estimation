from ultralytics import YOLO
import numpy as np


# COCO class IDs we care about for traffic scenes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    0: "person",
    1: "bicycle",
}


class Detector:
    """
    Wraps YOLOv8 for vehicle detection.
    Returns clean detection dicts for each frame.
    """

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.3, device: str = "cuda"):
        print(f"[Detector] Loading: {weights} on {device}")
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        print(f"[Detector] Ready.")

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Runs YOLO detection on a single BGR frame.
        Returns list of dicts: {bbox, label, conf}
        """
        results = self.model(
            frame_bgr,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                # Only keep vehicle-relevant classes
                if class_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = VEHICLE_CLASSES[class_id]

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "conf": conf
                })

        return detections