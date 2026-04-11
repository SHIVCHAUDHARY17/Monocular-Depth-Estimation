import torch
import cv2
import numpy as np


class DepthEstimator:
    """
    Supports three depth estimation models:
    - midas_small    : MiDaS v2.1 Small (fastest)
    - midas_large    : MiDaS DPT Large (slower, more accurate)
    - depth_anything : Depth Anything V2 Small (best quality/speed balance)
    """

    def __init__(self, model_name: str = "midas_small", device: str = "cuda"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.is_depth_anything = (model_name == "depth_anything")

        print(f"[DepthEstimator] Loading: {model_name} on {device}")

        if model_name == "midas_small":
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True
            )
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.small_transform

        elif model_name == "midas_large":
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "DPT_Large", pretrained=True, trust_repo=True
            )
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.dpt_transform

        elif model_name == "depth_anything":
            from transformers import pipeline
            self.pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if device == "cuda" else -1
            )
            self.model = None
            self.transform = None

        else:
            raise ValueError(f"Unknown model: {model_name}")

        if not self.is_depth_anything:
            self.model.to(self.device)
            self.model.eval()

        print(f"[DepthEstimator] Ready.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Input : BGR frame (H, W, 3)
        Output: normalized depth map uint8 (H, W) values 0-255
        """
        if self.is_depth_anything:
            return self._predict_depth_anything(frame_bgr)
        else:
            return self._predict_midas(frame_bgr)

    def _predict_midas(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        return self._normalize(depth)

    def _predict_depth_anything(self, frame_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        result = self.pipe(pil_image)
        depth = np.array(result["depth"], dtype=np.float32)
        return self._normalize(depth)

    def _normalize(self, depth: np.ndarray) -> np.ndarray:
        """Normalize raw depth values to 0-255 uint8."""
        depth_min = depth.min()
        depth_max = depth.max()
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        return (depth_norm * 255).astype(np.uint8)

    def colorize(
        self,
        depth_uint8: np.ndarray,
        colormap_name: str = "COLORMAP_INFERNO"
    ) -> np.ndarray:
        colormap = getattr(cv2, colormap_name)
        return cv2.applyColorMap(depth_uint8, colormap)