import cv2
import numpy as np
import onnxruntime as ort


class ONNXDepthEstimator:
    """
    Runs MiDaS depth estimation using ONNX Runtime.
    No PyTorch required at inference time.

    This is what a production deployment looks like:
    - Model exported once to ONNX
    - Inference via lightweight ONNX Runtime
    - Deployable on embedded targets without full PyTorch stack
    """

    # MiDaS normalization constants (ImageNet mean/std)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    def __init__(self, onnx_path: str, device: str = "cuda"):
        print(f"[ONNXDepthEstimator] Loading: {onnx_path}")

        # Select execution provider
        # CUDAExecutionProvider = GPU via ONNX Runtime
        # CPUExecutionProvider  = CPU fallback
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Read input shape from the ONNX model itself
        input_meta = self.session.get_inputs()[0]
        _, _, self.input_h, self.input_w = input_meta.shape
        self.input_name = input_meta.name

        print(f"[ONNXDepthEstimator] Input shape : {input_meta.shape}")
        print(f"[ONNXDepthEstimator] Provider    : {self.session.get_providers()[0]}")
        print(f"[ONNXDepthEstimator] Ready.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Input : BGR frame (H, W, 3)
        Output: normalized depth map uint8 (H, W)
        """
        original_h, original_w = frame_bgr.shape[:2]

        # Preprocess — must match MiDaS training preprocessing exactly
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model's native input size
        resized = cv2.resize(
            frame_rgb,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_CUBIC
        )

        # Normalize with ImageNet mean and std
        img = resized.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD

        # Convert to (batch, channels, height, width)
        img = np.transpose(img, (2, 0, 1))      # HWC → CHW
        img = np.expand_dims(img, axis=0)        # add batch dim → (1,3,H,W)
        img = img.astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: img})
        depth = outputs[0].squeeze()             # remove batch dim → (H, W)

        # Resize depth back to original frame size
        depth_resized = cv2.resize(
            depth,
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC
        )

        # Normalize to 0-255
        d_min = depth_resized.min()
        d_max = depth_resized.max()
        depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)
        return (depth_norm * 255).astype(np.uint8)

    def colorize(
        self,
        depth_uint8: np.ndarray,
        colormap_name: str = "COLORMAP_INFERNO"
    ) -> np.ndarray:
        colormap = getattr(cv2, colormap_name)
        return cv2.applyColorMap(depth_uint8, colormap)