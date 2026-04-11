import torch
import numpy as np


def export_midas_to_onnx(
    model_name: str = "midas_small",
    output_path: str = "outputs/midas_small.onnx",
    input_width: int = 1280,
    input_height: int = 720,
    device: str = "cuda"
):
    """
    Exports a MiDaS model to ONNX format.

    Why ONNX:
    - Framework-independent inference
    - Runs on ONNX Runtime (lighter than PyTorch)
    - Deployable on embedded targets: Jetson, automotive SOCs, edge devices
    - Required step for production perception pipelines

    Args:
        model_name   : midas_small or midas_large
        output_path  : where to save the .onnx file
        input_width  : expected input frame width
        input_height : expected input frame height
        device       : cuda or cpu
    """
    import torch

    dev = torch.device(device)

    print(f"[Exporter] Loading {model_name} for ONNX export...")

    if model_name == "midas_small":
        model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", pretrained=True
        )
        # MiDaS small internal input size is 256x256
        # We export at the model's native resolution
        # The resize happens before this in the pipeline
        onnx_input_h, onnx_input_w = 256, 256

    elif model_name == "midas_large":
        model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Large", pretrained=True
        )
        onnx_input_h, onnx_input_w = 384, 384

    else:
        raise ValueError(f"ONNX export not supported for: {model_name}")

    model.to(dev)
    model.eval()

    # Create a dummy input tensor matching the model's expected input shape
    # MiDaS expects (batch, channels, height, width) normalized float32
    dummy_input = torch.randn(1, 3, onnx_input_h, onnx_input_w).to(dev)

    print(f"[Exporter] Exporting to: {output_path}")
    print(f"[Exporter] Input shape : {dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,           # opset 11 is widely supported
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={              # allow variable batch size
            "input":  {0: "batch"},
            "output": {0: "batch"}
        }
    )

    print(f"[Exporter] Export complete: {output_path}")
    return output_path