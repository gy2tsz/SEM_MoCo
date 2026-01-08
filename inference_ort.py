#!/usr/bin/env python3
"""Inference using ONNX Runtime with TensorRT Execution Provider."""

import argparse
import os
import numpy as np
import torch
from PIL import Image
import onnxruntime as ort
from torchvision import transforms
import time
from onnx_optimizer import optimize_onnx_model
import platform


def create_session(
    onnx_model_path,
    use_tensorrt=True,
    graph_optimization_level=None,
    use_quantization=False,
    quantized_model_path=None,
):
    """
    Create ONNX Runtime session with optimizations.

    Args:
        onnx_model_path: Path to ONNX model
        use_tensorrt: Enable TensorRT execution provider
        graph_optimization_level: Optimization level (None, "disabled", "basic", "extended", "all")
        use_quantization: Use quantized model if available
        quantized_model_path: Path to quantized ONNX model

    Returns:
        ONNX Runtime InferenceSession
    """

    print("🚀 Creating ONNX Runtime session...")

    # Setup providers based on platform and availability
    providers = []

    # TensorRT is only available on Linux/Windows with NVIDIA GPUs
    import platform

    if use_tensorrt and platform.system() != "Darwin":  # Not macOS
        try:
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_max_workspace_size": 2147483648,  # 2GB
                        "trt_fp16_enabled": True,  # Enable FP16 for faster inference
                    },
                )
            )
            print("✓ TensorRT provider added")
        except Exception as e:
            print(f"⚠️  TensorRT not available: {e}")
    else:
        if use_tensorrt:
            print("⚠️  TensorRT not available on macOS, using CPU inference")

    # Add CUDA provider as fallback (if available)
    if platform.system() != "Darwin":
        try:
            providers.append("CUDAExecutionProvider")
        except:
            pass

    # Add CPU provider (always available)
    providers.append("CPUExecutionProvider")

    # Set graph optimization level
    if graph_optimization_level is None:
        graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Select model (quantized or original)
    model_path = onnx_model_path
    if use_quantization and quantized_model_path:
        if os.path.exists(quantized_model_path):
            model_path = quantized_model_path
            print(f"📊 Using quantized model: {quantized_model_path}")
        else:
            print(
                f"⚠️  Quantized model not found at {quantized_model_path}, using original model"
            )

    # Create session options with graph optimization
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = graph_optimization_level

    # Enable profiling for performance analysis
    # session_options.enable_profiling = True

    # Set execution mode
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Create session
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
        print(f"✓ Session created with providers: {session.get_providers()}")
        print(f"✓ Graph optimization: {graph_optimization_level}")
    except Exception as e:
        print(f"✗ Error creating session: {e}")
        raise

    return session


def preprocess_image(image_path, image_size=224) -> np.ndarray:
    """Preprocess image for inference."""

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor: torch.Tensor = transform(image)  # type: ignore
    image_np = image_tensor.numpy()

    # Add batch dimension: (C, H, W) -> (B, C, H, W)
    image_np = np.expand_dims(image_np, axis=0)

    return image_np.astype(np.float32)


def run_inference(session, image_path, image_size=224):
    """Run inference on image."""

    # Preprocess image
    image_input = preprocess_image(image_path, image_size)

    # Prepare inputs for ResNet backbone (single image tensor)
    input_name = session.get_inputs()[0].name
    inputs = {input_name: image_input}

    # Run inference
    print(f"⏱️  Running inference on {os.path.basename(image_path)}...")
    start_time = time.time()

    outputs = session.run(None, inputs)

    inference_time = time.time() - start_time
    print(f"✓ Inference time: {inference_time*1000:.2f}ms")

    # ResNet backbone exports a single feature tensor
    features = outputs[0]
    return features, inference_time


def batch_inference(session, image_dir, image_size=224):
    """Run inference on all images in a directory."""

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"✗ No images found in {image_dir}")
        return

    print(f"\n📊 Running inference on {len(image_files)} images...")
    inference_times = []

    for i, image_file in enumerate(image_files[:10], 1):  # Limit to 10 for demo
        image_path = os.path.join(image_dir, image_file)
        try:
            features, inference_time = run_inference(session, image_path, image_size)
            inference_times.append(inference_time)

            print(f"  [{i}] {image_file}: features shape {features.shape}")
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")

    if inference_times:
        avg_time = np.mean(inference_times)
        print(f"\n📈 Average inference time: {avg_time*1000:.2f}ms")
        print(f"📈 Throughput: {1/avg_time:.1f} images/sec")


def main(
    onnx_model_path,
    image_path=None,
    image_dir=None,
    image_size=224,
    use_tensorrt=True,
    graph_optimization_level="all",
    use_quantization=False,
    quantized_model_path=None,
):
    """Main inference function."""

    # Verify ONNX model exists
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    print(f"📦 ONNX model: {onnx_model_path}")

    # Map optimization level string to enum
    optimization_levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    opt_level = optimization_levels.get(
        graph_optimization_level.lower(),
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    )

    # Create ONNX Runtime session
    session = create_session(
        onnx_model_path,
        use_tensorrt=use_tensorrt,
        graph_optimization_level=opt_level,
        use_quantization=use_quantization,
        quantized_model_path=quantized_model_path,
    )

    # Run inference
    if image_path:
        features, inference_time = run_inference(session, image_path, image_size)
        print(f"\n✓ Inference complete")
        if hasattr(features, 'shape'):
            print(f"  Features shape: {features.shape}")  # type: ignore
        else:
            print(f"  Features type: {type(features)}")
    elif image_dir:
        batch_inference(session, image_dir, image_size)
    else:
        print("⚠️  Provide either --image or --image-dir for inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with ONNX Runtime + TensorRT + Graph Optimization"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to single image for inference",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing images for batch inference",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for inference",
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Disable TensorRT and use CUDA/CPU only",
    )
    parser.add_argument(
        "--graph-optimization",
        type=str,
        choices=["disabled", "basic", "extended", "all"],
        default="all",
        help="Graph optimization level (default: all)",
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Enable quantized model inference",
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        default=None,
        help="Path to quantized ONNX model",
    )
    args = parser.parse_args()

    main(
        args.model,
        image_path=args.image,
        image_dir=args.image_dir,
        image_size=args.image_size,
        use_tensorrt=not args.no_tensorrt,
        graph_optimization_level=args.graph_optimization,
        use_quantization=args.use_quantization,
        quantized_model_path=args.quantized_model,
    )
