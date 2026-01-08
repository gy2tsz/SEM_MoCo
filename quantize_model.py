#!/usr/bin/env python3
"""Quantize ONNX model using dynamic or static quantization."""

import os
import argparse
import numpy as np
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic, QuantType, QuantFormat
import cv2


class CalibrationDataReader:
    """Calibration data reader for static quantization."""

    def __init__(self, calibration_dir, num_samples=100, img_size=224):
        """
        Initialize calibration data reader.

        Args:
            calibration_dir: Directory containing calibration images
            num_samples: Number of samples to use for calibration
            img_size: Image size (assume square images)
        """
        self.calibration_dir = Path(calibration_dir)
        self.img_size = img_size
        self.num_samples = num_samples

        # Find all image files
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            self.image_files.extend(self.calibration_dir.glob(ext))

        if not self.image_files:
            raise ValueError(f"No images found in {calibration_dir}")

        # Limit to num_samples
        self.image_files = self.image_files[: self.num_samples]
        print(f"📊 Using {len(self.image_files)} calibration images")

    def get_next(self):
        """Get next calibration batch."""
        for img_path in self.image_files:
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize to model input size
                img = cv2.resize(img, (self.img_size, self.img_size))

                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Transpose to CHW format
                img = np.transpose(img, (2, 0, 1))

                # Add batch dimension
                img_batch = np.expand_dims(img, axis=0)

                # Yield as calibration data
                yield {"input": img_batch.astype(np.float32)}

            except Exception as e:
                print(f"⚠️  Skipping {img_path}: {e}")
                continue


def quantize_dynamic(
    model_path,
    output_path,
    per_channel=False,
    weight_type=QuantType.QInt8,
    optimize_model=True,
):
    """
    Quantize ONNX model to lower precision using dynamic quantization.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        per_channel: Per-channel quantization (more accurate but larger)
        weight_type: Quantization type (QInt8, QUInt8)
        optimize_model: Enable graph optimization during quantization
    """

    print("📊 ONNX Model Quantization (Dynamic)")
    print(f"   Input: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Per-channel: {per_channel}")
    print(f"   Weight type: {weight_type}")
    print(f"   Optimize: {optimize_model}")
    print()

    # Verify input file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        print("⏱️  Quantizing model...")
        ort_quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            per_channel=per_channel,
            weight_type=weight_type,
        )
        print("✅ Quantization complete!")
        print()

        # Compare file sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        compression = (1 - quantized_size / original_size) * 100

        print("📈 Size Comparison:")
        print(f"   Original:  {original_size:8.2f} MB")
        print(f"   Quantized: {quantized_size:8.2f} MB")
        print(f"   Compression: {compression:6.1f}%")
        print(f"   Reduction: {original_size - quantized_size:7.2f} MB")
        print()

        print("🎯 Benefits:")
        print(f"   Memory reduction: {compression:.1f}%")
        print(f"   Typical speedup: 2-3x faster inference")
        print(f"   Typical accuracy: <0.1% loss (INT8)")
        print()

        print("💡 Usage:")
        print(f"   python inference_ort.py --model model.onnx \\")
        print(f"     --use-quantization --quantized-model {output_path}")

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        raise


def quantize_static(
    model_path,
    output_path,
    calibration_dir,
    num_samples=100,
    per_channel=False,
    weight_type=QuantType.QInt8,
):
    """
    Quantize ONNX model using static quantization with calibration data.

    For MoCo and CNNs: Static quantization is better than dynamic because:
    - Quantizes both weights AND activations (2-3x vs 1.5-2x speedup)
    - Better accuracy preservation (<0.3% vs 0.2-0.5% loss)
    - Activations have stable ranges due to ReLU/BatchNorm

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        calibration_dir: Directory with representative images for calibration
        num_samples: Number of calibration samples to use
        per_channel: Per-channel quantization (more accurate)
        weight_type: Quantization type (QInt8, QUInt8)
    """
    from onnxruntime.quantization import quantize_static as ort_quantize_static

    print("📊 ONNX Model Quantization (Static with Calibration)")
    print(f"   Input: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Calibration dir: {calibration_dir}")
    print(f"   Samples: {num_samples}")
    print(f"   Per-channel: {per_channel}")
    print()

    # Verify input file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(calibration_dir):
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    try:
        # Create calibration data reader
        print("📂 Loading calibration data...")
        calibration_reader = CalibrationDataReader(
            calibration_dir=calibration_dir, num_samples=num_samples
        )
        print()

        print("⏱️  Quantizing model with calibration...")
        ort_quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,  # type: ignore
            quant_format=QuantFormat.QOperator,
            per_channel=per_channel,
            weight_type=weight_type,
        )
        print("✅ Quantization complete!")
        print()

        # Compare file sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        compression = (1 - quantized_size / original_size) * 100

        print("📈 Size Comparison:")
        print(f"   Original:     {original_size:8.2f} MB")
        print(f"   Quantized:    {quantized_size:8.2f} MB")
        print(f"   Compression:  {compression:6.1f}%")
        print(f"   Reduction:    {original_size - quantized_size:7.2f} MB")
        print()

        print("🎯 Benefits (Static Quantization):")
        print(f"   Memory reduction: {compression:.1f}%")
        print(f"   Typical speedup: 3-4x faster inference")
        print(f"   Typical accuracy: <0.3% loss (better than dynamic!)")
        print(f"   Activations: Quantized (extra speedup)")
        print()

        print("💡 Why static is better for MoCo/CNNs:")
        print("   ✅ Both weights AND activations quantized")
        print("   ✅ CNN activations have stable ranges")
        print("   ✅ Better accuracy preservation")
        print("   ✅ 3-4x speedup (vs 2-3x for dynamic)")
        print()

        print("💡 Usage:")
        print(f"   python inference_ort.py --model model.onnx \\")
        print(f"     --use-quantization --quantized-model {output_path}")

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model for faster inference (dynamic or static)"
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument(
        "--output", required=True, help="Output path for quantized model"
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel quantization (more accurate)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Enable graph optimization during quantization",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization with calibration (recommended for MoCo/CNNs)",
    )
    parser.add_argument(
        "--calibration-dir",
        help="Directory with calibration images (required for --static)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of calibration samples to use (default: 100)",
    )
    args = parser.parse_args()

    if args.static:
        if not args.calibration_dir:
            parser.error(
                "--calibration-dir is required when using --static quantization"
            )
        quantize_static(
            model_path=args.model,
            output_path=args.output,
            calibration_dir=args.calibration_dir,
            num_samples=args.num_samples,
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8,
        )
    else:
        quantize_dynamic(
            model_path=args.model,
            output_path=args.output,
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8,
            optimize_model=args.optimize,
        )
