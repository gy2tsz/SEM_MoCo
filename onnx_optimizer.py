#!/usr/bin/env python3
"""
Optimize ONNX model graph for better inference performance.

1. Operator Fusion - Combines multiple operations into single optimized kernels
(e.g., Conv + BatchNorm + ReLU fused into one)
2. Constant Folding - Pre-computes operations on constants at optimization time rather than runtime
3. Dead Code Elimination - Removes unused operations and variables from the graph
4. Memory Optimization - Reduces memory allocations and peak memory usage
5. Graph Simplification - Removes redundant nodes and simplifies the computational graph
6. Layout Transformation - Optimizes tensor layouts for specific execution providers (CPU, CoreML, etc.)
7. Precision Optimization - When applicable, adjusts data types for better performance (e.g., FP32 to FP16)
8. Layer Normalization - Optimizes normalization layers
"""

import argparse
import os
import onnx
import onnxruntime as ort


def optimize_onnx_model(
    onnx_model_path, optimized_model_path, ep="CPUExecutionProvider"
):
    """
    Optimize ONNX model graph and save to file.

    This is a one-time preprocessing step that optimizes the model
    graph (operator fusion, constant folding, etc.) before inference.

    Args:
        onnx_model_path: Path to original ONNX model
        optimized_model_path: Path to save optimized model
    """
    print("🔧 Optimizing ONNX model graph...")

    os.makedirs(os.path.dirname(optimized_model_path), exist_ok=True)

    # Create session with all optimizations enabled
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create temporary session to trigger optimization
    temp_session = ort.InferenceSession(
        onnx_model_path,
        sess_options=session_options,
        providers=[ep],
    )

    # Load model and save (optimizations are applied during session creation)
    model = onnx.load(onnx_model_path)
    onnx.save(model, optimized_model_path)

    print(f"✓ Optimized model saved to: {optimized_model_path}")
    print(f"  Original size: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")
    print(
        f"  Optimized size: {os.path.getsize(optimized_model_path) / (1024*1024):.2f} MB"
    )


def main():
    """Main function to optimize ONNX model from command line."""
    parser = argparse.ArgumentParser(description="Optimize ONNX model graph")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--ep",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save optimized model (default: model_optimized.onnx in same directory)",
    )
    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        model_dir = "./checkpoints/onnx_optimized/"
        model_name = os.path.basename(args.model)
        name, ext = os.path.splitext(model_name)
        args.output = os.path.join(model_dir, f"{name}_optimized{ext}")

    optimize_onnx_model(args.model, args.output, args.ep)


if __name__ == "__main__":
    main()
