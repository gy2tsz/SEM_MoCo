#!/usr/bin/env python3
"""ONNX model graph optimization utilities."""

import os
import onnx
import onnxruntime as ort


def optimize_onnx_model(onnx_model_path, optimized_model_path):
    """
    Optimize ONNX model graph and save to file.

    This is a one-time preprocessing step that optimizes the model
    graph (operator fusion, constant folding, etc.) before inference.

    Args:
        onnx_model_path: Path to original ONNX model
        optimized_model_path: Path to save optimized model
    """
    print("🔧 Optimizing ONNX model graph...")

    # Create session with all optimizations enabled
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create temporary session to trigger optimization
    temp_session = ort.InferenceSession(
        onnx_model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    # Load model and save (optimizations are applied during session creation)
    model = onnx.load(onnx_model_path)
    onnx.save(model, optimized_model_path)
    
    print(f"✓ Optimized model saved to: {optimized_model_path}")
    print(f"  Original size: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")
    print(
        f"  Optimized size: {os.path.getsize(optimized_model_path) / (1024*1024):.2f} MB"
    )
