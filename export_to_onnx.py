#!/usr/bin/env python3
"""Export MoCo model to ONNX Runtime (ORT) format for inference with TensorRT."""

import argparse
import os
import torch
import torch.onnx
from model import MoCo
from torchvision import models
from utils import get_config_hierarchical


def export_to_onnx(model, output_path, image_size=224, device="cuda"):
    """Export ResNet backbone to ONNX format."""

    print(f"📦 Exporting ResNet backbone to ONNX format...")

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # Export to ONNX
    torch.onnx.export(
        model.encoder_q,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["features"],
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"✓ Model exported to: {output_path}")


def load_checkpoint(checkpoint_path, model, device):
    """Load checkpoint into model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print("✓ Model state loaded")
    return model


def main(checkpoint_path, config_path, output_dir):
    """Main export function."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    cfg = get_config_hierarchical(config_path)

    # Create output directory
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Create model
    print("🏗️  Creating model...")
    model = MoCo(
        models.resnet50(weights=None),
        proj_dim=cfg["proj_dim"],
        hidden_dim=cfg["hidden_dim"],
        queue_size=cfg["queue_size"],
        momentum=cfg["momentum"],
        temperature=cfg["temperature"],
    ).to(device)

    # Load checkpoint
    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()

    # Export to ONNX
    export_to_onnx(model, output_dir, image_size=cfg["img_size"], device=str(device))

    print(f"\n✓ Export complete!")
    print(f"  ONNX model: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MoCo model to ONNX")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (.pth)",
    )
    parser.add_argument(
        "--config",
        default="./configs/base_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/onnx/resnet_backbone.onnx",
        help="Output directory for ONNX model",
    )
    args = parser.parse_args()

    main(args.checkpoint, args.config, args.output_dir)
