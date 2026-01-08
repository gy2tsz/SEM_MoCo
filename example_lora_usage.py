#!/usr/bin/env python3
"""
Simple example script demonstrating LoRA fine-tuning.
Minimal setup for quick testing.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MoCo
from finetune_lora import add_lora_to_model, freeze_backbone_params, count_parameters
from lora_utils import LoRAManager
from utils import get_config_hierarchical
from torchvision import models


def example_1_basic_setup():
    """Example 1: Basic LoRA setup"""
    print("\n" + "=" * 60)
    print("Example 1: Basic LoRA Setup")
    print("=" * 60)

    # Create a simple model
    backbone = models.resnet50(weights=None)
    model = MoCo(
        backbone,
        proj_dim=128,
        hidden_dim=2048,
        queue_size=65536,
        momentum=0.999,
        temperature=0.2,
    )

    # Add LoRA
    model = add_lora_to_model(model, r=8, lora_alpha=16)

    # Freeze backbone
    freeze_backbone_params(model, freeze=True)

    # Check parameters
    params = count_parameters(model)
    print(f"\n📊 Parameter counts:")
    print(f"   Total: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Efficiency: {100 * params['trainable'] / params['total']:.2f}%")

    return model


def example_2_training_setup():
    """Example 2: Setup for training"""
    print("\n" + "=" * 60)
    print("Example 2: Training Setup")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = example_1_basic_setup().to(device)

    # Create optimizer with only LoRA parameters
    lora_params = [p for name, p in model.named_parameters() if "lora" in name]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    print(f"\n⚙️  Training setup:")
    print(f"   Device: {device}")
    print(f"   LoRA parameters: {len(lora_params)}")
    print(f"   Optimizer: AdamW (lr=1e-4)")

    return model, optimizer, device


def example_3_checkpoint_management():
    """Example 3: Save and load LoRA checkpoints"""
    print("\n" + "=" * 60)
    print("Example 3: Checkpoint Management")
    print("=" * 60)

    # Create model with LoRA
    model = example_1_basic_setup()
    manager = LoRAManager(model)

    # Get info
    info = manager.get_lora_info()
    print(f"\n📊 LoRA Info:")
    print(f"   LoRA layers: {len(info['lora_layers'])}")
    print(f"   LoRA parameters: {info['total_lora_params']:,}")
    print(f"   LoRA ratio: {info['lora_ratio']:.2f}%")

    # Save checkpoint
    checkpoint_path = Path("./checkpoints/lora/example_model_lora.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": "MoCo",
        "lora_rank": 8,
        "lora_alpha": 16,
    }

    manager.save_lora_checkpoint(str(checkpoint_path), metadata=metadata)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")

    # Load checkpoint
    print(f"\n📂 Loading checkpoint...")
    loaded_metadata = manager.load_lora_checkpoint(str(checkpoint_path))
    print(f"   Metadata: {loaded_metadata}")

    return model, checkpoint_path


def example_4_inference():
    """Example 4: Run inference with LoRA model"""
    print("\n" + "=" * 60)
    print("Example 4: Inference")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = example_1_basic_setup().to(device)
    model.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    print(f"\n🔍 Running inference:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Device: {device}")

    with torch.no_grad():
        features = model.encoder_q(dummy_input)

    print(f"   Output shape: {features.shape}")
    print(f"\n✓ Inference successful!")

    return model, features


def example_5_compare_sizes():
    """Example 5: Compare checkpoint sizes"""
    print("\n" + "=" * 60)
    print("Example 5: Checkpoint Size Comparison")
    print("=" * 60)

    # Full model checkpoint
    model_full = MoCo(
        models.resnet50(weights=None),
        proj_dim=128,
        hidden_dim=2048,
        queue_size=65536,
        momentum=0.999,
        temperature=0.2,
    )

    # LoRA model
    model_lora = add_lora_to_model(model_full, r=8)

    # Calculate sizes
    full_params = sum(p.numel() for p in model_full.parameters())
    lora_params = sum(
        p.numel() for name, p in model_lora.named_parameters() if "lora" in name
    )

    full_size_mb = full_params * 4 / (1024**2)
    lora_size_mb = lora_params * 4 / (1024**2)

    print(f"\n📊 Checkpoint Sizes:")
    print(f"   Full model: {full_size_mb:.2f} MB")
    print(f"   LoRA only: {lora_size_mb:.2f} MB")
    print(f"   Savings: {100 * (1 - lora_size_mb/full_size_mb):.1f}%")


def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("# LoRA Fine-tuning Examples")
    print("#" * 60)

    try:
        example_1_basic_setup()
        example_2_training_setup()
        example_3_checkpoint_management()
        example_4_inference()
        example_5_compare_sizes()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
