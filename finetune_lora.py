#!/usr/bin/env python3
"""Fine-tune MoCo model with LoRA/QLoRA for efficient parameter-efficient learning."""

import argparse
import os
import torch
import torch.nn as nn
from typing import Dict, Optional
from model import MoCo
from torchvision import models
from utils import get_config_hierarchical, init_wandb, print_config, global_state
from dataset import set_seed, build_dataloader_from_dir, infinite_loader
from trainer import MoCoTrainer


class LoRALayer(nn.Module):
    """LoRA layer implementation for linear layers."""
    
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        return self.lora_A(x) @ self.lora_B.weight.T * self.scaling


class QLoRALayer(nn.Module):
    """QLoRA layer with 4-bit quantization support."""
    
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA matrices (will use bfloat16)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.lora_B.weight)
        
        # Store original dtype
        self.dtype = torch.float32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QLoRA transformation with reduced precision."""
        # Convert to bfloat16 for computation efficiency
        x_dtype = x.dtype
        x = x.to(torch.bfloat16)
        
        lora_out = self.lora_A(x) @ self.lora_B.weight.T * self.scaling
        return lora_out.to(x_dtype)


def add_lora_to_model(
    model: nn.Module, 
    r: int = 8, 
    lora_alpha: int = 16,
    target_modules: Optional[list] = None,
    use_qlora: bool = False
) -> nn.Module:
    """
    Add LoRA/QLoRA layers to specific modules in the model.
    
    Args:
        model: The model to add LoRA to
        r: Rank of LoRA matrices
        lora_alpha: Scaling factor for LoRA
        target_modules: List of module names to apply LoRA to (e.g., ['fc', 'conv1'])
        use_qlora: Whether to use QLoRA (4-bit quantization) instead of standard LoRA
    
    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ['fc', 'linear']  # Default target modules
    
    lora_class = QLoRALayer if use_qlora else LoRALayer
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        should_apply = any(target in name for target in target_modules)
        
        if should_apply and isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            # Create LoRA layer wrapper
            lora_layer = lora_class(in_features, out_features, r=r, lora_alpha=lora_alpha)
            
            # Register LoRA layer as submodule
            module.lora = lora_layer
    
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze the backbone parameters (non-LoRA parameters).
    
    Args:
        model: The model to freeze
        freeze: Whether to freeze (True) or unfreeze (False)
    """
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = not freeze
        else:
            param.requires_grad = True


def get_lora_parameters(model: nn.Module):
    """Get only the LoRA parameters for optimization."""
    return [param for name, param in model.named_parameters() if 'lora' in name]


def get_trainable_parameters(model: nn.Module):
    """Get all trainable parameters."""
    return [param for param in model.parameters() if param.requires_grad]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters.
    
    Returns:
        Dictionary with 'total' and 'trainable' counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def main(
    checkpoint_path: str,
    config_path: str,
    output_dir: str = "checkpoints/lora",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    use_qlora: bool = False,
    freeze_backbone: bool = True,
    learning_rate: float = 1e-4,
    epochs: int = 10,
):
    """
    Fine-tune MoCo model with LoRA/QLoRA.
    
    Args:
        checkpoint_path: Path to the pre-trained checkpoint
        config_path: Path to config file
        output_dir: Output directory for saving checkpoints
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling factor
        use_qlora: Use QLoRA (4-bit) instead of standard LoRA
        freeze_backbone: Whether to freeze non-LoRA parameters
        learning_rate: Learning rate for training
        epochs: Number of training epochs
    """
    
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = get_config_hierarchical(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb_config = cfg.copy()
    wandb_config.update({
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'use_qlora': use_qlora,
        'freeze_backbone': freeze_backbone,
        'learning_rate': learning_rate,
        'epochs': epochs,
    })
    init_wandb(wandb_config, run_name_suffix="lora")
    
    # Create model
    print("🏗️  Creating MoCo model...")
    backbone = models.resnet50(weights=None)
    model = MoCo(
        backbone,
        proj_dim=cfg["proj_dim"],
        hidden_dim=cfg["hidden_dim"],
        queue_size=cfg["queue_size"],
        momentum=cfg["momentum"],
        temperature=cfg["temperature"],
    ).to(device)
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print("✓ Model checkpoint loaded")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    # Add LoRA layers
    print(f"\n🔧 Adding {'QLoRA' if use_qlora else 'LoRA'} layers...")
    print(f"   Rank: {lora_rank}, Alpha: {lora_alpha}")
    model = add_lora_to_model(
        model, 
        r=lora_rank, 
        lora_alpha=lora_alpha,
        target_modules=['fc', 'linear'],  # Target fully connected and linear layers
        use_qlora=use_qlora
    )
    
    # Freeze backbone if requested
    if freeze_backbone:
        print("❄️  Freezing backbone parameters...")
        freeze_backbone_params(model, freeze=True)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"\n📊 Parameter counts:")
    print(f"   Total: {param_counts['total']:,}")
    print(f"   Trainable: {param_counts['trainable']:,}")
    print(f"   Frozen: {param_counts['frozen']:,}")
    print(f"   Efficiency: {100 * param_counts['trainable'] / param_counts['total']:.2f}% of parameters trainable")
    
    # Setup optimizer (only optimize LoRA parameters if backbone is frozen)
    if freeze_backbone:
        lora_params = get_lora_parameters(model)
        optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        print(f"\n⚙️  Optimizing {len(lora_params)} LoRA parameters")
    else:
        trainable_params = get_trainable_parameters(model)
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        print(f"\n⚙️  Optimizing {len(trainable_params)} parameters")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    
    # Load dataloaders
    print("\n📦 Loading datasets...")
    train_loader = build_dataloader_from_dir(
        cfg["data_path"],
        batch_size=cfg["batch_size"],
        image_size=cfg["img_size"],
        num_workers=cfg.get("num_workers", 4),
    )
    
    val_loader = build_dataloader_from_dir(
        cfg["data_path"],
        batch_size=cfg["batch_size"],
        image_size=cfg["img_size"],
        num_workers=cfg.get("num_workers", 4),
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = MoCoTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        stage=2,
        save_every_epochs=1,
        device=device,
        use_amp=True,
    )
    
    # Train
    print("\n🚀 Starting LoRA fine-tuning...")
    train_loader_inf = infinite_loader(train_loader)
    trainer.train(
        train_loader=train_loader_inf,
        val_loader=val_loader,
        eval_every_epochs=1,
        out_dir=output_dir,
    )
    
    print("\n✅ LoRA fine-tuning complete!")


def freeze_backbone_params(model: nn.Module, freeze: bool = True) -> None:
    """Freeze backbone parameters (keep LoRA trainable)."""
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = not freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MoCo model with LoRA/QLoRA"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to pre-trained MoCo checkpoint",
    )
    parser.add_argument(
        "--config",
        default="./configs/stage1.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/lora",
        help="Output directory for LoRA checkpoints",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization) instead of standard LoRA",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze backbone parameters (only train LoRA)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_qlora=args.use_qlora,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
