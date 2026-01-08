"""Utilities for LoRA/QLoRA model management and inference."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import os


class LoRAManager:
    """Manager class for handling LoRA-enabled models."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def save_lora_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> None:
        """
        Save only LoRA parameters (lightweight checkpoint).
        
        Args:
            path: Path to save the checkpoint
            metadata: Optional metadata to include in checkpoint
        """
        lora_state = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                lora_state[name] = param.data.cpu()
        
        checkpoint = {
            'lora_state': lora_state,
            'metadata': metadata or {},
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        
        # Calculate size
        size_mb = sum(p.numel() * 4 for p in lora_state.values()) / (1024 ** 2)
        print(f"✓ LoRA checkpoint saved: {path} ({size_mb:.2f} MB)")
    
    def load_lora_checkpoint(self, path: str) -> Dict:
        """
        Load LoRA parameters from checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Metadata from checkpoint
        """
        checkpoint = torch.load(path, map_location='cpu')
        lora_state = checkpoint['lora_state']
        
        # Load LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora' in name and name in lora_state:
                param.data = lora_state[name].to(param.device)
        
        print(f"✓ LoRA checkpoint loaded: {path}")
        return checkpoint.get('metadata', {})
    
    def merge_lora_weights(self, scaling: float = 1.0) -> None:
        """
        Merge LoRA weights into original weights.
        Warning: This modifies the original model weights permanently.
        
        Args:
            scaling: Scaling factor for LoRA contribution
        """
        print("⚠️  Merging LoRA weights into base model...")
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora'):
                # For each linear layer with LoRA
                with torch.no_grad():
                    lora_obj = getattr(module, 'lora')
                    if not isinstance(lora_obj, (LoRALayer, QLoRALayer)):
                        continue
                    
                    # Get LoRA weight matrices and compute contribution
                    lora_A = lora_obj.lora_A
                    lora_B = lora_obj.lora_B
                    lora_scale = lora_obj.scaling
                    
                    # Compute: (out, r) @ (r, in) = (out, in)
                    lora_weight = (lora_B.weight @ lora_A.weight) * (lora_scale * scaling)
                    
                    # Add LoRA contribution to original weight
                    if isinstance(module, torch.nn.Linear):
                        module.weight.data.add_(lora_weight)
        
        print("✓ LoRA weights merged")
    
    def get_lora_info(self) -> Dict:
        """Get information about LoRA layers in the model."""
        info = {
            'lora_layers': [],
            'total_lora_params': 0,
            'total_params': 0,
        }
        
        for name, param in self.model.named_parameters():
            info['total_params'] += param.numel()
            if 'lora' in name:
                info['lora_layers'].append(name)
                info['total_lora_params'] += param.numel()
        
        info['lora_ratio'] = (
            info['total_lora_params'] / info['total_params'] * 100
            if info['total_params'] > 0 else 0
        )
        
        return info
    
    def enable_lora(self) -> None:
        """Enable LoRA gradients for training."""
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
    
    def disable_lora(self) -> None:
        """Disable LoRA gradients (freeze LoRA)."""
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
    
    def get_lora_state_dict(self) -> Dict:
        """Get only LoRA state dict for saving."""
        return {
            name: param.data
            for name, param in self.model.named_parameters()
            if 'lora' in name
        }
    
    def load_lora_state_dict(self, state_dict: Dict) -> None:
        """Load LoRA state dict."""
        for name, param in self.model.named_parameters():
            if 'lora' in name and name in state_dict:
                param.data = state_dict[name].to(param.device)


class AdapterModule(nn.Module):
    """Generic adapter module that can be inserted into any layer."""
    
    def __init__(self, hidden_dim: int, adapter_dim: int = 64):
        super().__init__()
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = x + residual
        x = self.layer_norm(x)
        return x


def print_lora_config(model: nn.Module) -> None:
    """Print LoRA configuration of the model."""
    lora_layers = []
    total_lora_params = 0
    
    print("\n" + "="*60)
    print("LoRA Configuration")
    print("="*60)
    
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_layers.append(name)
            total_lora_params += param.numel()
            print(f"  {name}: {param.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nTotal LoRA parameters: {total_lora_params:,}")
    print(f"Total model parameters: {total_params:,}")
    print(f"LoRA ratio: {100 * total_lora_params / total_params:.2f}%")
    print("="*60 + "\n")


def benchmark_lora_training(model: nn.Module, num_iterations: int = 100) -> Dict:
    """
    Benchmark LoRA training performance.
    
    Args:
        model: Model with LoRA layers
        num_iterations: Number of iterations to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    device = next(model.parameters()).device
    model.train()
    
    # Dummy inputs
    batch_size = 16
    input_dim = 2048  # ResNet50 output
    
    forward_times = []
    backward_times = []
    
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, input_dim, device=device)
    
    for _ in range(num_iterations):
        # Forward pass
        start = time.time()
        output = model(dummy_input)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.mean()
        forward_times.append(time.time() - start)
        
        # Backward pass
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_times.append(time.time() - start)
    
    return {
        'avg_forward_time': sum(forward_times) / len(forward_times),
        'avg_backward_time': sum(backward_times) / len(backward_times),
        'avg_total_time': sum(forward_times) / len(forward_times) + 
                          sum(backward_times) / len(backward_times),
    }
