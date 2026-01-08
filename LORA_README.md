# LoRA/QLoRA Fine-tuning Guide

## Overview

LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning by:
- Freezing pre-trained weights
- Adding trainable low-rank matrices to linear layers
- Reducing trainable parameters by **99%** (1-2% trainable)
- Maintaining or improving model quality

**QLoRA** adds 4-bit quantization for extreme memory efficiency.

## Quick Start (3 Options)

### Option 1: Command Line (Fastest)
```bash
# Standard LoRA fine-tuning
python finetune_lora.py \
    --checkpoint ./checkpoints/moco_stage1/moco_stage1_epoch_100.pth \
    --config ./configs/stage1.yaml \
    --lora-rank 8 --epochs 10

# Memory-efficient QLoRA
python finetune_lora.py --use-qlora --lora-rank 4 --epochs 10
```

### Option 2: Jupyter Notebook (Interactive)
```bash
jupyter notebook finetune_lora_notebook.ipynb
```
9 sections: setup → training → visualization → evaluation

### Option 3: Run Examples (Testing)
```bash
python example_lora_usage.py
```
5 examples: setup, training, checkpoints, inference, size comparison

## Implementation

### Core Files
- **`finetune_lora.py`** - LoRA/QLoRA layers + training pipeline
- **`lora_utils.py`** - LoRAManager for checkpoint management
- **`example_lora_usage.py`** - Ready-to-run examples
- **`finetune_lora_notebook.ipynb`** - Interactive Jupyter notebook

### Python API

```python
from finetune_lora import add_lora_to_model, freeze_backbone_params
from lora_utils import LoRAManager

# Step 1: Add LoRA to model
model = add_lora_to_model(model, r=8, lora_alpha=16)
freeze_backbone_params(model, freeze=True)

# Step 2: Train normally with standard PyTorch
optimizer = torch.optim.AdamW(
    [p for name, p in model.named_parameters() if 'lora' in name],
    lr=1e-4
)
# ... standard training loop ...

# Step 3: Save lightweight checkpoint
manager = LoRAManager(model)
manager.save_lora_checkpoint('lora.pth')  # 10-50 MB only!

# Step 4: Load and use later
manager.load_lora_checkpoint('lora.pth')
# Or merge into base model:
manager.merge_lora_weights()
```

## Performance

| Metric | Standard | LoRA | QLoRA |
|--------|----------|------|-------|
| Trainable Params | 100% | 1-2% | 1-2% |
| Memory | 100% | 30-50% | 15-25% |
| Speed | 1x | 1.5-2x | 1.5-2x |
| Checkpoint | 500 MB | 10-50 MB | 10-50 MB |
| Quality | 100% | 98-99% | 95-98% |

## Configuration Guide

### LoRA Rank Selection
- **Rank 4**: Extreme constraints (95% quality)
- **Rank 8**: Standard/recommended (98% quality)
- **Rank 16**: High quality (99% quality)
- **Rank 32**: Maximum quality (99%+ quality)

### Command-line Arguments
```bash
python finetune_lora.py \
    --checkpoint PATH              # Pre-trained model checkpoint
    --config PATH                  # Config file
    --lora-rank 8                  # LoRA rank (default: 8)
    --lora-alpha 16                # Scaling factor (default: 16)
    --learning-rate 1e-4           # Fine-tuning LR (default: 1e-4)
    --epochs 10                    # Training epochs (default: 10)
    --use-qlora                    # Enable 4-bit quantization
    --freeze-backbone              # Only train LoRA (default: true)
    --output-dir PATH              # Checkpoint directory
```

### Recommended Settings
```bash
# Balanced (default)
--lora-rank 8 --lora-alpha 16 --learning-rate 1e-4

# Memory efficient
--lora-rank 4 --use-qlora --learning-rate 5e-5

# High quality
--lora-rank 16 --lora-alpha 32 --learning-rate 5e-5 --epochs 20
```

## Advanced Usage

### Custom Setup
```python
# Experiment with different ranks
for rank in [4, 8, 16]:
    model_lora = add_lora_to_model(model, r=rank, lora_alpha=rank*2)
    params = count_parameters(model_lora)
    print(f"Rank {rank}: {params['trainable']:,} params")
```

### Multi-Stage Training
```python
# Stage 1: LoRA only
freeze_backbone_params(model, freeze=True)
# ... train ...

# Stage 2: Fine-tune everything
freeze_backbone_params(model, freeze=False)
# ... train with lower LR ...
```

### Merge and Deploy
```python
manager = LoRAManager(model)
manager.merge_lora_weights()
torch.save(model.state_dict(), 'merged.pth')
# Now model works without LoRA infrastructure
```

## Troubleshooting

### Out of Memory?
```bash
# Use QLoRA
python finetune_lora.py --use-qlora --lora-rank 4

# Or reduce rank
python finetune_lora.py --lora-rank 4
```

### Poor Results?
```bash
# Lower learning rate
python finetune_lora.py --learning-rate 5e-5

# More epochs
python finetune_lora.py --epochs 20

# Verify checkpoint loaded
python -c "import torch; ckpt = torch.load('checkpoint.pth'); print(ckpt.keys())"
```

### Model Diverging?
```bash
# Reduce learning rate further
python finetune_lora.py --learning-rate 1e-5

# Add warmup (modify code):
from torch.optim.lr_scheduler import LinearLR
scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=100)
```

## Common Commands

```bash
# Quick test (1 epoch)
python finetune_lora.py --checkpoint ./checkpoints/moco_stage1/moco_stage1_epoch_100.pth --config ./configs/stage1.yaml --epochs 1

# Standard training
python finetune_lora.py --checkpoint ./checkpoints/moco_stage1/moco_stage1_epoch_100.pth --config ./configs/stage1.yaml --lora-rank 8

# Memory constrained
python finetune_lora.py --use-qlora --lora-rank 4 --config ./configs/stage1.yaml

# High quality
python finetune_lora.py --lora-rank 16 --epochs 20 --config ./configs/stage1.yaml

# Interactive learning
jupyter notebook finetune_lora_notebook.ipynb

# Run examples
python example_lora_usage.py
```

## Architecture

### LoRA Mechanism
```
Input ──┬──→ [Linear W] ───┐
        │                  ├──→ Output
        └──→ [A @ B] ─────┘
           (trainable)
```

For each linear layer: `output = W(input) + (alpha/r) * B @ A @ input`

### QLoRA Extension
- Base model: 4-bit quantized (frozen)
- LoRA: bfloat16 computation (trainable)
- Result: ~4x memory reduction

## References

- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **MoCo Paper**: https://arxiv.org/abs/1911.05722

---

**For detailed examples and walkthroughs, see `example_lora_usage.py` or run `jupyter notebook finetune_lora_notebook.ipynb`**
