import argparse
import yaml
import os
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()


BASE_CONFIG_PATH = "configs/base_config.yaml"


# Add this at the module level in utils.py (after imports)
class GlobalState:
    """Thread-safe global state for training."""

    def __init__(self):
        self.global_steps = 0

    def increment(self):
        self.global_steps += 1

    def set(self, value):
        self.global_steps = value

    def get(self):
        return self.global_steps

    def reset(self):
        self.global_steps = 0


# Create global instance
global_state = GlobalState()


@dataclass
class CFG:
    # Data
    train_dir: str = ""
    eval_dir: str = ""
    img_size: int = 224
    # val_fraction: float = 0.05  # Fraction of data used for validation
    eval_every_epochs: int = 1  # Evaluate every N epochs

    # Mix ratio per step: batch = nffa_bs + nano_bs
    total_batch_size: int = 128
    # nffa_fraction: float = 0.5  # 0.5 -> 50/50
    num_workers: int = 8

    # MoCo
    backbone: str = "resnet50"
    proj_dim: int = 128
    hidden_dim: int = 2048
    queue_size: int = 65536
    momentum: float = 0.999
    temperature: float = 0.2

    # Train
    epochs: int = 200
    lr: float = 0.03
    weight_decay: float = 1e-4
    momentum_sgd: float = 0.9
    use_amp: bool = True

    # Logging / ckpt
    out_dir: str = ""
    save_every_epochs: int = 10
    seed: int = 42


def load_yaml_config(file_path):
    """Load a single YAML file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must be a dictionary.")
    return config


def deep_merge_dicts(base_dict, override_dict):
    """Recursively merge override_dict into base_dict."""
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        if key in config:
            config[key] = value
    return config


def get_config_hierarchical(stage_config_path: str):
    base_config = load_yaml_config(BASE_CONFIG_PATH)
    cfg_dict = apply_overrides(CFG().__dict__, base_config)
    stage_config = load_yaml_config(stage_config_path)
    cfg_dict = deep_merge_dicts(cfg_dict, stage_config)
    return cfg_dict


def print_config(cfg):
    print("Final Configuration:")
    for key, value in cfg.items():
        print(f"{key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load YAML configuration")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    return args


def init_wandb(cfg, run_name_suffix="", resume=False, run_id=None):
    """
    Initialize Weights & Biases for tracking.

    Args:
        cfg: Configuration dict or object
        run_name_suffix: Suffix to append to run name
        resume: If True, resumes the last unfinished run. If False, creates a new run.
        run_id: Specific run ID to resume. If provided, overrides resume parameter.

    Returns:
        wandb module or None if initialization fails
    """
    import wandb
    import sys

    # Check if WANDB_API_KEY is available in environment
    if not os.getenv("WANDB_API_KEY"):
        print(
            "⚠️  Warning: WANDB_API_KEY not found in .env file. WandB will run in offline mode."
        )

    # Initialize wandb with optional API key
    init_kwargs = {
        "project": "moco-sem-pretrain",
        "name": f"sem_moco_resnet50_{run_name_suffix}",
        "config": cfg if isinstance(cfg, dict) else cfg.__dict__,
        "tags": ["moco", "resnet50", "sem"],
    }

    # Handle resumption
    if run_id:
        # Resume a specific run by ID
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "must"
        print(f"📊 Resuming WandB run: {run_id}")
    elif resume:
        # Resume the last unfinished run
        init_kwargs["resume"] = "allow"
        print("📊 Attempting to resume last unfinished WandB run...")
    else:
        # Start fresh
        init_kwargs["resume"] = "never"

    try:
        wandb.init(**init_kwargs)
    except Exception as e:
        print(f"⚠️  WandB init error: {e}. Continuing without WandB tracking.")
        return None

    try:
        wandb.config.update({"cmd": " ".join(sys.argv)}, allow_val_change=True)
    except TypeError:
        # For newer wandb versions that don't support allow_val_change parameter
        wandb.config.update({"cmd": " ".join(sys.argv)})

    return wandb


def get_last_wandb_run_id():
    """
    Retrieve the last run ID from the latest WandB run directory.

    Returns:
        str: Run ID of the last run, or None if not found
    """
    from pathlib import Path

    wandb_dir = Path("./wandb")
    if not wandb_dir.exists():
        print("⚠️  No WandB directory found. Starting fresh run.")
        return None

    # Find the latest run directory
    run_dirs = [
        d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")
    ]
    if not run_dirs:
        print("⚠️  No WandB runs found. Starting fresh run.")
        return None

    # Get the latest run by modification time
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)

    # Extract run ID from directory name: run-TIMESTAMP-RUN_ID
    run_dir_name = latest_run.name
    parts = run_dir_name.split("-")

    if len(parts) >= 3:
        run_id = parts[-1]  # Last part is the run ID
        print(f"📊 Found last run ID: {run_id}")
        return run_id

    print(f"⚠️  Could not extract run ID from {run_dir_name}. Starting fresh run.")
    return None


def init_wandb_with_resume(cfg, run_name_suffix="", auto_resume=True):
    """
    Initialize WandB with automatic resumption of last unfinished run.

    Args:
        cfg: Configuration dict or object
        run_name_suffix: Suffix to append to run name
        auto_resume: If True, automatically resume last run. If False, start fresh.

    Returns:
        wandb module or None if initialization fails
    """
    if auto_resume:
        run_id = get_last_wandb_run_id()
        if run_id:
            return init_wandb(cfg, run_name_suffix=run_name_suffix, run_id=run_id)

    return init_wandb(cfg, run_name_suffix=run_name_suffix, resume=False)
