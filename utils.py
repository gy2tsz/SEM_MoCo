import argparse
import yaml
from dataclasses import dataclass


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
    train_stage_1: str = "datasets/train_stage_1"
    eval_stage_1: str = "datasets/eval_stage_1"
    img_size: int = 224
    val_fraction: float = 0.05  # Fraction of data used for validation
    eval_every_epochs: int = 1  # Evaluate every N epochs

    # Mix ratio per step: batch = nffa_bs + nano_bs
    total_batch_size: int = 128
    nffa_fraction: float = 0.5  # 0.5 -> 50/50
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

    # WandB
    wandb_key: str = ""


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


def init_wandb(cfg, run_name_suffix=""):
    import wandb
    import sys

    wandb.login(key=cfg["wandb_key"])
    wandb.init(
        project="moco-sem-pretrain",
        name=f"sem_moco_resnet50_{run_name_suffix}",
        config=cfg if isinstance(cfg, dict) else cfg.__dict__,
        tags=["moco", "resnet50", "sem"],
    )
    wandb.config.update({"cmd": " ".join(sys.argv)}, allow_val_change=True)
    return wandb
