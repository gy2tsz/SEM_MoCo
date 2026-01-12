#!/usr/bin/env python3
"""
Train and evaluate a classification head on top of ResNet50 backbone.
Uses Carinthia dataset for training and evaluation.
"""

import argparse
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tqdm
import wandb
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from utils import (
    get_config_hierarchical,
    init_wandb,
    print_config,
    global_state,
    CLS_CFG,
)
from model import ClassificationHead
from dataset import set_seed, build_dataloaders_from_csv, inspect_csv_columns
import json
from PIL import Image
from train_stage import load_checkpoint


def load_backbone(
    backbone_name: str = "resnet50",
    optimizer=None,
    checkpoint_path: Optional[str] = None,
    device=None,
):
    """
    Load ResNet50 backbone, optionally with pre-trained MoCo weights.
    Extracts encoder_q from MoCo checkpoint if needed.
    """
    if backbone_name == "resnet50":
        backbone = models.resnet50(pretrained=False)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Load pre-trained weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Check if this is a MoCo model or plain backbone
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    # Full checkpoint with metadata
                    model_state = checkpoint["model"]
                else:
                    # Direct state dict
                    model_state = checkpoint

                # Check if it's a MoCo model (has encoder_q prefix)
                if any(key.startswith("encoder_q.") for key in model_state.keys()):
                    print("Detected MoCo model - extracting encoder_q backbone...")
                    # Extract encoder_q weights and remove the prefix
                    backbone_state = {}
                    for key, value in model_state.items():
                        if key.startswith("encoder_q."):
                            new_key = key.replace("encoder_q.", "")
                            if not new_key.startswith("fc"):  # Skip the FC layer
                                backbone_state[new_key] = value

                    if backbone_state:
                        backbone.load_state_dict(backbone_state, strict=False)
                        print(
                            f"✓ Loaded {len(backbone_state)} weights from MoCo encoder_q"
                        )
                    else:
                        print("⚠ No encoder_q weights found in checkpoint")
                else:
                    # Plain ResNet50 weights
                    print("Loading plain ResNet50 weights...")
                    backbone.load_state_dict(model_state, strict=False)
                    print("✓ Loaded backbone weights")
            else:
                print("⚠ Checkpoint format not recognized")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    return backbone


def create_model(
    num_classes: int,
    backbone_dim: int = 2048,
    checkpoint_path: Optional[str] = None,
    optimizer=None,
    device=None,
):
    """Create model with backbone and classification head."""
    backbone = load_backbone("resnet50", optimizer, checkpoint_path, device)
    head = ClassificationHead(backbone_dim, num_classes)
    model = nn.Sequential(backbone, head)
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    progress_bar = tqdm.tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, return_predictions=False):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "preds": all_preds if return_predictions else None,
        "labels": all_labels if return_predictions else None,
        "probs": all_probs if return_predictions else None,
    }

    return metrics


def plot_confusion_matrix(all_preds, all_labels):
    """Generate confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    return cm


def main(
    yaml_config_path: str = "./configs/cls_head.yaml",
    checkpoint_path: str = "",
    save_best: bool = True,
):
    """
    Train and evaluate classification head on Carinthia dataset using CSV labels.

    Args:
        yaml_config_path: Path to config YAML file
        checkpoint_path: Path to MoCo checkpoint for backbone initialization
        save_best: Whether to save best model
    """

    # Setup
    # reset global state
    global_state.reset()

    # config
    try:
        cfg = get_config_hierarchical(yaml_config_path, config_class=CLS_CFG)
    except FileNotFoundError:
        print(f"Error: Config file not found at {yaml_config_path}")
        raise
    except Exception as e:
        print(f"Error loading config from {yaml_config_path}: {e}")
        raise

    print_config(cfg)

    # Validate required config keys
    required_keys = [
        "seed",
        "out_dir",
        "use_amp",
        "csv_path",
        "train_dir",
        "img_size",
        "total_batch_size",
        "num_workers",
        "val_fraction",
        "num_classes",
        "lr",
        "weight_decay",
        "epochs",
    ]

    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        print(f"Error: Missing required config keys: {missing_keys}")
        print(f"Available keys: {list(cfg.keys())}")
        raise KeyError(f"Missing required config keys: {missing_keys}")

    set_seed(cfg["seed"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["use_amp"] and (device.type == "cuda")

    # Check if CSV file exists
    if not os.path.exists(cfg["csv_path"]):
        print(f"Error: CSV file not found at {cfg['csv_path']}")
        print(f"Expected columns: 'filename' (relative path) and 'label'")
        raise FileNotFoundError(f"CSV file not found: {cfg['csv_path']}")

    # Inspect CSV columns
    print(f"\nInspecting CSV file: {cfg['csv_path']}")
    columns = inspect_csv_columns(cfg["csv_path"])

    # Create dataloaders from CSV
    train_dl, val_dl, class_names = build_dataloaders_from_csv(
        csv_path=cfg["csv_path"],
        img_dir=cfg["train_dir"],
        image_size=cfg["img_size"],
        batch_size=cfg["total_batch_size"],
        num_workers=cfg["num_workers"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        img_col="file_name",
        label_col="label",
    )

    # Create model
    model = create_model(
        cfg["num_classes"], checkpoint_path=checkpoint_path, device=device
    )
    model = model.to(device)

    # Optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    # Initialize wandb
    init_wandb(cfg, run_name_suffix=f"cls_head")

    # Training loop
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_dl, criterion, optimizer, device, use_amp
        )
        scheduler.step()

        # Validates
        val_metrics = evaluate(model, val_dl, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Log to wandb
        try:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/f1": val_metrics["f1"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )
        except Exception as e:
            print(f"⚠ Could not log to wandb: {e}")

        global_state.increment()

        # Save best model
        if save_best and val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_path = os.path.join(cfg["out_dir"], "cls_head_best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "global_state": global_state.get(),
                },
                best_path,
            )
            print(f"✓ Best model saved at: {best_path}")

        # Save checkpoint
        if epoch % cfg["save_every_epochs"] == 0:
            ckpt_path = os.path.join(cfg["out_dir"], f"cls_head_epoch_{epoch}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"✓ Checkpoint saved at: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")

    # Final evaluation on test set
    print("\nFinal evaluation on validation set:")
    final_metrics = evaluate(model, val_dl, criterion, device, return_predictions=True)

    # Generate classification report
    class_report = classification_report(
        final_metrics["labels"],
        final_metrics["preds"],
        zero_division=0,
        output_dict=False,
    )
    print("\nClassification Report:")
    print(class_report)

    # Save classification report
    report_path = os.path.join(cfg["out_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(str(class_report))
    print(f"✓ Classification report saved at: {report_path}")

    # Save confusion matrix
    cm = plot_confusion_matrix(final_metrics["preds"], final_metrics["labels"])
    cm_path = os.path.join(cfg["out_dir"], "confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"✓ Confusion matrix saved at: {cm_path}")

    # Save final metrics
    metrics_dict = {
        "accuracy": float(final_metrics["accuracy"]),
        "precision": float(final_metrics["precision"]),
        "recall": float(final_metrics["recall"]),
        "f1": float(final_metrics["f1"]),
        "best_epoch": best_epoch,
        "best_f1": float(best_f1),
        "class_names": class_names,
    }
    metrics_path = os.path.join(cfg["out_dir"], "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✓ Final metrics saved at: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train classification head on Carinthia dataset using CSV labels"
    )
    parser.add_argument(
        "--yaml_config_path",
        type=str,
        default="./configs/cls_head.yaml",
        help="Path to yaml config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/stage2/stage_2_epoch_500.pth",
        help="Path to MoCo checkpoint file for backbone initialization",
    )

    args = parser.parse_args()

    main(
        yaml_config_path=args.yaml_config_path,
        checkpoint_path=args.checkpoint_path,
        save_best=True,
    )
