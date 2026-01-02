import random
import shutil
import os
from pathlib import Path

def split_png_dataset(
    src_dir,
    out_dir,
    val_fraction=0.05,
    seed=42
):
    src_dir = Path(os.path.expanduser(src_dir))
    out_dir = Path(os.path.expanduser(out_dir))

    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Collect PNG files
    images = list(src_dir.glob("*.png"))
    assert len(images) > 0, "No PNG images found"

    # Reproducibility
    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * val_fraction)
    eval_imgs = images[:split_idx]
    train_imgs = images[split_idx:]

    # Copy files
    for img in train_imgs:
        shutil.copy(img, train_dir / img.name)

    for img in eval_imgs:
        shutil.copy(img, eval_dir / img.name)

    print(f"Total images : {len(images)}")
    print(f"Train images : {len(train_imgs)}")
    print(f"Eval images  : {len(eval_imgs)}")

if __name__ == "__main__":
    split_png_dataset(
        src_dir="~/Datasets/Carinthia/rgb",
        out_dir="~/Datasets/Carinthia/split_rgb",
        val_fraction=0.05
    )
    print("Dataset split completed.")