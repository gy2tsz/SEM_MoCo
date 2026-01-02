import os
import torch
from model import MoCo
from torchvision import models
import torch.nn as nn
from utils import get_config_hierarchical, init_wandb, print_config, global_state
from dataset import set_seed
from dataset import build_dataloader_from_dir, infinite_loader
from trainer import MoCoTrainer


class MixedLoader:
    """Yield concatenated batches from two loaders to respect a given ratio."""

    def __init__(self, loader_a, loader_b, steps):
        self.loader_a = loader_a
        self.loader_b = loader_b
        self.steps = steps
        self.iter_a = infinite_loader(loader_a)
        self.iter_b = infinite_loader(loader_b)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            q_a, k_a = next(self.iter_a)
            q_b, k_b = next(self.iter_b)
            yield torch.cat([q_a, q_b], dim=0), torch.cat([k_a, k_b], dim=0)


def main():
    # reset global state
    global_state.reset()

    # config
    cfg = get_config_hierarchical("./configs/stage1.yaml")
    print_config(cfg)
    set_seed(cfg["seed"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["use_amp"] and (device.type == "cuda")

    # # dataloaders
    # nffa_bs = int(cfg['total_batch_size'] * cfg['nffa_fraction'])
    # nano_bs = cfg['total_batch_size'] - nffa_bs

    # nffa_train_loader = build_dataloader_from_dir(
    #     data_dir=cfg['nffa_train'],
    #     image_size=cfg['img_size'],
    #     batch_size=nffa_bs,
    #     num_workers=cfg['num_workers'],
    # )

    # nffa_eval_loader = build_dataloader_from_dir(
    #     data_dir=cfg['nffa_eval'],
    #     image_size=cfg['img_size'],
    #     batch_size=nffa_bs,
    #     num_workers=cfg['num_workers'],
    # )

    # nano_train_loader = build_dataloader_from_dir(
    #     data_dir=cfg['nano_train'],
    #     image_size=cfg['img_size'],
    #     batch_size=nano_bs,
    #     num_workers=cfg['num_workers'],
    # )

    # nano_eval_loader = build_dataloader_from_dir(
    #     data_dir=cfg['nano_eval'],
    #     image_size=cfg['img_size'],
    #     batch_size=nano_bs,
    #     num_workers=cfg['num_workers'],
    # )

    # steps_per_epoch = min(len(nffa_train_loader), len(nano_train_loader))
    # val_steps = min(len(nffa_eval_loader), len(nano_eval_loader))
    # total_steps = steps_per_epoch * cfg['epochs']
    # print(f"Total training steps: {total_steps} ({cfg['epochs']} epochs of {steps_per_epoch} steps each)")

    # # training loop
    # train_loader = MixedLoader(nffa_train_loader, nano_train_loader, steps_per_epoch)
    # val_loader = MixedLoader(nffa_eval_loader, nano_eval_loader, val_steps) if val_steps > 0 else None

    train_loader = build_dataloader_from_dir(
        data_dir=cfg["train_stage_1"],
        image_size=cfg["img_size"],
        batch_size=cfg["total_batch_size"],
        num_workers=cfg["num_workers"],
    )

    eval_loader = build_dataloader_from_dir(
        data_dir=cfg["eval_stage_1"],
        image_size=cfg["img_size"],
        batch_size=cfg["total_batch_size"],
        num_workers=cfg["num_workers"],
    )

    init_wandb(cfg, run_name_suffix="stage1")

    # model, optimizer, trainer, etc.
    model = MoCo(
        models.resnet50(weights=None),
        proj_dim=cfg["proj_dim"],
        hidden_dim=cfg["hidden_dim"],
        queue_size=cfg["queue_size"],
        momentum=cfg["momentum"],
        temperature=cfg["temperature"],
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum_sgd"],
        weight_decay=cfg["weight_decay"],
    )

    trainer = MoCoTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        epochs=cfg["epochs"],
        steps_per_epoch=len(train_loader),
        stage=1,
        save_every_epochs=cfg["save_every_epochs"],
        device=device,
        use_amp=use_amp,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=eval_loader,
        eval_every_epochs=cfg["eval_every_epochs"],
    )

    torch.save(
        model.state_dict(), os.path.join(cfg["out_dir"], "moco_stage1_final.pth")
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
