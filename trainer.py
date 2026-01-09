import torch
import tqdm
import wandb
import os
from utils import global_state


class MoCoTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        epochs,
        steps_per_epoch=None,
        stage=1,
        save_every_epochs=None,
        device=None,
        use_amp=False,
        start_epoch=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.stage = stage
        self.save_every_epochs = save_every_epochs
        self.device = device
        self.use_amp = use_amp
        self.start_epoch = start_epoch

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def eval(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0

        for im_q, im_k in val_loader:
            im_q = im_q.to(self.device)
            im_k = im_k.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits, labels = self.model(im_q, im_k)
                    loss = self.criterion(logits, labels)
            else:
                logits, labels = self.model(im_q, im_k)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)

        wandb.log(
            {
                "eval/loss": avg_loss,
                "eval/epoch": epoch,
            },
            step=global_state.get(),
        )

        return avg_loss

    def train_step(self, im_q, im_k, epoch):
        self.model.train()
        im_q = im_q.to(self.device)
        im_k = im_k.to(self.device)

        if self.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                q, k = self.model(im_q, im_k)
                loss = self.criterion(q, k)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q, k = self.model(im_q, im_k)
            loss = self.criterion(q, k)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            },
            step=global_state.get(),
        )

        return loss.item()

    def train(
        self,
        train_loader,
        val_loader=None,
        eval_every_epochs=1,
        out_dir="./checkpoints/",
    ):
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = 0.0
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")

            for im_q, im_k in progress_bar:
                train_step_loss = self.train_step(im_q, im_k, epoch)
                train_loss += train_step_loss
                progress_bar.set_postfix(loss=train_step_loss)
                global_state.increment()

            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
            wandb.log(
                {"train/epoch_loss": train_loss / len(train_loader)},
                step=global_state.get(),
            )

            if (
                val_loader is not None
                and eval_every_epochs
                and epoch % eval_every_epochs == 0
            ):
                avg_val_loss = self.eval(val_loader, epoch)
                print(f"Validation Loss after Epoch {epoch}: {avg_val_loss:.4f}")
                wandb.log({"val/epoch_loss": avg_val_loss}, step=global_state.get())

            if self.save_every_epochs and epoch % self.save_every_epochs == 0:
                save_path = os.path.join(
                    out_dir, f"stage_{self.stage}_epoch_{epoch}.pth"
                )
                os.makedirs(out_dir, exist_ok=True)
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "global_steps": global_state.get(),
                    },
                    save_path,
                )
                print(f"Model checkpoint saved at: {save_path}")
