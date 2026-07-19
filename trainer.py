import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from modeling.vit import ViT
from utils.logger import get_logger


class Trainer(nn.Module):
    def __init__(self, args, run_dir: Path) -> None:
        super().__init__()
        self.args = args
        self.model = ViT(args)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epoch = 0
    
        self.logger = get_logger("trainer", run_dir)

    def train(self, loader):
        self.model.to(self.args.device)
        self.logger.info("--- Start training ---")
        train_loss = []
        train_acc = []
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            epoch_acc = 0

            for images, labels in tqdm(loader, desc=f"Training epoch {epoch + 1}", leave=False):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                logits = self.model(images) # (B, C)

                loss = self.loss_fn(logits, labels)
                epoch_loss += loss.item()

                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1) # (B)

                acc = accuracy_score(labels.cpu(), preds.cpu(), normalize=True)
                epoch_acc += acc

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.epoch = epoch
            
            epoch_loss /= len(loader)
            epoch_acc /= len(loader)

            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            self.logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} "
                f"| Loss: {epoch_loss:.4f} "
                f"| Acc: {epoch_acc:.2f}"
            )
    
    @torch.no_grad()
    def test(self, loader):
        val_loss = 0
        val_acc = 0

        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            logits = self.model(images) # (B, C)

            loss = self.loss_fn(logits, labels)
            val_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1) # (B)

            acc = accuracy_score(labels.cpu(), preds.cpu(), normalize=True)
            val_acc += acc
        
        val_loss /= len(loader)
        val_acc /= len(loader)

        self.logger.info(
            f"Test | "
            f"Loss: {val_loss:.4f} "
            f"Acc: {val_acc*100:.2f}%"
        )

    @torch.no_grad()
    def visualize(self, loader: DataLoader, run_dir: Path, num_images: int = 4):
        self.model.eval()
        self.model.to(self.args.device)

        try:
            images, labels = next(iter(loader))
        except StopIteration:
            self.logger.warning(
                f"No data available. Skipping visualization"
            )
            return

        images = images[:num_images].to(self.args.device)
        labels = labels[:num_images].to(self.args.device)

        logits = self.model(images)
        preds = logits.argmax(dim=-1)

        fig, axes = plt.subplots(1, len(images), figsize=(2.5 * len(images), 2.5))
        if len(images) == 1:
            axes = [axes]

        for ax, image, label, pred in zip(axes, images.cpu(), labels.cpu(), preds.cpu()):
            image_np = image.squeeze(0).numpy()
            ax.imshow(image_np, cmap="gray")
            ax.set_title(f"Pred: {pred.item()} | True: {label.item()}")
            ax.axis("off")

        visualize_path = run_dir / f"{self.args.visualize_path}"
        visualize_path.parent.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        fig.savefig(visualize_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Saved visualization to {visualize_path}")

    def save_checkpoint(self, run_dir: Path):
        save_dir = run_dir / f"{self.args.save_ckpt_dir}"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"epoch_{self.epoch+1}.pth"

        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_epoch": self.args.epochs
        }, save_path
        )

        self.logger.info(f"Checkpoint saved to {save_path}")
    
