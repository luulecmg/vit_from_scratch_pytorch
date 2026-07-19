import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modeling.vit import ViT
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from tqdm import tqdm

class Trainer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = ViT(args)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    
    def train(self, loader):
        self.model.to(self.args.device)
        print(f"--- Start training ---")
        train_loss = []
        train_acc = []
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            epoch_acc = 0

            for images, labels in tqdm(loader, desc=f"Epoch: {epoch + 1}", leave=False):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                logits = self.model(images) # (B, C)

                loss = self.loss_fn(logits, labels)
                epoch_loss += loss.item()

                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1) # (B)

                acc = accuracy_score(labels, preds, normalize=True)
                epoch_acc += acc

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_loss /= len(loader)
            epoch_acc /= len(loader)

            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc: .2f}")
    
    @torch.no_grad()
    def test(self, val_loader):
        val_loss = 0
        val_acc = 0

        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            logits = self.model(images) # (B, C)

            loss = self.loss_fn(logits, labels)
            val_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1) # (B)

            acc = accuracy_score(labels.cpu(), preds.cpu(), normalize=True)
            val_acc += acc
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Test | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

    @torch.no_grad()
    def visualize(self, loader: DataLoader, num_images: int = 4, save_path: str = "visualization.png"):
        self.model.eval()
        self.model.to(self.args.device)

        try:
            images, labels = next(iter(loader))
        except StopIteration:
            print("No data available for visualization.")
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

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"Saved visualization to {save_path}")
