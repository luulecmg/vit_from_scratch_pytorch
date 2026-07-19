import argparse
from trainer import Trainer
import torch.nn as nn
from data.mnist_dataset import build_dataloader


def get_args():

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # model
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--in_channel", type=int, default=1)
    parser.add_argument("--embed_channel", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--mlp_scale", type=int, default=4)
    parser.add_argument("--activation", type=nn.Module, default=nn.GELU)
    parser.add_argument("--save_ckpt_dir", type=str, default="./output/checkpoints")

    # optimizer
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=0.05)

    # training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--visualize_num_images", type=int, default=5)
    parser.add_argument("--visualize_path", type=str, default="./output/visualization.png")

    return parser.parse_args()

def main():
    args = get_args()

    train_loader = build_dataloader(args, is_train=True)
    val_loader = build_dataloader(args, is_train=False)

    trainer = Trainer(args)
    trainer.train(train_loader)

    trainer.test(val_loader)

    if args.visualize:
        trainer.visualize(val_loader, num_images=args.visualize_num_images, save_path=args.visualize_path)

    trainer.save_checkpoint()

if __name__ == '__main__':
    main()