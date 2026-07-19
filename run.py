import argparse
import json
from datetime import datetime
from pathlib import Path

import torch.nn as nn

from trainer import Trainer
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
    parser.add_argument("--activation", type=str, default='gelu')

    # optimizer
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=0.05)

    # training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--visualize_num_images", type=int, default=5)

    # output
    parser.add_argument("--experiment_dir", type=str, default="./experiments")
    parser.add_argument("--experiment_name", type=str, default="vit_small")
    parser.add_argument("--config_dir", type=str, default="config.json")
    parser.add_argument("--save_ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--visualize_path", type=str, default="visualization.png")

    return parser.parse_args()


def save_config(args, run_dir: Path):
    config_path = run_dir / f"{args.config_dir}"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)


def main():
    args = get_args()

    experiment_dir = Path(args.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.experiment_name}_{timestamp}"
    run_dir = experiment_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, run_dir)

    train_loader = build_dataloader(args, is_train=True)
    val_loader = build_dataloader(args, is_train=False)

    trainer = Trainer(args, run_dir)
    trainer.train(train_loader)
    trainer.test(val_loader)

    if args.visualize:
        trainer.visualize(val_loader, num_images=args.visualize_num_images, run_dir=run_dir)

    trainer.save_checkpoint(run_dir)

if __name__ == '__main__':
    main()
