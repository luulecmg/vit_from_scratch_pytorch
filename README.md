# Vision Transformer from Scratch

A simple Vision Transformer (ViT) implementation in PyTorch, trained on MNIST.

> Note: This repo is intended for educational purposes only. The focus is on understanding the architecture and implementation rather than achieving state-of-the-art performance.

## Train

Install dependencies:

```bash
pip install torch torchvision hydra-core omegaconf tqdm
```

Run training:

```bash
python3 -m run
```

## Change configuration

Configuration is in:

```text
configs/config.yaml
```

Use Hydra overrides to change values:

```bash
# Train for 1 epoch
python3 -m run trainer.max_epochs=1

# Use CPU
python3 -m run trainer.accelerator=cpu

# Change learning rate
python3 -m run trainer.optim.options.lr=0.0001
```

## Outputs

Each training run creates a folder in `experiments/`:

```text
experiments/run_YYYY-MM-DD_HH-MM-SS/
├── checkpoints/    # Saved model checkpoints
├── logs/train.log  # Training and validation logs
└── .hydra/         # Configuration used for the run
```

## Test

Run basic model shape tests:

```bash
python3 -m test.test
```
