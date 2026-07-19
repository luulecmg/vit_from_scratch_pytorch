# vit_from_scratch_pytorch
Implement from scratch Vision Transformer of "An image is worth 16x16 words: Transformers for image recognition at scale." (ICLR 2021)

> Note: This repo is mainly for learning purposes

---
#### Train

```python
python3 -m run
```

Each run is stored under `experiments/<experiment_name>_<timestamp>/`. The
directory name is the run identifier and prefixes every generated artifact:

```text
experiments/run_20260719_165804/
├── run_20260719_165804_config.json
├── run_20260719_165804_train.log
├── run_20260719_165804_visualization.png
└── run_20260719_165804_checkpoints/
    └── run_20260719_165804_epoch_0.pth
```

#### Test code
```python
python3 -m test.test
```
