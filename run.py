import os
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

def setup_distributed_backend(backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            # world_size=int(os.environ["WORLD_SIZE"]),
            # rank=int(os.environ["RANK"])
        )

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # init gpu (automatically ignore if run 1 normal gpu)
    backend = cfg.trainer.get("distributed", {}).get("backend", "nccl")
    setup_distributed_backend(backend=backend)

    trainer = instantiate(cfg.trainer, _recursive_=False)

    trainer.run()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

"""
NEXT TASKS:
- check how to run on multiple gpus
- re-code again
- add wandb checking
- 
"""