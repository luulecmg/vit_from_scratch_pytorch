import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.logger import get_logger
from tqdm import tqdm
from pathlib import Path

# Config Schemes
@dataclass
class DistributedConf:
    backend: str = "nccl"
    find_unused_parameters: bool = False

@dataclass
class LoggingConf:
    log_dir: str
    logger_name: str

@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int = 1

class Trainer():
    def __init__(
        self,
        *,
        distributed: Optional[Dict[str, Any]] = None,
        logging: Dict[str, Any],
        data: Dict[str, Any],
        model: Dict[str, Any],
        optim: Dict[str, Any],
        loss: Dict[str, Any],
        accelerator: str = "cuda",
        max_epochs: int = 2,
        val_epoch_freq: int = 1,
        checkpoint: Dict[str, Any],
        skip_saving_ckpts: bool = False
    ):
        self.max_epochs = max_epochs
        self.val_epoch_freq = val_epoch_freq
        self.accelerator = accelerator
        self.skip_saving_ckpts = skip_saving_ckpts

        # 1. Map dataclass schemes
        self.distributed_conf = DistributedConf(**(distributed or {}))
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint)

        # 2. Setup Distributed / Devices
        self._setup_device_and_dist()
        if self.rank == 0:
            os.makedirs(self.logging_conf.log_dir, exist_ok=True)
            os.makedirs(self.checkpoint_conf.save_dir, exist_ok=True)
        
        # 3. Instantiate components via Hydra
        self.model = instantiate(model).to(self.device)
        self.loss_fn = instantiate(loss).to(self.device)

        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.accelerator == "cuda" else None,
                find_unused_parameters=self.distributed_conf.find_unused_parameters,
            )
        
        # 4. Instantiate Datasets & Optimizers
        self._setup_dataset(data)
        self._construct_optimizer(optim)

        self.epoch = 0
        self.logger = get_logger(
            logger_name=self.logging_conf.logger_name,
            log_dir=self.logging_conf.log_dir
        ) 

    def _setup_device_and_dist(self):
        """Init rank, gpu device, and Pytorch Distributed Process"""
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) # used for wrap model into DDP, rank for logging
        else:
            self.rank = 0
            self.local_rank = 0
        
        if self.accelerator =="cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

    def _setup_dataset(self, data_conf: DictConfig):
        self.train_dataset = instantiate(data_conf.train)
        self.val_dataset = instantiate(data_conf.val)

        self.train_sampler = DistributedSampler(self.train_dataset) if self.is_distributed else None
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.is_distributed else None
    
    def _construct_optimizer(self, optim_conf: Dict[str, Any]):
        raw_model = self.model.module if self.is_distributed else self.model
        lr = optim_conf.get("options", {}).get("lr", 1e-3)
        self.optimizer = instantiate(optim_conf["optimizer"], params=raw_model.parameters(), lr=lr)

    def _step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        # get corrects
        preds = logits.argmax(dim=1)
        corrects = (preds == labels).sum().item()

        return loss, corrects, images.shape[0] # why can we call shape?

    def run_train(self):
        while self.epoch < self.max_epochs:
            if self.is_distributed:
                self.train_sampler.set_epoch(self.epoch)

            self.train_loader = DataLoader(
                self.train_dataset, batch_size=128, sampler=self.train_sampler, shuffle=(self.train_sampler is None), num_workers=4
            )
            
            self.model.train()
            total_loss, total_correct, total_samples = 0.0, 0, 0

            for batch in tqdm(self.train_loader, desc="Training loader", leave=False):
                self.optimizer.zero_grad()
                loss, corrects, batch_size = self._step(batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_size
                total_correct += corrects
                total_samples += batch_size
            
            if self.rank == 0:
                self.logger.info(
                    f"[Epoch {self.epoch+1}/{self.max_epochs}] Train loss: {total_loss/total_samples:.4f} "
                    f"| Acc: {total_correct/total_samples:.4f}"
                    )
            
            if (self.epoch + 1) % self.checkpoint_conf.save_freq == 0:
                self.save_checkpoint()
            
            if (self.epoch + 1) % self.val_epoch_freq == 0:
                self.run_val()
            
            self.epoch += 1
    
    def run_val(self):
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=128, sampler=self.val_sampler, shuffle=False, num_workers=4
        )

        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.inference_mode():
            for batch in tqdm(self.val_loader, desc="Validation loading", leave=False):
                loss, corrects, batch_size = self._step(batch)
                total_loss += loss.item() * batch_size
                total_correct += corrects
                total_samples += batch_size
        
        if self.rank == 0:
            self.logger.info(
                f" -> [Val] Loss: {total_loss/total_samples:.4f} "
                f"| Acc: {total_correct/total_samples:.4f}\n"
            )
        
    def save_checkpoint(self):
        if self.skip_saving_ckpts or self.rank != 0:
            return
        
        raw_model = self.model.module if self.is_distributed else self.model
        ckpt_path = os.path.join(self.checkpoint_conf.save_dir, f"vit_mnist_ep{self.epoch+1}.pt")
        tmp_path = f"{ckpt_path}.tmp"

        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch+1
        }, tmp_path)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        os.rename(tmp_path, ckpt_path)
        self.logger.info(f" [+] Saved checkpoint to {ckpt_path}")
    

    def run(self):
        self.logger.info(f"🚀 Start ViT training pipeline with {self.rank}")
            
        self.run_train()
        
        self.logger.info(f"✅ ViT training completed with {self.rank}")