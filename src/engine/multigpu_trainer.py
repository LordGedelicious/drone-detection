"""
DistributedDataParallel (DDP) multi-GPU trainer.

NOTE: this path is a bonus deliverable and has not been re-validated since the
evaluation rewrite -- the single-GPU pod it was developed on has one GPU. It
shares the same evaluation implementation as SingleGPUTrainer (rank 0 evaluates
the full, un-sharded val loader). Give it a proper pass before relying on it.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.core.metrics import MetricEvaluator
from src.engine.evaluation import run_evaluation
from src.engine.wnb_tracker import WandbTracker


class MultiGPUTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_sampler: DistributedSampler,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        tracker: WandbTracker,
        local_rank: int,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
        img_size: int = 640,
        grad_clip: float = 10.0,
    ):
        self.local_rank = local_rank
        self.is_main_process = (local_rank == 0)
        self.device = torch.device(f"cuda:{local_rank}")

        self.model = DDP(model.to(self.device), device_ids=[local_rank], output_device=local_rank)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tracker = tracker
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.grad_clip = grad_clip

        self.scaler = torch.amp.GradScaler("cuda")
        self.evaluator = MetricEvaluator(img_size=img_size)
        self.best_mAP = 0.0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        running_loss = running_cls = running_box = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [DDP Train]",
                    disable=not self.is_main_process, leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss_dict["loss/total"]
            running_cls += loss_dict["loss/cls"]
            running_box += loss_dict["loss/box"]

        n = len(self.train_loader)
        stats = torch.tensor([running_loss, running_cls, running_box], device=self.device) / n
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats /= dist.get_world_size()
        return {
            "train/total_loss": stats[0].item(),
            "train/cls_loss": stats[1].item(),
            "train/box_loss": stats[2].item(),
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(self, epoch: int) -> dict:
        if not self.is_main_process:
            return {}
        return run_evaluation(
            self.model.module, self.val_loader, self.evaluator, self.device,
            criterion=self.criterion, desc=f"Epoch {epoch+1} [Val]",
        )

    def fit(self, epochs: int):
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.is_main_process:
                self.tracker.log({**train_metrics, **val_metrics, "epoch": epoch + 1}, step=epoch + 1)
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Train {train_metrics['train/total_loss']:.4f} | "
                    f"mAP@0.5 {val_metrics.get('mAP_50', 0.0):.4f}"
                )
                if val_metrics.get("mAP_50", 0.0) > self.best_mAP:
                    self.best_mAP = val_metrics["mAP_50"]
                    self.tracker.save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model_name": self.model_name,
                            "model_state_dict": self.model.module.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_mAP": self.best_mAP,
                        },
                        checkpoint_dir=self.checkpoint_dir,
                        filename=f"{self.model_name}_best.pth",
                    )
            dist.barrier()

        if self.is_main_process:
            self.tracker.finish()
