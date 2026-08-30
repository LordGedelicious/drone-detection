import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from tqdm import tqdm
from src.core.metrics import MetricEvaluator
from src.engine.wnb_tracker import WandbTracker


class MultiGPUTrainer:
    """
    DistributedDataParallel (DDP) Multi-GPU Trainer orchestrator.
    """
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
        model_name: str = "model"
    ):
        self.local_rank = local_rank
        self.is_main_process = (local_rank == 0)
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Wrap model in DDP
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

        self.scaler = torch.amp.GradScaler("cuda")
        self.evaluator = MetricEvaluator(conf_threshold=0.001)
        self.best_mAP = 0.0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_cls = 0.0
        running_box = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [DDP Train]", disable=not self.is_main_process, leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss_dict["loss/total"]
            running_cls += loss_dict["loss/cls"]
            running_box += loss_dict["loss/box"]

        num_batches = len(self.train_loader)
        loss_tensor = torch.tensor([running_loss, running_cls, running_box], device=self.device) / num_batches
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()

        return {
            "train/total_loss": loss_tensor[0].item(),
            "train/cls_loss": loss_tensor[1].item(),
            "train/box_loss": loss_tensor[2].item(),
            "train/lr": self.optimizer.param_groups[0]["lr"]
        }

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict:
        self.model.eval()
        self.evaluator.reset()
        val_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", disable=not self.is_main_process, leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            val_loss += loss_dict["loss/total"]

            if self.is_main_process:
                batch_size = images.shape[0]
                for b in range(batch_size):
                    tgt_boxes = targets[b]["boxes"].to(self.device)
                    all_boxes, all_confs = [], []
                    for scale_preds in preds:
                        H, W = scale_preds.shape[1], scale_preds.shape[2]
                        grid_y, grid_x = torch.meshgrid(
                            torch.arange(H, device=self.device),
                            torch.arange(W, device=self.device),
                            indexing="ij"
                        )
                        pred_b = scale_preds[b]
                        px = (torch.sigmoid(pred_b[..., 0]) + grid_x) / W
                        py = (torch.sigmoid(pred_b[..., 1]) + grid_y) / H
                        pw = torch.exp(pred_b[..., 2]) / W
                        ph = torch.exp(pred_b[..., 3]) / H
                        conf = torch.sigmoid(pred_b[..., 4])

                        all_boxes.append(torch.stack([px, py, pw, ph], dim=-1).view(-1, 4))
                        all_confs.append(conf.view(-1))

                    comb_boxes = torch.cat(all_boxes, dim=0)
                    comb_confs = torch.cat(all_confs, dim=0)

                    # 1. Pre-filter ultra-low confidences to speed up NMS computation
                    valid_mask = comb_confs > 0.001
                    comb_boxes = comb_boxes[valid_mask]
                    comb_confs = comb_confs[valid_mask]

                    # 2. Apply Non-Maximum Suppression (NMS)
                    if comb_boxes.shape[0] > 0:
                        # NMS requires boxes in (x1, y1, x2, y2) format
                        xyxy_boxes = torch.zeros_like(comb_boxes)
                        xyxy_boxes[:, 0] = comb_boxes[:, 0] - comb_boxes[:, 2] / 2
                        xyxy_boxes[:, 1] = comb_boxes[:, 1] - comb_boxes[:, 3] / 2
                        xyxy_boxes[:, 2] = comb_boxes[:, 0] + comb_boxes[:, 2] / 2
                        xyxy_boxes[:, 3] = comb_boxes[:, 1] + comb_boxes[:, 3] / 2
                        
                        # Suppress overlapping boxes with IoU > 0.45
                        keep_idx = torchvision.ops.nms(xyxy_boxes, comb_confs, iou_threshold=0.45)
                        
                        comb_boxes = comb_boxes[keep_idx]
                        comb_confs = comb_confs[keep_idx]

                    # Reduce probability of GPU OOM issues
                    self.evaluator.update(comb_boxes.detach().cpu(), comb_confs.detach().cpu(), tgt_boxes.detach().cpu())

        metrics = self.evaluator.compute_metrics() if self.is_main_process else {}
        val_loss_tensor = torch.tensor([val_loss / len(self.val_loader)], device=self.device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_avg = (val_loss_tensor / dist.get_world_size()).item()
        metrics["val/total_loss"] = val_loss_avg

        return metrics

    def fit(self, epochs: int):
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.is_main_process:
                log_payload = {**train_metrics, **val_metrics, "epoch": epoch + 1}
                self.tracker.log(log_payload, step=epoch + 1)

                print(
                    f"Epoch [{epoch+1}/{epochs}] | "
                    f"Train Loss: {train_metrics['train/total_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val/total_loss']:.4f} | "
                    f"mAP@0.5: {val_metrics.get('mAP_50', 0.0):.4f} | "
                    f"mAP@0.5:0.95: {val_metrics.get('mAP_50_95', 0.0):.4f}"
                )

                if val_metrics.get("mAP_50", 0.0) > self.best_mAP:
                    self.best_mAP = val_metrics["mAP_50"]
                    self.tracker.save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.module.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_mAP": self.best_mAP
                        },
                        checkpoint_dir=self.checkpoint_dir,
                        filename=f"{self.model_name}_best.pth"
                    )

        if self.is_main_process:
            self.tracker.finish()