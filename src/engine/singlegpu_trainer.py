import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.core.metrics import MetricEvaluator
from src.engine.wnb_tracker import WandbTracker


class SingleGPUTrainer:
    """
    Trainer orchestrator for single-GPU training and validation.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        tracker: WandbTracker,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tracker = tracker
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.evaluator = MetricEvaluator(conf_threshold=0.25)
        self.best_mAP = 0.0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_box = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
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
            pbar.set_postfix({"loss": f"{loss_dict['loss/total']:.4f}"})

        num_batches = len(self.train_loader)
        return {
            "train/total_loss": running_loss / num_batches,
            "train/cls_loss": running_cls / num_batches,
            "train/box_loss": running_box / num_batches,
            "train/lr": self.optimizer.param_groups[0]["lr"]
        }

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict:
        self.model.eval()
        self.evaluator.reset()
        val_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            val_loss += loss_dict["loss/total"]

            # Evaluate each sample in batch
            batch_size = images.shape[0]
            for b in range(batch_size):
                tgt_boxes = targets[b]["boxes"].to(self.device)
                
                # Flatten multi-scale predictions for decoding
                all_boxes = []
                all_confs = []
                for scale_preds in preds:
                    H, W = scale_preds.shape[1], scale_preds.shape[2]
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(H, device=self.device),
                        torch.arange(W, device=self.device),
                        indexing="ij"
                    )
                    pred_b = scale_preds[b]  # (H, W, 5 + C)
                    px = (torch.sigmoid(pred_b[..., 0]) + grid_x) / W
                    py = (torch.sigmoid(pred_b[..., 1]) + grid_y) / H
                    pw = torch.exp(pred_b[..., 2]) / W
                    ph = torch.exp(pred_b[..., 3]) / H
                    conf = torch.sigmoid(pred_b[..., 4])

                    boxes = torch.stack([px, py, pw, ph], dim=-1).view(-1, 4)
                    confs = conf.view(-1)
                    all_boxes.append(boxes)
                    all_confs.append(confs)

                comb_boxes = torch.cat(all_boxes, dim=0)
                comb_confs = torch.cat(all_confs, dim=0)
                self.evaluator.update(comb_boxes, comb_confs, tgt_boxes)

        metrics = self.evaluator.compute_metrics()
        metrics["val/total_loss"] = val_loss / len(self.val_loader)
        return metrics

    def fit(self, epochs: int):
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            log_payload = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            self.tracker.log(log_payload, step=epoch + 1)

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_metrics['train/total_loss']:.4f} | "
                f"Val Loss: {val_metrics['val/total_loss']:.4f} | "
                f"mAP@0.5: {val_metrics['mAP_50']:.4f} | "
                f"mAP@0.5:0.95: {val_metrics['mAP_50_95']:.4f}"
            )

            # Checkpoint best model
            if val_metrics["mAP_50"] > self.best_mAP:
                self.best_mAP = val_metrics["mAP_50"]
                self.tracker.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_mAP": self.best_mAP
                    },
                    checkpoint_dir=self.checkpoint_dir,
                    filename=f"{self.model_name}_best.pth"
                )

        self.tracker.finish()