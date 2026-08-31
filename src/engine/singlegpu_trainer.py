import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.metrics import MetricEvaluator
from src.engine.evaluation import run_evaluation
from src.engine.wnb_tracker import WandbTracker


class SingleGPUTrainer:
    """Trainer orchestrator for single-GPU training and validation."""

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
        model_name: str = "model",
        img_size: int = 640,
        grad_clip: float = 10.0,
        eval_conf: float = 0.01,
        eval_iou: float = 0.5,
        max_det: int = 300,
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
        self.grad_clip = grad_clip
        self.eval_conf = eval_conf
        self.eval_iou = eval_iou
        self.max_det = max_det

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.evaluator = MetricEvaluator(img_size=img_size)
        self.best_mAP = 0.0
        self._backed_up = set()  # filenames whose pre-existing copy we already moved aside

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = running_cls = running_box = 0.0
        n_ok = 0
        skipped = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                preds = self.model(images)
                loss, loss_dict = self.criterion(preds, targets)

            # A non-finite loss (should not happen now that the box exp() is
            # clamped, but keep the guard) would poison every weight via backward.
            if not torch.isfinite(loss):
                skipped += 1
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss_dict["loss/total"]
            running_cls += loss_dict["loss/cls"]
            running_box += loss_dict["loss/box"]
            n_ok += 1
            pbar.set_postfix({"loss": f"{loss_dict['loss/total']:.4f}"})

        if skipped:
            print(f"[trainer] epoch {epoch+1}: skipped {skipped} non-finite batch(es)")
        n_ok = max(n_ok, 1)
        return {
            "train/total_loss": running_loss / n_ok,
            "train/cls_loss": running_cls / n_ok,
            "train/box_loss": running_box / n_ok,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "train/skipped_batches": skipped,
        }

    def evaluate(self, epoch: int) -> dict:
        return run_evaluation(
            self.model, self.val_loader, self.evaluator, self.device,
            criterion=self.criterion, conf_thresh=self.eval_conf,
            iou_thresh=self.eval_iou, max_det=self.max_det,
            desc=f"Epoch {epoch+1} [Val]",
        )

    def _save(self, epoch: int, filename: str):
        # Never silently clobber a checkpoint from a previous run: the first time
        # this run writes a given filename, move any existing file aside.
        path = os.path.join(self.checkpoint_dir, filename)
        if filename not in self._backed_up:
            self._backed_up.add(filename)
            if os.path.exists(path):
                bak = f"{path}.prev-{time.strftime('%Y%m%d-%H%M%S')}"
                os.rename(path, bak)
                print(f"[trainer] existing {filename} preserved as {os.path.basename(bak)}")

        self.tracker.save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_name": self.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_mAP": self.best_mAP,
            },
            checkpoint_dir=self.checkpoint_dir,
            filename=filename,
        )

    def fit(self, epochs: int):
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.tracker.log({**train_metrics, **val_metrics, "epoch": epoch + 1}, step=epoch + 1)
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train {train_metrics['train/total_loss']:.4f} | "
                f"Val {val_metrics.get('val/total_loss', float('nan')):.4f} | "
                f"mAP@0.5 {val_metrics['mAP_50']:.4f} | "
                f"mAP@0.5:0.95 {val_metrics['mAP_50_95']:.4f} | "
                f"F1 {val_metrics['f1_score']:.4f}"
            )

            self._save(epoch, f"{self.model_name}_last.pth")
            if val_metrics["mAP_50"] > self.best_mAP:
                self.best_mAP = val_metrics["mAP_50"]
                self._save(epoch, f"{self.model_name}_best.pth")

        self.tracker.finish()
