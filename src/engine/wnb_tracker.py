import os
import torch
try:
    import wandb
except ImportError:
    wandb = None


class WandbTracker:
    """
    Experiment tracker wrapper for Weights & Biases logging and checkpoint management.
    """
    def __init__(
        self,
        project_name: str = "drone-detection",
        run_name: str = None,
        config: dict = None,
        enabled: bool = True
    ):
        self.enabled = enabled and (wandb is not None)
        self.run_name = run_name

        if self.enabled:
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config or {}
            )
        else:
            self.run = None

    def log(self, metrics: dict, step: int = None):
        if not self.enabled:
            return
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_summary(self, summary_metrics: dict):
        if not self.enabled:
            return
        for k, v in summary_metrics.items():
            wandb.run.summary[k] = v

    def save_checkpoint(self, state_dict: dict, checkpoint_dir: str, filename: str = "best_model.pth"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, filename)
        torch.save(state_dict, save_path)
        if self.enabled:
            wandb.save(save_path)

    def finish(self):
        if self.enabled:
            wandb.finish()