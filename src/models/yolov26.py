import os
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOv26Benchmark(nn.Module):
    """
    Wrapper for Ultralytics YOLO models to serve as a pre-trained benchmark/baseline.
    """
    def __init__(self, model_weight: str = "yolov8n.pt", num_classes: int = 1):
        super().__init__()
        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            )
        self.model_weight = model_weight
        self.num_classes = num_classes
        self.model = YOLO(model_weight)

    def train_benchmark(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
        project: str = "drone-detection",
        name: str = "yolo_benchmark"
    ):
        """
        Executes pre-trained fine-tuning using the Ultralytics engine.
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name
        )
        return results

    def forward(self, x: torch.Tensor):
        """
        Runs standard inference.
        """
        return self.model(x)

    def evaluate(self, data_yaml: str, imgsz: int = 640, device: str = "0"):
        """
        Runs standard validation to retrieve mAP50 and mAP50-95.
        """
        metrics = self.model.val(data=data_yaml, imgsz=imgsz, device=device)
        return {
            "mAP_50": metrics.box.map50,
            "mAP_50_95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr
        }


if __name__ == "__main__":
    # Test initialization
    print("Testing YOLO Benchmark wrapper initialization...")
    benchmark = YOLOv26Benchmark(model_weight="yolov8n.pt")
    print("YOLO Benchmark wrapper initialized successfully.")