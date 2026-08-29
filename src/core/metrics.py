# THe contents are specifically for model evaluation.
# loss.py are used for model training.

import time
import torch
import numpy as np

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from (x_c, y_c, w, h) to (x1, y1, x2, y2)."""
    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return new_boxes

def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes an N x M IoU matrix between N predicted boxes and M target boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

class MetricEvaluator:
    def __init__(self, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.predictions = []  
        self.ground_truths = {}  
        self.total_gt_boxes = 0
        self.image_counter = 0

    def update(self, pred_boxes: torch.Tensor, pred_confs: torch.Tensor, target_boxes: torch.Tensor):
        self.ground_truths[self.image_counter] = {
            "boxes": xywh_to_xyxy(target_boxes),
            "matched": torch.zeros(target_boxes.shape[0], dtype=torch.bool)
        }
        self.total_gt_boxes += target_boxes.shape[0]

        # Filter out low-confidence predictions before evaluation to speed up math
        valid_mask = pred_confs > self.conf_threshold
        pred_boxes = pred_boxes[valid_mask]
        pred_confs = pred_confs[valid_mask]

        if pred_boxes.shape[0] > 0:
            pred_boxes_xyxy = xywh_to_xyxy(pred_boxes)
            for i in range(pred_boxes.shape[0]):
                self.predictions.append({
                    "img_idx": self.image_counter,
                    "conf": pred_confs[i].item(),
                    "box": pred_boxes_xyxy[i]
                })
        self.image_counter += 1

    def compute_metrics(self) -> dict:
        """Computes Precision, Recall, F1-Score, mAP@0.5, and mAP@0.5:0.95"""
        if len(self.predictions) == 0 or self.total_gt_boxes == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "mAP_50": 0.0, "mAP_50_95": 0.0}

        self.predictions.sort(key=lambda x: x["conf"], reverse=True)
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        ap_per_threshold = []

        for iou_thresh in iou_thresholds:
            TP = np.zeros(len(self.predictions))
            FP = np.zeros(len(self.predictions))
            
            for img_idx in self.ground_truths:
                self.ground_truths[img_idx]["matched"] = torch.zeros(
                    self.ground_truths[img_idx]["boxes"].shape[0], dtype=torch.bool
                )

            for i, pred in enumerate(self.predictions):
                img_idx = pred["img_idx"]
                gt_data = self.ground_truths[img_idx]
                gt_boxes = gt_data["boxes"]

                if gt_boxes.shape[0] == 0:
                    FP[i] = 1
                    continue

                ious = pairwise_box_iou(pred["box"].unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_gt_idx = ious.max(dim=0)

                if best_iou >= iou_thresh and not gt_data["matched"][best_gt_idx]:
                    TP[i] = 1
                    gt_data["matched"][best_gt_idx] = True 
                else:
                    FP[i] = 1

            cum_TP = np.cumsum(TP)
            cum_FP = np.cumsum(FP)
            recalls = cum_TP / self.total_gt_boxes
            precisions = cum_TP / (cum_TP + cum_FP + 1e-6)

            # Area Under Curve (AUC)
            rec_pad = np.concatenate(([0.0], recalls, [1.0]))
            prec_pad = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(len(prec_pad) - 1, 0, -1):
                prec_pad[i - 1] = np.maximum(prec_pad[i - 1], prec_pad[i])

            indices = np.where(rec_pad[1:] != rec_pad[:-1])[0]
            ap = np.sum((rec_pad[indices + 1] - rec_pad[indices]) * prec_pad[indices + 1])
            ap_per_threshold.append(ap)

        final_precision = precisions[-1]
        final_recall = recalls[-1]
        f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-6)

        return {
            "precision": final_precision, 
            "recall": final_recall, 
            "f1_score": f1_score,      
            "mAP_50": ap_per_threshold[0],                 
            "mAP_50_95": np.mean(ap_per_threshold)         
        }
    
    def reset(self):
        self.predictions = []
        self.ground_truths = {}
        self.total_gt_boxes = 0
        self.image_counter = 0

def measure_efficiency(model: torch.nn.Module, device: str, img_size: int = 640) -> dict:
    """
    Calculates parameter count and times a batch of 100 dummy inferences to output FPS.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Model Complexity
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Inference Speed (Warm-up)
    for _ in range(10):
        _ = model(dummy_input)
        
    runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / runs
    
    return {
        "parameters": trainable_params,
        "inference_latency_ms": avg_latency * 1000,
        "fps": 1.0 / avg_latency
    }