"""
Detection metrics.

``MetricEvaluator`` accumulates per-image results and computes a detection
report: mAP@0.5, mAP@0.5:0.95, precision/recall/F1 at the best-F1 operating
point, AP broken down by drone size and by environment/weather condition, and
the mean absolute error on the per-image drone count.

Design notes
------------
* The IoU matrix between (already NMS-ed, <=max_det) predictions and the few
  ground-truth boxes is computed once per image at ``update`` time on the
  caller's device (``torchvision.ops.box_iou``), then cached as a small numpy
  array. ``compute_metrics`` is therefore pure-numpy and fast -- no giant Python
  loop over tens of thousands of raw grid cells, which is what used to pin the
  CPU during evaluation.
* Size-stratified AP follows the COCO convention: ground truths outside the
  size range are *ignored* (a prediction matching only an ignored GT counts as
  neither TP nor FP).
"""

import time
import numpy as np
import torch
import torchvision

# COCO-style area ranges are calibrated for everyday objects; drones here are
# tiny, so we bucket by box side length (sqrt(area), in pixels at img_size).
SIZE_BUCKETS = {
    "tiny":   (0, 16),      # < 16 px
    "small":  (16, 32),     # 16-32 px
    "medium": (32, 1e9),    # > 32 px
}


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(x_c, y_c, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-points interpolated area under the precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class MetricEvaluator:
    def __init__(self, img_size: int = 640, pr_iou: float = 0.5, count_conf: float = 0.25):
        self.img_size = img_size
        self.pr_iou = pr_iou
        self.count_conf = count_conf
        self.iou_grid = np.round(np.linspace(0.5, 0.95, 10), 2)
        self.reset()

    def reset(self):
        self.images = []          # one record per image
        self.total_gt = 0

    @torch.no_grad()
    def update(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
               gt_boxes: torch.Tensor, condition: str = "all"):
        """
        pred_boxes  : (n, 4) xyxy, any consistent scale (normalized is fine)
        pred_scores : (n,)
        gt_boxes    : (m, 4) xyxy, same scale as pred_boxes
        condition   : env/weather tag for per-condition AP
        """
        n, m = pred_boxes.shape[0], gt_boxes.shape[0]
        if n and m:
            iou = torchvision.ops.box_iou(pred_boxes.float(), gt_boxes.float()).cpu().numpy()
        else:
            iou = np.zeros((n, m), dtype=np.float32)

        if m:
            gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
            gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
            gt_side = torch.sqrt(gw * gh).cpu().numpy() * self.img_size
        else:
            gt_side = np.zeros(0, dtype=np.float32)

        self.images.append({
            "scores": pred_scores.detach().cpu().numpy().astype(np.float64),
            "iou": iou,
            "n_gt": m,
            "gt_side": gt_side,
            "condition": condition,
        })
        self.total_gt += m

    # ------------------------------------------------------------------ core
    def _ap(self, images, iou_thr: float, side_range=None):
        """Returns (ap, recall_curve, precision_curve, sorted_scores)."""
        ignore_per_img, n_gt = [], 0
        entries = []  # (score, img_idx, pred_idx)
        for i, im in enumerate(images):
            if side_range is None:
                ignore = np.zeros(im["n_gt"], dtype=bool)
            else:
                lo, hi = side_range
                ignore = ~((im["gt_side"] >= lo) & (im["gt_side"] < hi))
            ignore_per_img.append(ignore)
            n_gt += int((~ignore).sum())
            for j, s in enumerate(im["scores"]):
                entries.append((s, i, j))

        if n_gt == 0 or not entries:
            return 0.0, np.array([0.0]), np.array([0.0]), np.array([0.0])

        entries.sort(key=lambda e: -e[0])
        matched = [np.zeros(im["n_gt"], dtype=bool) for im in images]
        tp = np.zeros(len(entries))
        fp = np.zeros(len(entries))

        for k, (_, i, j) in enumerate(entries):
            im = images[i]
            if im["n_gt"] == 0:
                fp[k] = 1
                continue
            ious = im["iou"][j].copy()
            ious[matched[i]] = -1.0
            best = int(ious.argmax())
            if ious[best] >= iou_thr:
                if ignore_per_img[i][best]:
                    continue                     # matched an out-of-range GT -> skip
                tp[k] = 1
                matched[i][best] = True
            else:
                fp[k] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        scores = np.array([e[0] for e in entries])
        return _voc_ap(recall, precision), recall, precision, scores

    def _map_range(self, images):
        aps = [self._ap(images, t)[0] for t in self.iou_grid]
        return aps[0], float(np.mean(aps))

    # -------------------------------------------------------------- reporting
    def compute_metrics(self) -> dict:
        base = {
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "best_conf": 0.0,
            "mAP_50": 0.0, "mAP_50_95": 0.0, "count_mae": 0.0,
        }
        if not self.images or self.total_gt == 0:
            return base

        map50, map5095 = self._map_range(self.images)

        # Precision / recall / F1 at the confidence that maximizes F1 (IoU = pr_iou).
        _, rec, prec, scores = self._ap(self.images, self.pr_iou)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        bi = int(f1.argmax())

        # Per-image drone count error at a fixed confidence.
        count_err = [
            abs(int((im["scores"] >= self.count_conf).sum()) - im["n_gt"])
            for im in self.images
        ]

        out = {
            "precision": float(prec[bi]),
            "recall": float(rec[bi]),
            "f1_score": float(f1[bi]),
            "best_conf": float(scores[bi]),
            "mAP_50": float(map50),
            "mAP_50_95": float(map5095),
            "count_mae": float(np.mean(count_err)),
        }

        # AP@0.5 per drone-size bucket (NaN when the split has no GT in a bucket).
        for name, (lo, hi) in SIZE_BUCKETS.items():
            n_in = sum(int(((im["gt_side"] >= lo) & (im["gt_side"] < hi)).sum()) for im in self.images)
            out[f"AP_50/size_{name}"] = (
                self._ap(self.images, 0.5, side_range=(lo, hi))[0] if n_in else float("nan")
            )
            out[f"n_gt/size_{name}"] = n_in

        # mAP per environment/weather condition.
        conds = sorted({im["condition"] for im in self.images if im["condition"] != "all"})
        for c in conds:
            subset = [im for im in self.images if im["condition"] == c]
            out[f"mAP_50/cond_{c}"] = self._map_range(subset)[0]

        return out

    def summary_table(self) -> str:
        m = self.compute_metrics()
        rows = [
            ("mAP@0.5", m["mAP_50"]),
            ("mAP@0.5:0.95", m["mAP_50_95"]),
            (f"precision @F1* (conf={m['best_conf']:.2f})", m["precision"]),
            ("recall @F1*", m["recall"]),
            ("F1 (max)", m["f1_score"]),
            ("count MAE / image", m["count_mae"]),
        ]
        rows += [(k, v) for k, v in m.items() if k.startswith("AP_50/size_")]
        rows += [(k, v) for k, v in m.items() if k.startswith("mAP_50/cond_")]
        w = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{w}} : {v:.4f}" for k, v in rows)


def time_inference(model: torch.nn.Module, device: str, img_size: int = 640,
                   batch_size: int = 1, runs: int = 100, warmup: int = 15) -> dict:
    """Latency percentiles + FPS for a fixed batch size."""
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    is_cuda = torch.device(device).type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    t = np.array(times)
    return {
        "batch_size": batch_size,
        "latency_ms_p50": float(np.percentile(t, 50) * 1000),
        "latency_ms_p95": float(np.percentile(t, 95) * 1000),
        "fps": float(batch_size / t.mean()),
    }
