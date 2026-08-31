"""
Single evaluation implementation shared by the trainers and ``eval.py``.

Decoding happens in fp32 (outside autocast) via ``src.core.postprocess`` and the
per-image prediction set is NMS-ed and capped before it reaches the metric, so
evaluation stays on the GPU and the metric only ever sees a few hundred boxes
per image -- not the ~34k raw grid cells the old loop pushed through NumPy.
"""

import torch
from tqdm import tqdm

from src.core.postprocess import decode_predictions, cxcywh_to_xyxy


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader,
    evaluator,
    device: torch.device,
    criterion=None,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.5,
    max_det: int = 300,
    use_amp: bool = True,
    desc: str = "eval",
    progress: bool = True,
    decode=None,
) -> dict:
    model.eval()
    evaluator.reset()
    amp = use_amp and device.type == "cuda"
    # decode: callable(list_of_head_outputs) -> per-image dets. Default = v1 head.
    if decode is None:
        def decode(preds):
            return decode_predictions(preds, conf_thresh, iou_thresh, max_det)

    loss_sum, n_batches = 0.0, 0
    it = tqdm(loader, desc=desc, leave=False) if progress else loader
    for images, targets in it:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            preds = model(images)
            if criterion is not None:
                _, loss_dict = criterion(preds, targets)
                loss_sum += loss_dict["loss/total"]
                n_batches += 1

        dets = decode([p.float() for p in preds])
        for b, det in enumerate(dets):
            gt = targets[b]["boxes"].to(device).reshape(-1, 4)
            gt_xyxy = cxcywh_to_xyxy(gt) if gt.shape[0] else gt
            cond = targets[b].get("meta", {}).get("condition", "all")
            evaluator.update(det["boxes"], det["scores"], gt_xyxy, condition=cond)

    metrics = evaluator.compute_metrics()
    if n_batches:
        metrics["val/total_loss"] = loss_sum / n_batches
    return metrics
