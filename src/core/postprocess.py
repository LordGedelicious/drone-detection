"""
Shared prediction decoding + NMS.

Every head in this repo emits raw logits shaped ``(B, H, W, 5 + num_classes)``
with channel layout ``(tx, ty, tw, th, obj, cls...)``. This module turns that
into boxes, entirely on the model's device (GPU), so training-time evaluation,
``eval.py`` and ``infer.py`` all share one code path.

Only the objectness channel is used for the score: the class channels are never
supervised by ``DetectionLoss`` (single-class problem), so ``score = sigmoid(obj)``.
"""

import torch
import torchvision


WH_EXP_CLAMP = 8.0  # exp() exponent ceiling for box w/h decode; keep in sync with src/core/loss.py


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _decode_scale(raw: torch.Tensor, wh_clamp: float = WH_EXP_CLAMP):
    """raw: (B, H, W, 5+C) -> boxes (B, H*W, 4) cxcywh normalized, scores (B, H*W)."""
    B, H, W, _ = raw.shape
    dev = raw.device
    gy, gx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=raw.dtype),
        torch.arange(W, device=dev, dtype=raw.dtype),
        indexing="ij",
    )
    grid = torch.stack((gx, gy), dim=-1)                       # (H, W, 2)
    norm = torch.tensor([W, H], device=dev, dtype=raw.dtype)

    # (2*sigmoid - 0.5) centre parameterisation -- must match DetectionLoss.
    xy = (2.0 * raw[..., 0:2].sigmoid() - 0.5 + grid) / norm
    # clamp the exponent so a diverging model (or fp16 autocast) can't produce inf
    wh = raw[..., 2:4].clamp(max=wh_clamp).exp() / norm
    obj = raw[..., 4].sigmoid()

    boxes = torch.cat([xy, wh], dim=-1).reshape(B, H * W, 4)
    return boxes, obj.reshape(B, H * W)


@torch.no_grad()
def decode_predictions(
    head_outputs: list,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.5,
    max_det: int = 300,
):
    """
    head_outputs : list of (B, H_i, W_i, 5+C) raw logit tensors.
    Returns a list of length B; each item is a dict with
        boxes  : (n, 4) xyxy, normalized to [0, 1]
        scores : (n,) descending
    NMS and the max_det cap run per image, on the input device.

    ``conf_thresh`` should stay low for mAP (a complete PR curve); raise it in
    ``infer.py`` for a clean visual result.
    """
    boxes_all, scores_all = [], []
    for raw in head_outputs:
        b, s = _decode_scale(raw)
        boxes_all.append(b)
        scores_all.append(s)
    boxes = cxcywh_to_xyxy(torch.cat(boxes_all, dim=1)).clamp(0.0, 1.0)  # (B, N, 4)
    scores = torch.cat(scores_all, dim=1)                                # (B, N)
    return _nms_per_image(boxes, scores, conf_thresh, iou_thresh, max_det)


def _nms_per_image(boxes, scores, conf_thresh, iou_thresh, max_det):
    out = []
    for i in range(boxes.shape[0]):
        m = scores[i] > conf_thresh
        bx, sc = boxes[i][m], scores[i][m]
        if bx.numel():
            keep = torchvision.ops.nms(bx, sc, iou_thresh)[:max_det]
            bx, sc = bx[keep], sc[keep]
        else:
            order = sc.argsort(descending=True)
            bx, sc = bx[order], sc[order]
        out.append({"boxes": bx, "scores": sc})
    return out


@torch.no_grad()
def decode_predictions_v2(
    head_outputs: list,
    reg_max: int = 16,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.5,
    max_det: int = 300,
):
    """Decoder for the V2 DecoupledDFLHead. Per-cell channels:
    ``[obj(1), cls(C), box(4*(reg_max+1))]``. Box edges (l,t,r,b, cell units)
    are the softmax-expectation of their distributions (Generalized Focal Loss).
    Score = sigmoid(obj) (the GFL joint quality-class score)."""
    from src.models.v2 import dfl_expectation

    n_box = 4 * (reg_max + 1)
    boxes_all, scores_all = [], []
    for raw in head_outputs:
        B, H, W, _ = raw.shape
        dev = raw.device
        obj = raw[..., 0].sigmoid()                                    # (B,H,W)
        ltrb = dfl_expectation(raw[..., -n_box:].float(), reg_max)      # (B,H,W,4) cell units
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        cxc = (gx + 0.5).to(ltrb.dtype)
        cyc = (gy + 0.5).to(ltrb.dtype)
        x1 = (cxc - ltrb[..., 0]) / W
        y1 = (cyc - ltrb[..., 1]) / H
        x2 = (cxc + ltrb[..., 2]) / W
        y2 = (cyc + ltrb[..., 3]) / H
        boxes_all.append(torch.stack([x1, y1, x2, y2], dim=-1).reshape(B, H * W, 4))
        scores_all.append(obj.reshape(B, H * W))
    boxes = torch.cat(boxes_all, dim=1).clamp(0.0, 1.0)
    scores = torch.cat(scores_all, dim=1)
    return _nms_per_image(boxes, scores, conf_thresh, iou_thresh, max_det)


def letterbox_boxes_to_original(
    boxes_xyxy_norm: torch.Tensor,
    orig_h: int,
    orig_w: int,
    img_size: int = 640,
) -> torch.Tensor:
    """Map boxes in [0,1] (relative to the letterboxed square) back to pixel
    coordinates in the original image. Mirrors LongestMaxSize + centered
    PadIfNeeded from ``dataset._letterbox``."""
    r = img_size / max(orig_h, orig_w)
    new_h, new_w = orig_h * r, orig_w * r
    pad_x = (img_size - new_w) / 2
    pad_y = (img_size - new_h) / 2

    b = boxes_xyxy_norm.clone() * img_size
    b[..., [0, 2]] = (b[..., [0, 2]] - pad_x) / r
    b[..., [1, 3]] = (b[..., [1, 3]] - pad_y) / r
    b[..., [0, 2]] = b[..., [0, 2]].clamp(0, orig_w)
    b[..., [1, 3]] = b[..., [1, 3]].clamp(0, orig_h)
    return b
