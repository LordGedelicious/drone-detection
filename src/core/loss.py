import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_pos: int = 1) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        loss = focal_weight * ce_loss
        # Sum all cells and normalize ONLY by the count of positive drone targets
        return loss.sum() / max(num_pos, 1)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Standard Intersection over Union (IoU) between boxes formatted as (x_c, y_c, w, h).
    Loss is calculated as: 1.0 - bbox_iou(...)
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter = inter_w * inter_h

    # Union area
    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou.clamp(0.0, 1.0)

def bbox_iciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Improved Complete IoU (ICIoU) between boxes formatted as (x_c, y_c, w, h).
    Reference: ICIoU (https://ieeexplore.ieee.org/document/9497076).
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter = inter_w * inter_h

    # Union area
    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center point euclidean distance squared
    rho2 = ((box1[..., 0] - box2[..., 0]) ** 2) + ((box1[..., 1] - box2[..., 1]) ** 2)

    # Addition from paper: CIoU Aspect Ratio Term
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # Addition from paper: ICIoU Angle & Scale Regularization Term (Diagonal orientation deviation)
    theta1 = torch.atan(h1 / (w1 + eps))
    theta2 = torch.atan(h2 / (w2 + eps))
    delta_theta = (4 / (math.pi ** 2)) * torch.pow(theta1 - theta2, 2)
    
    with torch.no_grad():
        gamma = delta_theta / (1 - iou + delta_theta + eps)

    # Combine CIoU center & aspect terms with the angle-guided regularization
    iciou = iou - (rho2 / c2 + alpha * v + gamma * delta_theta)
    return iciou.clamp(-1.0, 1.0)

def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Normal Complete IoU (CIoU) between boxes formatted as (x_c, y_c, w, h).
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance squared
    rho2 = ((box1[..., 0] - box2[..., 0]) ** 2) + ((box1[..., 1] - box2[..., 1]) ** 2)

    # Aspect ratio consistency
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou.clamp(-1.0, 1.0)

def bbox_eiou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Efficient IoU (EIoU) Loss.
    Separates aspect ratio penalty into explicit width and height normalized differences.
    Boxes formatted as (x_c, y_c, w, h).
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter = inter_w * inter_h

    # Union area
    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance squared
    rho2_center = ((box1[..., 0] - box2[..., 0]) ** 2) + ((box1[..., 1] - box2[..., 1]) ** 2)

    # Explicit width & height discrepancy squared
    rho2_w = (w1 - w2) ** 2
    rho2_h = (h1 - h2) ** 2

    cw2 = cw ** 2 + eps
    ch2 = ch ** 2 + eps

    # EIoU formula: IoU - (distance_cost + width_cost + height_cost)
    eiou = iou - (rho2_center / c2 + rho2_w / cw2 + rho2_h / ch2)
    return eiou.clamp(-1.0, 1.0)

def bbox_siou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Scylla IoU (SIoU) Loss.
    Combines Angle, Distance, Shape, and IoU costs to align box trajectories.
    Boxes formatted as (x_c, y_c, w, h).
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter = inter_w * inter_h

    # Union area
    w1, h1 = box1[..., 2], box1[..., 3]
    w2, h2 = box2[..., 2], box2[..., 3]
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    # Center offsets
    s_cw = (box2[..., 0] - box1[..., 0])
    s_ch = (box2[..., 1] - box1[..., 1])
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + eps)

    # 1. Angle Cost
    sin_alpha = torch.abs(s_ch) / sigma
    sin_beta = torch.abs(s_cw) / sigma
    sin_alpha = torch.where(sin_alpha > math.sin(math.pi / 4), sin_beta, sin_alpha)
    safe_sin_alpha = sin_alpha.clamp(-1.0 + 1e-4, 1.0 - 1e-4) # If sin_alpha is exactly 1.0, arcsin will return NaN and the gradient will be infinite. Hence, this safeguard.
    angle_cost = 1.0 - 2.0 * torch.pow(torch.sin(torch.arcsin(safe_sin_alpha) - (math.pi / 4)), 2)

    # 2. Distance Cost
    gamma = 2.0 - angle_cost
    rho_x = ((s_cw / (cw + eps)) ** 2)
    rho_y = ((s_ch / (ch + eps)) ** 2)
    dist_cost = 2.0 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

    # 3. Shape Cost
    omega_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + eps)
    omega_h = torch.abs(h1 - h2) / (torch.max(h1, h2) + eps)
    shape_cost = torch.pow(1.0 - torch.exp(-omega_w), 4) + torch.pow(1.0 - torch.exp(-omega_h), 4)

    # SIoU formula
    siou = iou - (dist_cost + shape_cost) * 0.5
    return siou.clamp(-1.0, 1.0)

# Detection loss class that combines focal Loss for classification and IoU-based loss for bounding box regression
# Standard detection loss fail because a model can be overly confident about the presence of an object 
# but still predict a poor bounding box, leading to high classification confidence but low localization accuracy. 
# Hence, detection loss here adds focal loss so actual drone pixels won't get muted during training due to class imbalance.
class DetectionLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        lambda_box: float = 2.0,
        lambda_cls: float = 1.0,
        loss_type: str = "iciou",
        scale_ranges: list = None,
        neighbor_cells: bool = True,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.loss_type = loss_type.lower()
        # When True, each GT is assigned to its centre cell plus the two cardinal
        # neighbour cells it leans toward (YOLOv5-style), ~tripling the positive
        # signal. The centre decode uses the (2*sigmoid - 0.5) form so a cell can
        # place a box centre up to half a cell outside itself -- required for the
        # neighbour cells to reach the target, and harmless for centre-only.
        self.neighbor_cells = neighbor_cells
        self.scale_ranges = scale_ranges or [
            (0.0, 0.10),
            (0.08, 0.25),
            (0.20, 1.00)
        ]

    def forward(self, pred_head_outputs: list, targets: list):
        device = pred_head_outputs[0].device
        batch_size = len(targets)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        num_scales = len(pred_head_outputs)

        # Count total positive drones in this batch
        total_pos_count = sum(t["boxes"].shape[0] for t in targets)

        for scale_idx, preds in enumerate(pred_head_outputs):
            B, H, W, C = preds.shape
            target_obj = torch.zeros((B, H, W), device=device)
            target_boxes = torch.zeros((B, H, W, 4), device=device)
            pos_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

            min_size, max_size = (
                self.scale_ranges[scale_idx] if scale_idx < len(self.scale_ranges) else (0.0, 1.0)
            )

            for b in range(batch_size):
                boxes = targets[b]["boxes"].to(device)
                if boxes.shape[0] == 0:
                    continue

                max_edge = torch.max(boxes[:, 2], boxes[:, 3])
                scale_keep = (max_edge >= min_size) & (max_edge <= max_size)
                filtered_boxes = boxes[scale_keep]
                if filtered_boxes.shape[0] == 0:
                    continue

                cx = (filtered_boxes[:, 0] * W).clamp(0, W - 1e-4)
                cy = (filtered_boxes[:, 1] * H).clamp(0, H - 1e-4)
                gx = cx.long()
                gy = cy.long()

                for i in range(len(filtered_boxes)):
                    xi, yi = int(gx[i]), int(gy[i])
                    cells = [(xi, yi)]
                    if self.neighbor_cells:
                        # fractional position inside the centre cell, in [0, 1)
                        fx = float(cx[i] - xi)
                        fy = float(cy[i] - yi)
                        nx = xi + (-1 if fx < 0.5 else 1)
                        ny = yi + (-1 if fy < 0.5 else 1)
                        if 0 <= nx < W:
                            cells.append((nx, yi))
                        if 0 <= ny < H:
                            cells.append((xi, ny))
                    for xx, yy in cells:
                        target_obj[b, yy, xx] = 1.0
                        target_boxes[b, yy, xx] = filtered_boxes[i]
                        pos_mask[b, yy, xx] = True

            total_cls_loss += self.focal_loss(preds[..., 4], target_obj, num_pos=total_pos_count)

            if pos_mask.sum() > 0:
                b_idx, y_idx, x_idx = torch.nonzero(pos_mask, as_tuple=True)
                pred_pos_raw = preds[pos_mask][..., :4]
                target_pos = target_boxes[pos_mask]

                # (2*sigmoid - 0.5): centre can land in [-0.5, 1.5) cells -> a
                # neighbour cell can reach the true centre. Matches _decode_scale
                # in src/core/postprocess.py.
                px = (2.0 * torch.sigmoid(pred_pos_raw[:, 0]) - 0.5 + x_idx.float()) / W
                py = (2.0 * torch.sigmoid(pred_pos_raw[:, 1]) - 0.5 + y_idx.float()) / H
                # Clamp the exponent: under fp16 autocast exp(>11) overflows to inf,
                # and inf/inf in the CIoU aspect term becomes NaN -> whole loss NaN.
                # (Matches WH_EXP_CLAMP in src/core/postprocess.py.)
                pw = torch.exp(pred_pos_raw[:, 2].clamp(max=8.0)) / W
                ph = torch.exp(pred_pos_raw[:, 3].clamp(max=8.0)) / H
                pred_boxes = torch.stack([px, py, pw, ph], dim=-1)

                if self.loss_type == "eiou":
                    iou_loss = 1.0 - bbox_eiou(pred_boxes, target_pos)
                elif self.loss_type == "siou":
                    iou_loss = 1.0 - bbox_siou(pred_boxes, target_pos)
                elif self.loss_type == "ciou":
                    iou_loss = 1.0 - bbox_ciou(pred_boxes, target_pos)
                elif self.loss_type == "iciou":
                    iou_loss = 1.0 - bbox_iciou(pred_boxes, target_pos)
                else:
                    iou_loss = 1.0 - bbox_iou(pred_boxes, target_pos)

                # Mean over assigned cells (not GT count) so the box/cls balance
                # is the same whether or not neighbour cells are assigned.
                total_box_loss += iou_loss.mean()

        total_loss = (self.lambda_cls * (total_cls_loss / num_scales)) + (
            self.lambda_box * (total_box_loss / num_scales)
        )

        return total_loss, {
            "loss/total": total_loss.item(),
            "loss/cls": (total_cls_loss / num_scales).item(),
            "loss/box": (total_box_loss / num_scales).item(),
        }


# =============================================================================
# V2 objective: Generalized Focal Loss (Li et al., NeurIPS 2020)
#   QFL  — quality-focal objectness, soft target = IoU(pred, gt)
#   DFL  — distribution focal loss on the 4 box-edge distributions
#   CIoU — on the DFL-decoded box
# For the DecoupledDFLHead in src/models/v2.py.
# Head channel layout per cell: [obj(1), cls(C), box(4*(reg_max+1))].
# =============================================================================
from src.models.v2 import dfl_expectation  # noqa: E402


def _xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]],
        dim=-1,
    )


class DetectionLossV2(nn.Module):
    def __init__(
        self,
        reg_max: int = 16,
        scale_ranges: list = None,
        neighbor_cells: bool = True,
        lambda_qfl: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_box: float = 2.0,
        lambda_dfl: float = 0.5,
        beta: float = 2.0,
    ):
        super().__init__()
        self.reg_max = reg_max
        self.neighbor_cells = neighbor_cells
        self.scale_ranges = scale_ranges or [(0.0, 1.0)]
        self.l_qfl, self.l_cls, self.l_box, self.l_dfl, self.beta = (
            lambda_qfl, lambda_cls, lambda_box, lambda_dfl, beta,
        )

    def _dfl_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits (N, 4, reg_max+1); target (N, 4) in [0, reg_max]
        dl = target.floor().long().clamp(0, self.reg_max - 1)
        dr = dl + 1
        wl = dr.float() - target
        wr = target - dl.float()
        logits = logits.reshape(-1, self.reg_max + 1)
        dl, dr, wl, wr = dl.reshape(-1), dr.reshape(-1), wl.reshape(-1), wr.reshape(-1)
        loss = (F.cross_entropy(logits, dl, reduction="none") * wl
                + F.cross_entropy(logits, dr, reduction="none") * wr)
        return loss.mean()

    def forward(self, pred_head_outputs: list, targets: list):
        device = pred_head_outputs[0].device
        n_gt = max(sum(t["boxes"].shape[0] for t in targets), 1)
        num_scales = len(pred_head_outputs)
        reg_ch = 4 * (self.reg_max + 1)

        acc = {"qfl": torch.zeros((), device=device), "cls": torch.zeros((), device=device),
               "box": torch.zeros((), device=device), "dfl": torch.zeros((), device=device)}

        for si, p in enumerate(pred_head_outputs):
            B, H, W, C = p.shape
            n_classes = C - 1 - reg_ch
            obj_logit = p[..., 0]
            cls_logit = p[..., 1:1 + n_classes]
            box_logit = p[..., 1 + n_classes:]
            mn, mx = self.scale_ranges[si] if si < len(self.scale_ranges) else (0.0, 1.0)

            obj_target = torch.zeros((B, H, W), device=device)
            pos_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
            gt_cell = torch.zeros((B, H, W, 4), device=device)  # normalized xyxy

            for b in range(B):
                gt = targets[b]["boxes"].to(device)
                if gt.numel() == 0:
                    continue
                max_edge = torch.max(gt[:, 2], gt[:, 3])
                gt = gt[(max_edge >= mn) & (max_edge <= mx)]
                if gt.numel() == 0:
                    continue
                cx = (gt[:, 0] * W).clamp(0, W - 1e-4)
                cy = (gt[:, 1] * H).clamp(0, H - 1e-4)
                gx, gy = cx.long(), cy.long()
                gt_xyxy = torch.stack([gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2,
                                       gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], dim=-1)
                for i in range(gt.shape[0]):
                    xi, yi = int(gx[i]), int(gy[i])
                    cells = [(xi, yi)]
                    if self.neighbor_cells:
                        fx, fy = float(cx[i] - xi), float(cy[i] - yi)
                        nx = xi + (-1 if fx < 0.5 else 1)
                        ny = yi + (-1 if fy < 0.5 else 1)
                        if 0 <= nx < W:
                            cells.append((nx, yi))
                        if 0 <= ny < H:
                            cells.append((xi, ny))
                    for xx, yy in cells:
                        obj_target[b, yy, xx] = 1.0
                        pos_mask[b, yy, xx] = True
                        gt_cell[b, yy, xx] = gt_xyxy[i]

            if pos_mask.any():
                b_i, y_i, x_i = torch.nonzero(pos_mask, as_tuple=True)
                bl = box_logit[pos_mask]                             # (n_pos, 4*(rm+1))
                ltrb = dfl_expectation(bl, self.reg_max)             # (n_pos, 4) cell units
                cxg = x_i.float() + 0.5
                cyg = y_i.float() + 0.5
                pred_xyxy = torch.stack([(cxg - ltrb[:, 0]) / W, (cyg - ltrb[:, 1]) / H,
                                         (cxg + ltrb[:, 2]) / W, (cyg + ltrb[:, 3]) / H], dim=-1)
                tgt_xyxy = gt_cell[pos_mask]
                iou = bbox_ciou(_xyxy_to_cxcywh(pred_xyxy), _xyxy_to_cxcywh(tgt_xyxy)).clamp(-1, 1)
                acc["box"] = acc["box"] + (1.0 - iou).mean()

                tgt_ltrb = torch.stack([cxg - tgt_xyxy[:, 0] * W, cyg - tgt_xyxy[:, 1] * H,
                                        tgt_xyxy[:, 2] * W - cxg, tgt_xyxy[:, 3] * H - cyg], dim=-1)
                tgt_ltrb = tgt_ltrb.clamp(0, self.reg_max - 0.01)
                acc["dfl"] = acc["dfl"] + self._dfl_loss(
                    bl.reshape(-1, 4, self.reg_max + 1), tgt_ltrb)

                # soft objectness target = localisation quality
                obj_target[b_i, y_i, x_i] = iou.detach().clamp(0, 1)
                acc["cls"] = acc["cls"] + F.binary_cross_entropy_with_logits(
                    cls_logit[pos_mask], torch.ones_like(cls_logit[pos_mask]), reduction="mean")

            p_obj = torch.sigmoid(obj_logit)
            qfl = ((obj_target - p_obj).abs().pow(self.beta)
                   * F.binary_cross_entropy_with_logits(obj_logit, obj_target, reduction="none"))
            acc["qfl"] = acc["qfl"] + qfl.sum() / n_gt

        total = (self.l_qfl * acc["qfl"] + self.l_cls * acc["cls"]
                 + self.l_box * acc["box"] + self.l_dfl * acc["dfl"]) / num_scales
        return total, {
            "loss/total": total.item(),
            "loss/cls": ((self.l_qfl * acc["qfl"] + self.l_cls * acc["cls"]) / num_scales).item(),
            "loss/box": ((self.l_box * acc["box"] + self.l_dfl * acc["dfl"]) / num_scales).item(),
        }


# =============================================================================
# V3 objective: Task-Aligned Loss for FinalDetector.
#   dynamic task-aligned assignment (TOOD / YOLOv8) + Varifocal objectness
#   (VarifocalNet) + alignment-weighted DFL + MPDIoU (or CIoU) box term.
# Head channel layout per cell: [obj(1), cls(C), box(4*(reg_max+1))].
# =============================================================================
def _pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(n,4),(m,4) xyxy -> (n,m) IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def bbox_mpdiou(p: torch.Tensor, t: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Elementwise MPDIoU for matched (n,4) xyxy pairs, normalised coords
    (image diagonal squared = 1^2 + 1^2 = 2)."""
    x1 = torch.max(p[:, 0], t[:, 0]); y1 = torch.max(p[:, 1], t[:, 1])
    x2 = torch.min(p[:, 2], t[:, 2]); y2 = torch.min(p[:, 3], t[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    ap = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    at = (t[:, 2] - t[:, 0]).clamp(0) * (t[:, 3] - t[:, 1]).clamp(0)
    iou = inter / (ap + at - inter + eps)
    d1 = (p[:, 0] - t[:, 0]) ** 2 + (p[:, 1] - t[:, 1]) ** 2
    d2 = (p[:, 2] - t[:, 2]) ** 2 + (p[:, 3] - t[:, 3]) ** 2
    return iou - (d1 + d2) / 2.0


class TaskAlignedLoss(nn.Module):
    def __init__(self, reg_max: int = 16, num_classes: int = 1,
                 topk: int = 10, alpha: float = 0.5, beta: float = 6.0,
                 iou_type: str = "ciou",
                 w_obj: float = 1.0, w_cls: float = 0.5, w_box: float = 2.5, w_dfl: float = 0.5,
                 vfl_alpha: float = 0.75, vfl_gamma: float = 2.0):
        super().__init__()
        self.reg_max = reg_max
        self.nc = num_classes
        self.topk, self.alpha, self.beta = topk, alpha, beta
        self.iou_type = iou_type
        self.w_obj, self.w_cls, self.w_box, self.w_dfl = w_obj, w_cls, w_box, w_dfl
        self.vfl_alpha, self.vfl_gamma = vfl_alpha, vfl_gamma

    # ---- flatten heads into one cell list, with per-cell grid geometry ----
    def _flatten(self, preds):
        obj, cls, box, cxy, gw = [], [], [], [], []
        for p in preds:
            B, H, W, C = p.shape
            dev = p.device
            gy, gx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
            cx = ((gx + 0.5) / W).reshape(-1)
            cy = ((gy + 0.5) / H).reshape(-1)
            obj.append(p[..., 0].reshape(B, -1))
            cls.append(p[..., 1:1 + self.nc].reshape(B, -1, self.nc))
            box.append(p[..., 1 + self.nc:].reshape(B, -1, 4 * (self.reg_max + 1)))
            cxy.append(torch.stack([cx, cy], -1))                 # (HW, 2)
            gw.append(torch.full((H * W,), float(W), device=dev)) # grid width per cell
        return (torch.cat(obj, 1), torch.cat(cls, 1), torch.cat(box, 1),
                torch.cat(cxy, 0), torch.cat(gw, 0))

    def _decode_xyxy(self, box_logits, cxy, gw):
        """box_logits (N, 4*(rm+1)); returns (N,4) xyxy normalised."""
        ltrb = dfl_expectation(box_logits, self.reg_max) / gw[:, None]   # cell units -> normalised
        x1 = cxy[:, 0] - ltrb[:, 0]; y1 = cxy[:, 1] - ltrb[:, 1]
        x2 = cxy[:, 0] + ltrb[:, 2]; y2 = cxy[:, 1] + ltrb[:, 3]
        return torch.stack([x1, y1, x2, y2], -1)

    @torch.no_grad()
    def _assign(self, pred_xyxy, scores, gt_xyxy, cxy):
        N, M = pred_xyxy.shape[0], gt_xyxy.shape[0]
        dev = pred_xyxy.device
        # cell centre inside GT box
        lt = cxy[:, None, :] - gt_xyxy[None, :, :2]
        rb = gt_xyxy[None, :, 2:] - cxy[:, None, :]
        in_gt = torch.cat([lt, rb], -1).amin(-1) > 1e-6              # (N, M)
        iou = _pairwise_iou_xyxy(pred_xyxy, gt_xyxy)                 # (N, M)
        align = scores[:, None].clamp(min=1e-6) ** self.alpha * iou.clamp(min=0) ** self.beta
        align = align * in_gt
        is_pos = torch.zeros(N, M, dtype=torch.bool, device=dev)
        k = min(self.topk, N)
        topv, topi = align.topk(k, dim=0)                           # (k, M)
        for j in range(M):
            sel = topi[topv[:, j] > 0, j]
            is_pos[sel, j] = True
        is_pos &= in_gt
        # resolve multi-assignment by max IoU
        multi = is_pos.sum(1) > 1
        if multi.any():
            best = iou.argmax(1)
            is_pos[multi] = False
            is_pos[multi, best[multi]] = True
        fg = is_pos.any(1)
        if not fg.any():
            return fg, torch.zeros(N, dtype=torch.long, device=dev), torch.zeros(N, device=dev)
        gt_idx = is_pos.float().argmax(1)
        # per-GT normalised alignment target (YOLOv8): align / align_max * iou_max
        align_pos = align * is_pos
        norm = align_pos / (align_pos.amax(0, keepdim=True) + 1e-9) * (iou * is_pos).amax(0, keepdim=True)
        t_norm = norm.amax(1)
        return fg, gt_idx, t_norm

    def _dfl(self, box_logits, tgt_ltrb):
        # box_logits (n, 4*(rm+1)); tgt_ltrb (n,4) in cell units, clamped
        dl = tgt_ltrb.clamp(0, self.reg_max - 0.01)
        lo = dl.floor().long(); hi = lo + 1
        wl = (hi.float() - dl); wr = (dl - lo.float())
        logits = box_logits.reshape(-1, 4, self.reg_max + 1).reshape(-1, self.reg_max + 1)
        lo = lo.reshape(-1); hi = hi.reshape(-1); wl = wl.reshape(-1); wr = wr.reshape(-1)
        return (F.cross_entropy(logits, lo, reduction="none") * wl
                + F.cross_entropy(logits, hi, reduction="none") * wr).reshape(-1, 4).mean(1)

    def forward(self, preds, targets):
        dev = preds[0].device
        obj_l, cls_l, box_l, cxy, gw = self._flatten(preds)
        B = obj_l.shape[0]
        iou_fn = bbox_mpdiou if self.iou_type == "mpdiou" else None

        L_obj = torch.zeros((), device=dev)
        L_cls = torch.zeros((), device=dev)
        L_box = torch.zeros((), device=dev)
        L_dfl = torch.zeros((), device=dev)
        total_norm = 0.0

        for b in range(B):
            gt = targets[b]["boxes"].to(dev)
            obj_target = torch.zeros_like(obj_l[b])
            if gt.numel():
                gt_xyxy = torch.stack([gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2,
                                       gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], -1).clamp(0, 1)
                with torch.no_grad():
                    pxyxy = self._decode_xyxy(box_l[b], cxy, gw)
                    scores = obj_l[b].sigmoid()
                    fg, gt_idx, t_norm = self._assign(pxyxy, scores, gt_xyxy, cxy)
                obj_target[fg] = t_norm[fg].to(obj_target.dtype)
                if fg.any():
                    w = t_norm[fg].clamp(min=1e-6)
                    total_norm += float(w.sum())
                    pb = self._decode_xyxy(box_l[b][fg], cxy[fg], gw[fg])
                    tb = gt_xyxy[gt_idx[fg]]
                    if iou_fn is None:
                        iou = bbox_ciou(_xyxy_to_cxcywh(pb), _xyxy_to_cxcywh(tb)).clamp(-1, 1)
                    else:
                        iou = iou_fn(pb, tb).clamp(-1, 1)
                    L_box = L_box + ((1.0 - iou) * w).sum()
                    # DFL target: cell-centre -> GT edges, in cell units
                    g = gw[fg]
                    tl = torch.stack([(cxy[fg][:, 0] - tb[:, 0]) * g, (cxy[fg][:, 1] - tb[:, 1]) * g,
                                      (tb[:, 2] - cxy[fg][:, 0]) * g, (tb[:, 3] - cxy[fg][:, 1]) * g], -1)
                    L_dfl = L_dfl + (self._dfl(box_l[b][fg], tl) * w).sum()
                    L_cls = L_cls + F.binary_cross_entropy_with_logits(
                        cls_l[b][fg], torch.ones_like(cls_l[b][fg]), reduction="sum")

            # Varifocal objectness over all cells
            p = obj_l[b].sigmoid()
            bce = F.binary_cross_entropy_with_logits(obj_l[b], obj_target, reduction="none")
            weight = torch.where(obj_target > 0, obj_target,
                                 self.vfl_alpha * p.pow(self.vfl_gamma))
            L_obj = L_obj + (weight * bce).sum()

        n = max(total_norm, 1.0)
        n_pos = max(sum((t["boxes"].shape[0] for t in targets)), 1)
        L_obj = L_obj / n
        L_box = L_box / n
        L_dfl = L_dfl / n
        L_cls = L_cls / n

        total = self.w_obj * L_obj + self.w_cls * L_cls + self.w_box * L_box + self.w_dfl * L_dfl
        return total, {
            "loss/total": total.item(),
            "loss/cls": (self.w_obj * L_obj + self.w_cls * L_cls).item(),
            "loss/box": (self.w_box * L_box + self.w_dfl * L_dfl).item(),
        }
