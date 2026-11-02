"""
简易检测评估（教学用）：
  - decode_predictions：从三尺度 head 解码 box（简化，非完整 YOLO decode）
  - evaluate_batch / evaluate_loader：按 IoU 匹配计算 Precision / Recall / mAP@0.5 近似

说明：匹配与 decode 为简化实现，数值仅用于同配置下 baseline vs CBAM 对比，
不可与官方 COCO mAP 或 Ultralytics 结果直接横比。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (N,4) b (M,4) xyxy -> (N,M) IoU."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """boxes (...,4) normalized or absolute cxcywh -> xyxy same unit."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def decode_predictions(
    preds: List[torch.Tensor],
    num_classes: int = 2,
    num_anchors: int = 3,
    conf_thres: float = 0.25,
    max_det: int = 100,
) -> List[torch.Tensor]:
    """
    简化解码。每个尺度 pred: (B, na*(5+nc), H, W)
    返回 list[B] of (N,6) = x1,y1,x2,y2,score,cls  （坐标相对 0~1）
    """
    device = preds[0].device
    B = preds[0].shape[0]
    out: List[List[torch.Tensor]] = [[] for _ in range(B)]

    for pred in preds:
        b, _, h, w = pred.shape
        pred = pred.view(b, num_anchors, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2)
        # xy: sigmoid cell offset; wh: softplus-ish via exp of raw for size in grid units then /grid
        xy = pred[..., 0:2].sigmoid()
        wh = pred[..., 2:4].exp().clamp(max=4.0)  # relative to cell; clamp for stability
        obj = pred[..., 4].sigmoid()
        cls = pred[..., 5:].sigmoid()

        gy, gx = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        # cell center in normalized image coords
        cx = (gx[None, None] + xy[..., 0]) / w
        cy = (gy[None, None] + xy[..., 1]) / h
        bw = wh[..., 0] / w
        bh = wh[..., 1] / h

        conf_cls, cls_id = cls.max(dim=-1)
        score = obj * conf_cls

        for bi in range(b):
            mask = score[bi] > conf_thres
            if not mask.any():
                continue
            s = score[bi][mask]
            c = cls_id[bi][mask].float()
            boxes = torch.stack(
                [
                    cx[bi][mask],
                    cy[bi][mask],
                    bw[bi][mask].clamp(0, 1),
                    bh[bi][mask].clamp(0, 1),
                ],
                dim=-1,
            )
            xyxy = cxcywh_to_xyxy(boxes).clamp(0, 1)
            det = torch.cat([xyxy, s.unsqueeze(-1), c.unsqueeze(-1)], dim=-1)
            out[bi].append(det)

    results = []
    for bi in range(B):
        if not out[bi]:
            results.append(torch.zeros((0, 6), device=device))
            continue
        det = torch.cat(out[bi], dim=0)
        # keep top-k by score
        if det.shape[0] > max_det:
            idx = det[:, 4].topk(max_det).indices
            det = det[idx]
        results.append(det)
    return results


def match_and_stats(
    dets: torch.Tensor,
    gts: torch.Tensor,
    iou_thres: float = 0.5,
    num_classes: int = 2,
) -> Tuple[int, int, int, List[Tuple[float, int]]]:
    """
    dets (N,6) xyxy score cls; gts (M,5) cls cx cy w h (norm)
    返回 tp, fp, n_gt, list of (score, tp_flag) for AP
    """
    if gts.numel() == 0:
        return 0, int(dets.shape[0]), 0, [(float(d[4]), 0) for d in dets]
    gt_xyxy = cxcywh_to_xyxy(gts[:, 1:5])
    gt_cls = gts[:, 0].long()
    n_gt = gts.shape[0]
    if dets.numel() == 0:
        return 0, 0, n_gt, []

    order = dets[:, 4].argsort(descending=True)
    dets = dets[order]
    matched = torch.zeros(n_gt, dtype=torch.bool, device=dets.device)
    tp = fp = 0
    scored = []
    for d in dets:
        cls = int(d[5].item())
        ious = box_iou_xyxy(d[:4].unsqueeze(0), gt_xyxy)[0]
        # only same class
        ious = torch.where(gt_cls == cls, ious, torch.zeros_like(ious))
        best = int(ious.argmax().item())
        if ious[best] >= iou_thres and not matched[best]:
            matched[best] = True
            tp += 1
            scored.append((float(d[4]), 1))
        else:
            fp += 1
            scored.append((float(d[4]), 0))
    return tp, fp, n_gt, scored


def average_precision(scored: List[Tuple[float, int]], n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    scored = sorted(scored, key=lambda x: -x[0])
    tps = fps = 0
    precisions = []
    recalls = []
    for _, t in scored:
        if t:
            tps += 1
        else:
            fps += 1
        precisions.append(tps / max(tps + fps, 1))
        recalls.append(tps / n_gt)
    if not precisions:
        return 0.0
    # 11-point style approx
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p = 0.0
        for pr, rc in zip(precisions, recalls):
            if rc >= t:
                p = max(p, pr)
        ap += p
    return ap / 11.0


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    criterion,
    device,
    num_classes: int = 2,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    running = 0.0
    tp = fp = n_gt = 0
    scored_all: List[Tuple[float, int]] = []
    n_batches = 0
    for imgs, targets, _paths in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss, stats = criterion(preds, targets)
        running += stats["loss"]
        n_batches += 1
        dets_batch = decode_predictions(preds, num_classes=num_classes, conf_thres=conf_thres)
        B = imgs.shape[0]
        for bi in range(B):
            if targets.numel():
                g = targets[targets[:, 0] == bi][:, 1:]  # cls cx cy w h
            else:
                g = targets.new_zeros((0, 5))
            tpi, fpi, ngi, scored = match_and_stats(
                dets_batch[bi], g, iou_thres=iou_thres, num_classes=num_classes
            )
            tp += tpi
            fp += fpi
            n_gt += ngi
            scored_all.extend(scored)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    ap50 = average_precision(scored_all, n_gt)
    return {
        "val_loss": running / max(n_batches, 1),
        "precision": precision,
        "recall": recall,
        "mAP50": ap50,
        "tp": float(tp),
        "fp": float(fp),
        "n_gt": float(n_gt),
    }


__all__ = ["evaluate_loader", "decode_predictions", "box_iou_xyxy"]
