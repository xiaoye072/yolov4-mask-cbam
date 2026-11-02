"""
简化检测损失（教学用）：
  - obj / noobj：BCE
  - cls：BCE
  - box：基于预测与 GT 的平滑 L1（cxcywh，网格归一化近似）

完整 CIoU + 匹配策略可后续替换；此处保证 train 一步可反传。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, num_classes: int = 2, num_anchors: int = 3):
        super().__init__()
        self.nc = num_classes
        self.na = num_anchors
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5

    def forward(
        self,
        preds: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        preds: list of (B, na*(5+nc), H, W)
        targets: (N, 6) -> batch_idx, cls, cx, cy, w, h  (相对整图 0~1)
        """
        device = preds[0].device
        total_box = torch.zeros(1, device=device)
        total_obj = torch.zeros(1, device=device)
        total_cls = torch.zeros(1, device=device)
        n_gt = max(int(targets.shape[0]), 1)

        for pred in preds:
            b, _, h, w = pred.shape
            pred = pred.view(b, self.na, 5 + self.nc, h, w).permute(0, 1, 3, 4, 2).contiguous()
            # pred[..., 0:2]=xy, 2:4=wh, 4=obj, 5:=cls  (logits)

            obj_target = torch.zeros_like(pred[..., 4])
            # 粗糙匹配：把每个 GT 投到当前尺度网格，落在中心 cell、anchor0
            if targets.numel():
                for t in targets:
                    bi = int(t[0].item())
                    if bi < 0 or bi >= b:
                        continue
                    cls = int(t[1].item())
                    gx, gy = t[2] * w, t[3] * h
                    gi, gj = int(gx.clamp(0, w - 1).item()), int(gy.clamp(0, h - 1).item())
                    ai = 0
                    obj_target[bi, ai, gj, gi] = 1.0
                    # box：预测用 sigmoid 约束到 cell
                    ptx = pred[bi, ai, gj, gi, 0]
                    pty = pred[bi, ai, gj, gi, 1]
                    ptw = pred[bi, ai, gj, gi, 2]
                    pth = pred[bi, ai, gj, gi, 3]
                    tx = (gx - gi).clamp(0, 1)
                    ty = (gy - gj).clamp(0, 1)
                    tw = t[4].clamp(1e-3).log()  # 简化
                    th = t[5].clamp(1e-3).log()
                    total_box = total_box + F.smooth_l1_loss(
                        torch.stack([ptx.sigmoid(), pty.sigmoid(), ptw, pth]),
                        torch.stack([tx, ty, tw, th]),
                    )
                    if self.nc > 0:
                        cls_target = torch.zeros(self.nc, device=device)
                        if 0 <= cls < self.nc:
                            cls_target[cls] = 1.0
                        total_cls = total_cls + self.bce(pred[bi, ai, gj, gi, 5:], cls_target)

            total_obj = total_obj + self.bce(pred[..., 4], obj_target)

        loss = (
            self.lambda_box * total_box / n_gt
            + self.lambda_obj * total_obj
            + self.lambda_cls * total_cls / n_gt
        )
        stats = {
            "loss": float(loss.detach().item()),
            "box": float(total_box.detach().item()),
            "obj": float(total_obj.detach().item()),
            "cls": float(total_cls.detach().item()),
        }
        return loss, stats


__all__ = ["YOLOLoss"]
