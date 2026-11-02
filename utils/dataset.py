"""
YOLO 格式数据集：images + labels txt
类别：0=with_mask, 1=without_mask
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(
    img,
    new_shape: int = 416,
    color: Tuple[int, int, int] = (114, 114, 114),
):
    """等比缩放 + 灰边填充到正方形。返回 (img, ratio, (pad_w, pad_h))."""
    import numpy as np

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def xywhn_to_xyxy(labels, w: int, h: int):
    """归一化 cx,cy,w,h → 像素 xyxy。"""
    import numpy as np

    out = labels.copy()
    if out.size == 0:
        return out
    cx, cy, bw, bh = out[:, 1], out[:, 2], out[:, 3], out[:, 4]
    out[:, 1] = (cx - bw / 2) * w
    out[:, 2] = (cy - bh / 2) * h
    out[:, 3] = (cx + bw / 2) * w
    out[:, 4] = (cy + bh / 2) * h
    return out


class YOLODataset(Dataset):
    """
    目录约定（与数据方案.md 一致，images/labels 分离）：
      root/
        images/{split}/*.jpg
        labels/{split}/*.txt   # 同行同名

    亦可传入 list_file（每行一张图路径），标签为同目录或平行 labels 路径替换。
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: Optional[str] = None,
        img_size: int = 416,
        augment: bool = False,
        list_file: Optional[str] = None,
    ):
        self.img_size = img_size
        self.augment = augment
        self.samples: List[Tuple[str, str]] = []

        if list_file and os.path.isfile(list_file):
            with open(list_file, "r", encoding="utf-8") as f:
                paths = [ln.strip() for ln in f if ln.strip()]
            for p in paths:
                lp = self._guess_label_path(p, label_dir)
                self.samples.append((p, lp))
        else:
            img_dir_p = Path(img_dir)
            if label_dir is None:
                # images/train -> labels/train
                parts = list(img_dir_p.parts)
                if "images" in parts:
                    i = parts.index("images")
                    parts[i] = "labels"
                    label_dir_p = Path(*parts) if parts[0] != "/" else Path("/").joinpath(*parts[1:])
                    # 更稳妥：
                    label_dir_p = Path(str(img_dir_p).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
                else:
                    label_dir_p = img_dir_p
            else:
                label_dir_p = Path(label_dir)

            for img_path in sorted(img_dir_p.rglob("*")):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = label_dir_p / (img_path.stem + ".txt")
                self.samples.append((str(img_path), str(lbl)))

        if not self.samples:
            # 允许空集：训练脚本可用 dummy；此处不抛错，仅提示
            print(f"[YOLODataset] warning: no images under {img_dir} (list_file={list_file})")

    @staticmethod
    def _guess_label_path(img_path: str, label_dir: Optional[str]) -> str:
        p = Path(img_path)
        if label_dir:
            return str(Path(label_dir) / (p.stem + ".txt"))
        # 同目录 txt，或 images -> labels
        alt = Path(str(p).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))
        alt = alt.with_suffix(".txt")
        same = p.with_suffix(".txt")
        return str(alt if alt.exists() or not same.exists() else same)

    def __len__(self) -> int:
        return max(len(self.samples), 0)

    def _load_image(self, path: str):
        import numpy as np

        if cv2 is not None:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        if Image is not None:
            return np.array(Image.open(path).convert("RGB"))
        raise ImportError("需要 opencv-python 或 Pillow 以读取图像")

    def _load_labels(self, path: str):
        import numpy as np

        if not os.path.isfile(path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                rows.append([float(x) for x in parts])
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def __getitem__(self, idx: int):
        import numpy as np

        img_path, lbl_path = self.samples[idx]
        img = self._load_image(img_path)
        h0, w0 = img.shape[:2]
        labels = self._load_labels(lbl_path)

        img, ratio, (pad_x, pad_y) = letterbox(img, self.img_size)
        # 将归一化框转到 letterbox 后的归一化坐标
        if labels.size:
            # 原图像素 xyxy
            xyxy = xywhn_to_xyxy(labels.copy(), w0, h0)
            xyxy[:, 1] = xyxy[:, 1] * ratio + pad_x
            xyxy[:, 2] = xyxy[:, 2] * ratio + pad_y
            xyxy[:, 3] = xyxy[:, 3] * ratio + pad_x
            xyxy[:, 4] = xyxy[:, 4] * ratio + pad_y
            # 再变回归一化 cxcywh（相对 letterbox 边长）
            s = float(self.img_size)
            cx = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / s
            cy = ((xyxy[:, 2] + xyxy[:, 4]) / 2) / s
            bw = (xyxy[:, 3] - xyxy[:, 1]) / s
            bh = (xyxy[:, 4] - xyxy[:, 2]) / s
            labels = np.stack([labels[:, 0], cx, cy, bw, bh], axis=1).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            if labels.size:
                labels[:, 1] = 1.0 - labels[:, 1]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        targets = torch.from_numpy(labels) if labels.size else torch.zeros((0, 5))
        return img_t, targets, img_path


def collate_fn(batch):
    """返回 imgs (B,3,H,W), targets (N,6) 其中第 0 列为 batch_idx。"""
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = []
    for i, lb in enumerate(labels):
        if lb.numel() == 0:
            continue
        ti = torch.zeros((lb.shape[0], 6))
        ti[:, 0] = i
        ti[:, 1:] = lb
        targets.append(ti)
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    return imgs, targets, paths


class DummyYOLODataset(Dataset):
    """无真实数据时跑通 1 个 epoch 用的假数据。"""

    def __init__(self, n: int = 4, img_size: int = 416, num_classes: int = 2):
        self.n = n
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        img = torch.rand(3, self.img_size, self.img_size)
        # 每张 1~3 个随机框
        k = int(torch.randint(1, 4, (1,)).item())
        labels = torch.zeros((k, 5))
        labels[:, 0] = torch.randint(0, self.num_classes, (k,)).float()
        labels[:, 1:] = torch.rand(k, 4) * 0.5 + 0.25  # 中心附近
        labels[:, 3:] = labels[:, 3:].clamp(0.05, 0.4)
        return img, labels, f"dummy_{idx}.jpg"


__all__ = ["YOLODataset", "DummyYOLODataset", "collate_fn", "letterbox"]
