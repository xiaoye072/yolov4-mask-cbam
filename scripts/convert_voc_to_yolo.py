#!/usr/bin/env python3
"""
将 Kaggle andrewmvd/face-mask-detection 的 VOC XML 转为 YOLO txt，
并把 mask_weared_incorrect 并入 without_mask；按 seed 划分 train/val/test。

用法示例：
  python scripts/convert_voc_to_yolo.py \
    --images /path/to/images \
    --ann /path/to/annotations \
    --out data \
    --seed 42 --train 0.7 --val 0.2 --test 0.1

输出：
  data/images/{train,val,test}/*.jpg
  data/labels/{train,val,test}/*.txt
  data/train.txt data/val.txt data/test.txt
  data/classes.names
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# 原三类 → 本课题两类
CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 1,  # 并入 without_mask（见 数据方案.md）
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="原始图片目录")
    p.add_argument("--ann", required=True, help="VOC XML 目录")
    p.add_argument("--out", default="data", help="输出根目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--copy", action="store_true", help="复制图片（默认硬链接，失败则复制）")
    return p.parse_args()


def voc_to_yolo_line(obj, w: int, h: int):
    name = obj.findtext("name", default="").strip()
    if name not in CLASS_MAP:
        return None
    cls_id = CLASS_MAP[name]
    bb = obj.find("bndbox")
    xmin = float(bb.findtext("xmin"))
    ymin = float(bb.findtext("ymin"))
    xmax = float(bb.findtext("xmax"))
    ymax = float(bb.findtext("ymax"))
    cx = ((xmin + xmax) / 2.0) / w
    cy = ((ymin + ymax) / 2.0) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    # clip
    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
    bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_one(xml_path: Path, img_dir: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    lines = []
    for obj in root.findall("object"):
        line = voc_to_yolo_line(obj, w, h)
        if line:
            lines.append(line)
    img_path = img_dir / filename
    if not img_path.is_file():
        # 尝试同 stem 不同扩展名
        stem = Path(filename).stem
        cands = list(img_dir.glob(stem + ".*"))
        img_path = cands[0] if cands else img_path
    return img_path, lines


def place_file(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os_link = getattr(__import__("os"), "link")
        os_link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    img_dir = Path(args.images)
    ann_dir = Path(args.ann)
    out = Path(args.out)

    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        raise SystemExit(f"未找到 XML: {ann_dir}")

    items = []
    for xp in xmls:
        img_path, lines = convert_one(xp, img_dir)
        if not img_path.is_file():
            print(f"[skip] missing image for {xp.name}")
            continue
        if not lines:
            print(f"[skip] empty labels {xp.name}")
            continue
        items.append((img_path, lines))

    random.seed(args.seed)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }
    print(f"total={n} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    (out / "classes.names").parent.mkdir(parents=True, exist_ok=True)
    (out / "classes.names").write_text("with_mask\nwithout_mask\n", encoding="utf-8")

    for split, subset in splits.items():
        list_lines = []
        for img_path, lines in subset:
            stem = img_path.stem
            dst_img = out / "images" / split / (stem + img_path.suffix.lower())
            dst_lbl = out / "labels" / split / (stem + ".txt")
            place_file(img_path, dst_img, copy=args.copy)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            list_lines.append(str(dst_img.resolve()))
        (out / f"{split}.txt").write_text("\n".join(list_lines) + "\n", encoding="utf-8")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
