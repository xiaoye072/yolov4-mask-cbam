#!/usr/bin/env python3
"""
训练入口：基线 YOLOv4-like 与 YOLOv4-like+CBAM 可通过 --with-cbam 切换。

示例：
  python train.py --epochs 1 --dummy          # 假数据跑通 1 epoch
  python train.py --cfg configs/default.yaml --save-dir weights/baseline
  python train.py --with-cbam --save-dir weights/cbam --epochs 10

说明：本仓库为教学简化骨干；日志与 results_metrics.md 中的数字须来自真实运行输出。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv4-mask-cbam trainer (teaching skeleton)")
    p.add_argument("--cfg", type=str, default="configs/default.yaml")
    p.add_argument("--with-cbam", action="store_true", help="在骨干插入 CBAM（改进模型）")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--dummy", action="store_true", help="强制使用假数据（至少跑通 1 epoch）")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--no-val", action="store_true", help="跳过验证（仅训练）")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str):
    import torch

    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_loaders(cfg: dict, args, dummy: bool):
    import torch
    from torch.utils.data import DataLoader

    from utils.dataset import YOLODataset, DummyYOLODataset, collate_fn

    img_size = cfg["img_size"]
    train_img = cfg.get("train_images", "data/images/train")
    train_lbl = cfg.get("train_labels", "data/labels/train")
    val_img = cfg.get("val_images", "data/images/val")
    val_lbl = cfg.get("val_labels", "data/labels/val")

    use_dummy = dummy or args.dummy
    if not use_dummy:
        n_imgs = 0
        if os.path.isdir(train_img):
            n_imgs = sum(
                1
                for f in Path(train_img).rglob("*")
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        if n_imgs == 0:
            print("[train] 未找到训练图片，自动切换 DummyYOLODataset")
            use_dummy = True

    nw = int(cfg.get("num_workers", 0))
    if use_dummy:
        ds = DummyYOLODataset(
            n=max(cfg["batch_size"] * 2, 4), img_size=img_size, num_classes=cfg["num_classes"]
        )
        val_ds = DummyYOLODataset(
            n=max(cfg["batch_size"] * 2, 4), img_size=img_size, num_classes=cfg["num_classes"]
        )
    else:
        ds = YOLODataset(train_img, train_lbl, img_size=img_size, augment=True)
        val_ds = YOLODataset(val_img, val_lbl, img_size=img_size, augment=False)

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return loader, val_loader, use_dummy


def main():
    args = parse_args()
    cfg_path = args.cfg if os.path.isabs(args.cfg) else str(ROOT / args.cfg)
    cfg = load_cfg(cfg_path)

    if args.with_cbam:
        cfg["with_cbam"] = True
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.img_size is not None:
        cfg["img_size"] = args.img_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.device is not None:
        cfg["device"] = args.device
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    elif cfg.get("with_cbam") and cfg.get("save_dir") == "weights/baseline":
        cfg["save_dir"] = "weights/cbam"
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.seed is not None:
        cfg["seed"] = args.seed

    set_seed(int(cfg.get("seed", 42)))

    import torch
    from tqdm import tqdm

    from models import build_model
    from utils.loss import YOLOLoss
    from utils.metrics import evaluate_loader

    device = resolve_device(str(cfg.get("device", "cuda")))
    print(f"[train] device={device} with_cbam={cfg.get('with_cbam')} img_size={cfg['img_size']}")

    model = build_model(
        num_classes=cfg["num_classes"],
        with_cbam=bool(cfg.get("with_cbam", False)),
        img_size=cfg["img_size"],
        width_mult=float(cfg.get("width_mult", 0.5)),
    ).to(device)

    params = count_params(model)
    print(f"[train] params total={params['total']} trainable={params['trainable']}")

    loader, val_loader, used_dummy = build_loaders(cfg, args, dummy=False)
    criterion = YOLOLoss(num_classes=cfg["num_classes"], num_anchors=cfg.get("num_anchors", 3))
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    save_dir = ROOT / cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.jsonl"
    summary_path = save_dir / "summary.json"

    # 清掉旧 log（同目录重新训）
    if log_path.exists():
        log_path.unlink()

    epochs = int(cfg["epochs"])
    print(f"[train] epochs={epochs} batches/epoch≈{len(loader)} val_batches≈{len(val_loader)} dummy={used_dummy}")
    print(f"[train] save_dir={save_dir}")

    best_val = float("inf")
    best_map = -1.0
    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=True)
        running = 0.0
        for imgs, targets, _paths in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss, stats = criterion(preds, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += stats["loss"]
            pbar.set_postfix(loss=f"{stats['loss']:.4f}", box=f"{stats['box']:.3f}", obj=f"{stats['obj']:.3f}")

        avg_train = running / max(len(loader), 1)
        row = {
            "epoch": epoch,
            "train_loss": avg_train,
            "with_cbam": bool(cfg.get("with_cbam", False)),
            "params_total": params["total"],
        }

        if not args.no_val:
            metrics = evaluate_loader(
                model,
                val_loader,
                criterion,
                device,
                num_classes=cfg["num_classes"],
                conf_thres=args.conf_thres,
            )
            row.update(metrics)
            print(
                f"[train] epoch {epoch} train_loss={avg_train:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"mAP50={metrics['mAP50']:.4f}"
            )
        else:
            print(f"[train] epoch {epoch} train_loss={avg_train:.4f} (no val)")

        history.append(row)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "with_cbam": bool(cfg.get("with_cbam", False)),
            "num_classes": cfg["num_classes"],
            "cfg": cfg,
            "params": params,
            "metrics": row,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if epoch == 1 or epoch % int(cfg.get("save_period", 10)) == 0 or epoch == epochs:
            torch.save(ckpt, save_dir / f"epoch_{epoch}.pt")

        # best by val_loss（有 val 时）；同时记录 best_map
        if "val_loss" in row and row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"[train] new best.pt (val_loss={best_val:.4f})")
        if "mAP50" in row and row["mAP50"] > best_map:
            best_map = row["mAP50"]
            torch.save(ckpt, save_dir / "best_map.pt")

    elapsed = time.time() - t_start
    summary = {
        "with_cbam": bool(cfg.get("with_cbam", False)),
        "device": str(device),
        "epochs": epochs,
        "batch_size": cfg["batch_size"],
        "img_size": cfg["img_size"],
        "lr": cfg["lr"],
        "seed": cfg.get("seed", 42),
        "dummy": used_dummy,
        "params": params,
        "elapsed_sec": elapsed,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "best_mAP50": best_map if best_map >= 0 else None,
        "final": history[-1] if history else None,
        "log_path": str(log_path),
        "save_dir": str(save_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] done elapsed={elapsed:.1f}s summary -> {summary_path}")
    print("[train] (教学简化骨干；指标为同配置对比用，非官方 COCO mAP。)")


if __name__ == "__main__":
    main()
