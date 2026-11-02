# 实验结果与指标（真实运行记录）

> 生成时间：2026-11-02 20:12 CST（Asia/Shanghai）  
> **禁止编造**：下表数字均摘自 `weights/*/summary.json` 与 `weights/*/train_log.jsonl`。  
> 模型为教学简化版 YOLOv4-like（`width_mult=0.5`），**非**官方完整 CSPDarknet53。

## 1. 数据集

| 项 | 值 |
|----|----|
| 来源（原始） | Kaggle andrewmvd/face-mask-detection |
| 本实验下载 URL | Google Drive（Pseudo-Lab 镜像）`https://drive.google.com/uc?id=16Gzn1w38yZLJpzmzKcRIRveFeHZtCEr7` |
| 镜像说明 | https://github.com/Pseudo-Lab/Tutorial-Book-Utils `PL_data_loader.py` → FaceMaskDetection |
| 原始规模 | 853 图 + 853 VOC XML |
| 类别处理 | `mask_weared_incorrect` → 并入 `without_mask`（class_id=1） |
| 划分 | seed=42，train/val/test = 0.7/0.2/0.1 |
| 样本数 | **train=597 / val=170 / test=86** |
| 实例数（约） | train objs 2918（with_mask 2313 / without_mask 605）；val 838（650/188）；test 316（269/47） |
| 转换命令 | 见下方 |

```bash
python scripts/convert_voc_to_yolo.py \
  --images /workspace/datasets/face-mask-raw/images \
  --ann /workspace/datasets/face-mask-raw/annotations \
  --out data --seed 42 --train 0.7 --val 0.2 --test 0.1 --copy
```

## 2. 训练配置（两侧一致，仅 CBAM 开关不同）

| 项 | 值 |
|----|----|
| 设备 | CPU（`torch 2.15.0+cpu`，无 CUDA） |
| img_size | 320 |
| batch_size | 4 |
| epochs | 10 |
| lr | 0.001（Adam） |
| weight_decay | 0.0005 |
| seed | 42 |
| width_mult | 0.5 |
| num_workers | 2 |

## 3. 参数量对比

| 模型 | 总参数 | 可训练 | 相对基线 |
|------|--------|--------|----------|
| Baseline（无 CBAM） | **14,879,359** | 14,879,359 | — |
| +CBAM | **14,922,661** | 14,922,661 | +43,302（约 +0.29%） |

来源：训练启动日志 / `summary.json` → `params`。

## 4. 训练命令与耗时

### Baseline

```bash
python train.py --cfg configs/default.yaml --save-dir weights/baseline \
  --epochs 10 --img-size 320 --batch-size 4 --device cpu --num-workers 2
```

| 项 | 值 |
|----|----|
| 耗时 | **452.7 s**（约 7.5 min） |
| 日志 | `weights/baseline/train_log.jsonl` |
| 摘要 | `weights/baseline/summary.json` |
| 控制台摘录 | `weights/baseline/console.log`（`grep '^\[train\]'`） |
| 权重 | `weights/baseline/last.pt`、`best.pt`（按最低 val_loss） |

### CBAM

```bash
python train.py --cfg configs/default.yaml --with-cbam --save-dir weights/cbam \
  --epochs 10 --img-size 320 --batch-size 4 --device cpu --num-workers 2
```

| 项 | 值 |
|----|----|
| 耗时 | **493.0 s**（约 8.2 min） |
| 日志 | `weights/cbam/train_log.jsonl` |
| 摘要 | `weights/cbam/summary.json` |
| 控制台摘录 | `weights/cbam/console.log` |
| 权重 | `weights/cbam/last.pt`、`best.pt` |

## 5. 指标表（摘自真实 log）

### 最终 epoch（epoch 10）

| 模型 | train_loss | val_loss | Precision* | Recall* | mAP@0.5* |
|------|------------|----------|------------|---------|----------|
| Baseline | 0.8307 | 0.8649 | 0.0 | 0.0 | 0.0 |
| CBAM | 0.8091 | **0.8245** | 0.0 | 0.0 | 0.0 |

### 最佳 val_loss（训练过程中）

| 模型 | best_val_loss | 出现于 |
|------|---------------|--------|
| Baseline | **0.8532** | epoch 6（`best.pt`） |
| CBAM | **0.8245** | epoch 10（`best.pt`） |

\* `Precision` / `Recall` / `mAP@0.5` 由 `utils/metrics.py` 简化 decode（非官方 YOLO NMS 全流程）在 **仅 10 epoch、CPU、教学损失** 下统计；本轮 **TP=0**，故检测类指标为 0。报告中应明确：当前可引用对比数字以 **参数量、train/val loss 曲线** 为主；检测 mAP 需更长训练 / GPU / 更完整匹配与 NMS 后重测，**不得伪造非零 mAP**。

### 逐 epoch 摘要（val_loss）

| epoch | Baseline train | Baseline val | CBAM train | CBAM val |
|------:|---------------:|-------------:|-----------:|---------:|
| 1 | 1.3776 | 0.8893 | 1.5027 | 0.8580 |
| 2 | 0.8427 | 1.0799 | 0.8891 | 0.8671 |
| 3 | 0.8130 | 0.8655 | 0.8608 | 0.8884 |
| 4 | 0.8618 | 0.9299 | 0.8086 | 1.0809 |
| 5 | 0.8379 | 1.0893 | 0.8519 | 0.8874 |
| 6 | 0.8257 | **0.8532** | 0.8419 | 0.8407 |
| 7 | 0.8131 | 0.9540 | 0.8157 | 0.8554 |
| 8 | 0.8083 | 1.2292 | 0.8269 | 1.0897 |
| 9 | 0.7959 | 0.9153 | 0.8256 | 0.8277 |
| 10 | 0.8307 | 0.8649 | 0.8091 | **0.8245** |

## 6. 诚实评估（是否达到「基准准确率」）

- **已做到**：真实公开数据、真实反传、baseline 与 CBAM 同配置对比、参数量与 loss 可引用。  
- **尚未做到**：有意义的检测 mAP（本轮简化评估 TP=0）；完整 YOLOv4 骨干 + 预训练；校园实景泛化。  
- **建议**：有 GPU 时将 `img_size=416`、`epochs≥50~100`，并完善 decode/NMS 后再报 mAP；否则报告中以 loss/参数量对比 + 说明限制为准。
