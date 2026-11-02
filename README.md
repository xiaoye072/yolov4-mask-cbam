# 基于 CBAM 的 YOLOv4 校园口罩佩戴检测

深度学习课程设计：**基于 CBAM 注意力改进的 YOLOv4 校园口罩佩戴检测方法研究**

仓库：https://github.com/xiaoye072/yolov4-mask-cbam

- 教材方法：第 7 章 YOLOv4
- 改进手段：CBAM（教材 7.2.1 CBAM-CSPDarknet53）
- 类别：`with_mask` / `without_mask`
- 状态：已在 **真实 Face Mask Detection 数据** 上完成 baseline 与 CBAM 各 10 epoch CPU 训练。数字只来自 `weights/*/summary.json` 与 `train_log.jsonl`，见 [`results_metrics.md`](./results_metrics.md)。本轮简化评估 **mAP@0.5 = 0**（TP=0），报告中不得改成非零。

## 克隆

```bash
git clone https://github.com/xiaoye072/yolov4-mask-cbam.git
cd yolov4-mask-cbam
```

## 环境

建议 Python 3.8+。

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

有 GPU 时安装对应 CUDA 版 PyTorch，并把 `configs/default.yaml` 里的 `device` 改为 `cuda`（无 GPU 时 `train.py` 会回退到 CPU）。下载数据集时另需 `pip install gdown`。

## 假数据冒烟（不代表真实训练）

仓库不含图片。没有 `data/images/` 时，`train.py` 也会自动改用假数据；下面这条显式走通 1 个 epoch：

```bash
python train.py --epochs 1 --dummy --device cpu
```

## 准备真实数据并做 VOC → YOLO 转换

标签已在仓库中（`data/labels/{train,val,test}`，seed=42 时为 **597 / 170 / 86**）。图片不入库，需要下载后放到同名路径，或重新转换。

1. 下载 Face Mask Detection（Kaggle 或 Pseudo-Lab Drive 镜像）：

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=16Gzn1w38yZLJpzmzKcRIRveFeHZtCEr7" -O Face_Mask_Detection.zip
unzip Face_Mask_Detection.zip -d face-mask-raw
```

原始出处：https://www.kaggle.com/datasets/andrewmvd/face-mask-detection  
说明：[`scripts/prepare_dataset.md`](./scripts/prepare_dataset.md)

2. VOC XML 转 YOLO，并按 seed=42 划分（`mask_weared_incorrect` 并入 `without_mask`）：

```bash
python scripts/convert_voc_to_yolo.py \
  --images face-mask-raw/images \
  --ann face-mask-raw/annotations \
  --out data --seed 42 --train 0.7 --val 0.2 --test 0.1 --copy
```

图片写入 `data/images/{train,val,test}/`（已被 `.gitignore` 排除）。期望划分仍是 train=597 / val=170 / test=86。

## 训练：Baseline 与 CBAM

两侧只差 `--with-cbam`，其余超参与已记录实验一致。`*.pt` 写在本地 `weights/`，不进入 Git。

```bash
# 基线
python train.py --cfg configs/default.yaml --save-dir weights/baseline \
  --epochs 10 --img-size 320 --batch-size 4 --device cpu

# CBAM（骨干 P3/P4/P5 后插入）
python train.py --cfg configs/default.yaml --with-cbam --save-dir weights/cbam \
  --epochs 10 --img-size 320 --batch-size 4 --device cpu
```

每次运行写出 `last.pt`、`best.pt`（按最低 val_loss）、`train_log.jsonl`、`summary.json`。引用指标时只抄这些文件，不要手填 mAP。

## 已跑通实验摘要（勿改数字）

| 模型 | 参数量 | 耗时 (CPU) | best val_loss | 最终 train_loss | mAP@0.5 |
|------|--------|------------|---------------|-----------------|---------|
| Baseline | 14,879,359 | 452.7 s | 0.8532 | 0.8307 | 0.0 |
| +CBAM | 14,922,661 | 493.0 s | 0.8245 | 0.8091 | 0.0 |

来源：`weights/baseline/summary.json`、`weights/cbam/summary.json`。Precision / Recall / mAP 由 `utils/metrics.py` 的简化 decode 统计；10 epoch、CPU、教学损失下 **TP=0**，所以检测指标是 0。可以对比的是参数量和 train/val loss，不能把 mAP 写成非零。逐 epoch 表见 [`results_metrics.md`](./results_metrics.md)。

## 重要说明：教学简化版骨干

`models/yolov4.py` 是 **教学简化版 YOLOv4-like**（`width_mult=0.5`），可读、可训练对比，**不是**逐层复刻官方 CSPDarknet53。

| 模块 | 本仓库（简化） | 与教材/正式版对照 |
|------|----------------|-------------------|
| 骨干 | CSP 风格残差 + 跨阶段拼接，`width_mult=0.5` | 对应 CSPDarknet53 思想 |
| 颈部 | SPP + 简化 FPN/PAN | 对应 SPP+PANet |
| 检测头 | 三尺度 YOLO head | `(5+nc)*na` 输出 |
| CBAM | `models/cbam.py`；骨干 P3/P4/P5 后插入 | 对应教材 7.2.1 |

## 目录结构

```text
.
├── README.md
├── results_metrics.md          # 真实训练数字（mAP=0）
├── 大作业报告-实验数据.md
├── 开题说明.md / 数据方案.md / 报告提纲.md
├── requirements.txt
├── train.py
├── configs/default.yaml
├── models/                     # yolov4.py + cbam.py
├── utils/                      # dataset / loss / metrics
├── scripts/
│   ├── convert_voc_to_yolo.py
│   └── prepare_dataset.md
├── data/
│   ├── classes.names / mask.yaml
│   ├── images/{train,val,test}/   # gitignore，需本地准备
│   ├── labels/{train,val,test}/   # 已入库
│   └── samples/labels/            # 标注样例
└── weights/{baseline,cbam}/       # summary.json、train_log.jsonl、console 日志；*.pt 不入库
```

## 哪些文件会进 Git

保留：代码、配置、脚本、文档、`data/labels/`、`data/samples/`、小型指标文件（`summary.json`、`train_log.jsonl`、`console.log`、`console_excerpt.log`）。

`.gitignore` 排除：`*.pt`、`.venv/`、`data/images/`（仅留 `.gitkeep`）、压缩包与常见大二进制（图片、视频、归档）。

## 文档索引

| 文件 | 用途 |
|------|------|
| `开题说明.md` | 题目与范围 |
| `数据方案.md` | 数据选型 |
| `报告提纲.md` | 报告结构 |
| `大作业报告-实验数据.md` | 真实实验表草稿 |
| `results_metrics.md` | 指标、命令与日志路径 |
| `REPO_PUSH.md` | 本仓库地址与不入库内容 |
