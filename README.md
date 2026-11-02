# 基于 CBAM 的 YOLOv4 校园口罩佩戴检测

深度学习课程设计：**基于 CBAM 注意力改进的 YOLOv4 校园口罩佩戴检测方法研究**

- 教材方法：第 7 章 YOLOv4  
- 改进手段：CBAM（教材 7.2.1 CBAM-CSPDarknet53）  
- 类别：`with_mask` / `without_mask`  
- 状态：已在 **真实 Face Mask Detection 数据** 上完成 baseline 与 CBAM 各 10 epoch CPU 训练；指标见 [`results_metrics.md`](./results_metrics.md)
- **课程报告**：见 [`大作业报告.md`](./大作业报告.md)（完整可提交正文）

## 重要说明：教学简化版骨干

本仓库 `models/yolov4.py` 实现的是 **「教学简化版 YOLOv4-like」**，结构完整可读、可训练对比，但**不是**逐层复刻官方 CSPDarknet53 全宽网络：

| 模块 | 本仓库（简化） | 与教材/正式版对照 |
|------|----------------|-------------------|
| 骨干 | CSP 风格残差 + 跨阶段拼接，`width_mult=0.5` | 对应 CSPDarknet53 思想 |
| 颈部 | SPP + 简化 FPN/PAN | 对应 SPP+PANet |
| 检测头 | 三尺度 YOLO head | `(5+nc)*na` 输出 |
| CBAM | `models/cbam.py`；骨干 P3/P4/P5 后插入（【CBAM 插入点】） | 对应教材 7.2.1 |

## 目录结构

```text
yolov4-mask-cbam/
├── README.md
├── results_metrics.md       # 真实训练数字
├── 大作业报告-实验数据.md
├── 开题说明.md / 数据方案.md / 报告提纲.md
├── requirements.txt
├── train.py                 # 训练（--with-cbam / 验证 loss + 简化 mAP）
├── configs/default.yaml
├── models/                  # yolov4.py + cbam.py
├── utils/                   # dataset / loss / metrics
├── scripts/
│   ├── convert_voc_to_yolo.py
│   └── prepare_dataset.md
├── data/
│   ├── classes.names / mask.yaml
│   ├── images/{train,val,test}/   # 大图默认 gitignore，需本地准备
│   ├── labels/...
│   └── samples/labels/            # 标注样例
└── weights/{baseline,cbam}/       # *.pt 默认不入库；保留 summary/log
```

## 环境

```bash
cd yolov4-mask-cbam
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 下载数据集时另需：pip install gdown
```

建议 Python 3.8+；有 GPU 时安装 CUDA 版 PyTorch，并将 `configs/default.yaml` 中 `device: cuda`。

## 数据准备

详见 [`scripts/prepare_dataset.md`](./scripts/prepare_dataset.md)。

1. 下载 Face Mask Detection（Kaggle 或 Pseudo-Lab Drive 镜像）：

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=16Gzn1w38yZLJpzmzKcRIRveFeHZtCEr7" -O Face_Mask_Detection.zip
unzip Face_Mask_Detection.zip -d face-mask-raw
```

原始出处：https://www.kaggle.com/datasets/andrewmvd/face-mask-detection  

2. 转为 YOLO 并划分（seed=42 → 597/170/86）：

```bash
python scripts/convert_voc_to_yolo.py \
  --images face-mask-raw/images \
  --ann face-mask-raw/annotations \
  --out data --seed 42 --train 0.7 --val 0.2 --test 0.1 --copy
```

`mask_weared_incorrect` 已并入 `without_mask`。

## 训练

```bash
# 假数据冒烟
python train.py --epochs 1 --dummy --device cpu

# 基线
python train.py --cfg configs/default.yaml --save-dir weights/baseline \
  --epochs 10 --img-size 320 --batch-size 4 --device cpu

# CBAM 改进（其余超参与基线一致）
python train.py --cfg configs/default.yaml --with-cbam --save-dir weights/cbam \
  --epochs 10 --img-size 320 --batch-size 4 --device cpu
```

输出：`last.pt` / `best.pt`（最低 val_loss）/ `train_log.jsonl` / `summary.json`。

## 已跑通实验摘要（勿改数字）

| 模型 | 参数量 | 耗时(CPU) | best val_loss | 最终 train_loss |
|------|--------|-----------|---------------|-----------------|
| Baseline | 14,879,359 | 452.7 s | 0.8532 | 0.8307 |
| +CBAM | 14,922,661 | 493.0 s | 0.8245 | 0.8091 |

完整表与日志路径：[`results_metrics.md`](./results_metrics.md)。  
简化 mAP 在本轮短训下为 0（诚实记录）；要报检测准确率请加长训练并完善评估。

## 仓库与推送

`.gitignore` 已排除：`.venv/`、`weights/**/*.pt`、`data/images/**`、`data/labels/**`、zip。  
保留代码、配置、文档、`weights/*/summary.json` 与 `train_log.jsonl`。

### 本地已 init 时创建 GitHub 仓库示例

```bash
cd yolov4-mask-cbam
git status
# 若尚未登录：gh auth login
gh repo create yolov4-mask-cbam --public --source=. --remote=origin --push
# 或私有：--private
```

Gitee：

```bash
git remote add origin git@gitee.com:<你的用户名>/yolov4-mask-cbam.git
git push -u origin main
```

## 文档索引

| 文件 | 用途 |
|------|------|
| `开题说明.md` | 题目与范围 |
| `数据方案.md` | 数据选型 |
| `报告提纲.md` | 报告结构 |
| `大作业报告.md` | **完整可提交课程报告** |
| `大作业报告-实验数据.md` | 真实实验表草稿 |
| `results_metrics.md` | 指标与命令、日志路径 |
