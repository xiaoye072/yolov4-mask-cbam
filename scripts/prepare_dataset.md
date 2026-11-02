# 数据集准备说明（andrewmvd / face-mask-detection）

## 1. 下载

- **首选（Kaggle）**：https://www.kaggle.com/datasets/andrewmvd/face-mask-detection  
  CLI：`kaggle datasets download -d andrewmvd/face-mask-detection`
- **公开镜像（本实验实际使用）**：Pseudo-Lab Tutorial Book 的 Google Drive 副本  
  - 文件 ID：`16Gzn1w38yZLJpzmzKcRIRveFeHZtCEr7`  
  - 下载示例：
    ```bash
    pip install gdown
    gdown "https://drive.google.com/uc?id=16Gzn1w38yZLJpzmzKcRIRveFeHZtCEr7" -O Face_Mask_Detection.zip
    unzip Face_Mask_Detection.zip -d face-mask-raw
    ```
  - 来源说明：https://github.com/Pseudo-Lab/Tutorial-Book-Utils （`PL_data_loader.py` → FaceMaskDetection）

解压后通常包含：

- `images/`：png/jpg（853 张）
- `annotations/`：Pascal VOC XML（853 个）

## 2. 类别映射（本课题）

| 原始类别 | class_id | 本课题名称 |
|----------|----------|------------|
| with_mask | 0 | with_mask |
| without_mask | 1 | without_mask |
| mask_weared_incorrect | 1 | **并入 without_mask** |

## 3. 一键转换 + 划分（seed=42，可复现）

```bash
python scripts/convert_voc_to_yolo.py \
  --images /path/to/face-mask-raw/images \
  --ann /path/to/face-mask-raw/annotations \
  --out data \
  --seed 42 --train 0.7 --val 0.2 --test 0.1 --copy
```

期望划分（853 张）：**train=597 / val=170 / test=86**。

生成：

```text
data/images/{train,val,test}/
data/labels/{train,val,test}/
data/train.txt  data/val.txt  data/test.txt
data/classes.names
```

## 4. 与训练的衔接

- `configs/default.yaml` 中 `train_images` / `val_images` 已指向上述目录。  
- 若尚未下载数据，可直接：

```bash
python train.py --epochs 1 --dummy
```

用假数据验证代码链路；**不代表真实训练完成**。

## 5. 标注格式复查

每行：`class_id x_center y_center width height`（相对宽高归一化）。  
示例：`0 0.512000 0.431000 0.186000 0.254000`
