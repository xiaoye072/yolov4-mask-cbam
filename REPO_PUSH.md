# 仓库地址

公开仓库（默认分支 `main`）：

https://github.com/xiaoye072/yolov4-mask-cbam

克隆：

```bash
git clone https://github.com/xiaoye072/yolov4-mask-cbam.git
cd yolov4-mask-cbam
```

运行步骤见 [`README.md`](./README.md)。

## 不会进仓库的内容

`.gitignore` 排除：

- 权重 `*.pt`（以及 `.pth` / `.onnx` / `.weights` / `.ckpt`）
- 虚拟环境 `.venv/`
- 完整图片 `data/images/`（目录用 `.gitkeep` 占位）
- 压缩包、常见图片/视频等大二进制、解压后的 `face-mask-raw/`

会保留：代码、配置、脚本、文档、`data/labels/`、`data/samples/`、`weights/*/summary.json`、`train_log.jsonl` 和小型 console 日志。

图片请按 README / `scripts/prepare_dataset.md` 重新下载，再用 `scripts/convert_voc_to_yolo.py` 转成 YOLO。权重在本地 `weights/` 训练生成，不要把编造的 mAP 写进文档；已记录实验的简化 mAP@0.5 为 0。
