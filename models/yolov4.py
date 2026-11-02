"""
教学简化版 YOLOv4-like（口罩二分类检测）

与教材 / 正式 YOLOv4 对照点（README 亦有说明）：
  - 骨干：CSP 风格残差块的缩减版（通道与层数更少），接口仍称 CSPDarknet-like
  - 颈部：简化 SPP + 自上而下 FPN 式融合（PANet 双向可后续替换）
  - 检测头：三尺度 YOLO head，输出 (B, na*(5+nc), H, W)
  - CBAM：可选，插在骨干深层 CSP 输出后（见下方「【CBAM 插入点】」注释）

正式接口：
  build_model(num_classes, with_cbam, img_size) -> nn.Module
  YOLOv4.load_backbone_weights(path)  # 预留，便于日后换完整预训练权重
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .cbam import CBAM


def conv_bn_act(
    in_ch: int,
    out_ch: int,
    k: int = 3,
    s: int = 1,
    p: Optional[int] = None,
    act: bool = True,
) -> nn.Sequential:
    if p is None:
        p = k // 2
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """基础残差块：1x1 降维 + 3x3 恢复。"""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 2
        self.block = nn.Sequential(
            conv_bn_act(channels, hidden, k=1),
            conv_bn_act(hidden, channels, k=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CSPBlock(nn.Module):
    """
    简化 CSP：特征分两路，一路经若干 Residual，再拼接融合。
    对应教材 CSPDarknet 中跨阶段部分连接思想（通道数已缩减）。
    """

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 1, downsample: bool = True):
        super().__init__()
        self.down = conv_bn_act(in_ch, out_ch, k=3, s=2) if downsample else conv_bn_act(in_ch, out_ch, k=3, s=1)
        mid = out_ch // 2
        self.split_conv = conv_bn_act(out_ch, mid, k=1)
        self.trans_conv = conv_bn_act(out_ch, mid, k=1)
        self.blocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(n_blocks)])
        self.merge = conv_bn_act(mid * 2, out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x1 = self.split_conv(x)
        x2 = self.trans_conv(x)
        x2 = self.blocks(x2)
        return self.merge(torch.cat([x1, x2], dim=1))


class SPP(nn.Module):
    """简化 Spatial Pyramid Pooling：多尺度 max-pool 后拼接再 1x1。"""

    def __init__(self, channels: int, pools: Tuple[int, ...] = (5, 9, 13)):
        super().__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in pools]
        )
        self.conv = conv_bn_act(channels * (len(pools) + 1), channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x] + [p(x) for p in self.pools]
        return self.conv(torch.cat(feats, dim=1))


class YOLOHead(nn.Module):
    """单尺度检测头：若干卷积 + 预测层。"""

    def __init__(self, in_ch: int, mid_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, mid_ch, k=3),
            conv_bn_act(mid_ch, mid_ch // 2, k=1),
            conv_bn_act(mid_ch // 2, mid_ch, k=3),
        )
        self.pred = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.stem(x))


class YOLOv4(nn.Module):
    """
    教学简化 YOLOv4-like。

    输出：List[Tensor]，三尺度，每个形状 (B, na*(5+nc), H, W)
      stride 约 8 / 16 / 32（相对输入）
    """

    def __init__(
        self,
        num_classes: int = 2,
        with_cbam: bool = False,
        num_anchors: int = 3,
        width_mult: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.with_cbam = with_cbam
        self.num_anchors = num_anchors

        def c(ch: int) -> int:
            return max(int(ch * width_mult), 16)

        # -------- Backbone: CSPDarknet-like --------
        self.stem = nn.Sequential(
            conv_bn_act(3, c(32), k=3, s=1),
            conv_bn_act(c(32), c(64), k=3, s=2),
        )
        self.csp1 = CSPBlock(c(64), c(64), n_blocks=1, downsample=False)   # ~ /2
        self.csp2 = CSPBlock(c(64), c(128), n_blocks=2, downsample=True)   # ~ /4
        self.csp3 = CSPBlock(c(128), c(256), n_blocks=4, downsample=True)  # ~ /8  -> P3
        self.csp4 = CSPBlock(c(256), c(512), n_blocks=4, downsample=True)  # ~ /16 -> P4
        self.csp5 = CSPBlock(c(512), c(1024), n_blocks=2, downsample=True) # ~ /32 -> P5

        # 【CBAM 插入点】教材 7.2.1：在骨干深层 CSP 输出后嵌入注意力
        # 此处对 P3 / P4 / P5 三个尺度特征分别可选加 CBAM（与基线仅差这些模块）
        ch3, ch4, ch5 = c(256), c(512), c(1024)
        if with_cbam:
            self.cbam_p3 = CBAM(ch3, reduction=16)
            self.cbam_p4 = CBAM(ch4, reduction=16)
            self.cbam_p5 = CBAM(ch5, reduction=16)
        else:
            self.cbam_p3 = nn.Identity()
            self.cbam_p4 = nn.Identity()
            self.cbam_p5 = nn.Identity()

        # -------- Neck: SPP + 简化 FPN --------
        self.spp = SPP(ch5)
        self.reduce5 = conv_bn_act(ch5, ch4, k=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse4 = conv_bn_act(ch4 + ch4, ch4, k=3)
        self.reduce4 = conv_bn_act(ch4, ch3, k=1)
        self.fuse3 = conv_bn_act(ch3 + ch3, ch3, k=3)

        # 自底向上（简化 PANet 的一层）
        self.down3 = conv_bn_act(ch3, ch3, k=3, s=2)
        self.pan4 = conv_bn_act(ch3 + ch4, ch4, k=3)
        self.down4 = conv_bn_act(ch4, ch4, k=3, s=2)
        self.pan5 = conv_bn_act(ch4 + ch4, ch5, k=3)

        # -------- Heads --------
        self.head3 = YOLOHead(ch3, ch3, num_anchors, num_classes)  # 大尺度 / 小目标
        self.head4 = YOLOHead(ch4, ch4, num_anchors, num_classes)
        self.head5 = YOLOHead(ch5, ch5, num_anchors, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.csp1(x)
        x = self.csp2(x)
        p3 = self.csp3(x)
        p4 = self.csp4(p3)
        p5 = self.csp5(p4)

        # 【CBAM 插入点】骨干特征增强
        p3 = self.cbam_p3(p3)
        p4 = self.cbam_p4(p4)
        p5 = self.cbam_p5(p5)

        p5 = self.spp(p5)
        p5_r = self.reduce5(p5)
        p4_f = self.fuse4(torch.cat([self.up(p5_r), p4], dim=1))
        p4_r = self.reduce4(p4_f)
        p3_f = self.fuse3(torch.cat([self.up(p4_r), p3], dim=1))

        p4_p = self.pan4(torch.cat([self.down3(p3_f), p4_f], dim=1))
        p5_p = self.pan5(torch.cat([self.down4(p4_p), p5_r], dim=1))

        return [self.head3(p3_f), self.head4(p4_p), self.head5(p5_p)]

    def load_backbone_weights(self, path: str, strict: bool = False) -> None:
        """
        正式接口预留：加载完整 / 官方骨干权重。
        教学简化版通道数与完整 CSPDarknet53 不一致时会 partial load；
        换完整实现后可在此对接 state_dict 映射。
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # 去掉常见前缀
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            cleaned[nk] = v
        missing, unexpected = self.load_state_dict(cleaned, strict=strict)
        print(f"[load_backbone_weights] missing={len(missing)} unexpected={len(unexpected)}")


def build_model(
    num_classes: int = 2,
    with_cbam: bool = False,
    img_size: int = 416,
    width_mult: float = 0.5,
) -> YOLOv4:
    """统一构建入口，供 train.py / 评估脚本调用。"""
    _ = img_size  # 输入尺寸由数据管线保证；模型为全卷积
    return YOLOv4(
        num_classes=num_classes,
        with_cbam=with_cbam,
        width_mult=width_mult,
    )


__all__ = ["YOLOv4", "build_model"]
