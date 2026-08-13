"""
v17 解码器 - 无旋转环，所有环都嵌入bit序列

与v15/v16的区别：
- 去掉旋转环的概念
- 所有环都嵌入bit序列
- 使用 Transformer 隐式学习旋转不变性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def cartesian_to_polar(input_tensor, output_shape, min_radius, max_radius):
    """
    笛卡尔坐标转极坐标（与v15一致）
    input_tensor: (B, 1, H, W) - 频谱图
    output_shape: (R_bins, T_bins) - 极坐标分辨率
    """
    B, C, H, W = input_tensor.shape
    R_bins, T_bins = output_shape

    input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

    # 可微分的线性组合
    t = torch.linspace(0, 1, R_bins, device=input_tensor.device)
    rho = min_radius + t * (max_radius - min_radius)

    theta = torch.linspace(0, np.pi, T_bins, device=input_tensor.device)

    grid_rho, grid_theta = torch.meshgrid(rho, theta, indexing='ij')
    grid_x = grid_rho * torch.cos(grid_theta) / (W / 2)
    grid_y = grid_rho * torch.sin(grid_theta) / (H / 2)

    grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    polar_map = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)

    polar_map = torch.nan_to_num(polar_map, nan=0.0, posinf=1e6, neginf=-1e6)
    return polar_map


class ResidualBlock(nn.Module):
    """残差块（与v15一致）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.scale1 = nn.Parameter(torch.ones(out_channels))
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.scale2 = nn.Parameter(torch.ones(out_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = out * self.scale1.view(1, -1, 1, 1) + self.bias1.view(1, -1, 1, 1)
        out = self.relu(out)

        out = self.conv2(out)
        out = out * self.scale2.view(1, -1, 1, 1) + self.bias2.view(1, -1, 1, 1)

        out += identity
        out = self.relu(out)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out


class ResUNetFilter(nn.Module):
    """残差UNet滤波器（与v15一致）"""
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(32, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(32 + 16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        # 处理尺寸不匹配
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


class PatchEmbedding(nn.Module):
    """Patch嵌入"""
    def __init__(self, in_dim, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x


class WatermarkDecoderV17(nn.Module):
    """
    v17 解码器 - 无旋转环

    解码流程：
    1. 输入单通道图像
    2. FFT -> 幅度谱
    3. 对所有环分别极坐标采样
    4. 送入 Transformer
    5. 输出 n_sectors bits

    环结构：
    - 所有环都嵌入bit序列（无旋转环）
    - 使用 Transformer 隐式学习旋转不变性
    """
    def __init__(self, n_sectors=60, bits=None,
                 angle_bins=200, radius_bins=12,
                 ring_positions_init=None):
        """
        Args:
            n_sectors: 水印总位数
            bits: 每个环的位数 [20, 20, 20]（全部是水印环）
            angle_bins: 角度分辨率
            radius_bins: 半径分辨率
            ring_positions_init: 环位置初始值 [8, 13, 18]
        """
        super().__init__()
        self.n_sectors = n_sectors
        self.pre_filter = ResUNetFilter()

        # 环配置（全部是水印环）
        self.bits = bits if bits is not None else [20, 20, 20]

        # 环位置
        if ring_positions_init is None:
            ring_positions_init = [8.0, 13.0, 18.0]
        self.register_buffer('ring_positions', torch.tensor(ring_positions_init, dtype=torch.float32))
        self.ring_width = 2

        self.angle_bins = angle_bins
        self.radius_bins = radius_bins

        # ViT 配置
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 2

        # 每个 ring 的 Patch Embedding
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                in_dim=(self.angle_bins // bit_num) * self.radius_bins,
                embed_dim=self.embed_dim,
                num_patches=bit_num
            ) for bit_num in self.bits
        ])

        # 每个 ring 的 Transformer Encoder
        num_rings = len(self.bits)
        self.ring_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    batch_first=True,
                    dim_feedforward=self.embed_dim * 4
                ),
                num_layers=self.num_layers
            ) for _ in range(num_rings)
        ])

        # 跨 ring 融合
        self.cross_ring_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * num_rings, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        # 水印头
        self.head = nn.Linear(self.embed_dim, n_sectors)

    def forward(self, x, return_rotation=False):
        """
        前向传播

        Args:
            x: (B, 1, H, W) - 输入图像
            return_rotation: 是否返回旋转角度（始终返回None）

        Returns:
            logits: (B, n_sectors)
            mag: FFT幅度谱
            rotation_angle: 始终返回None
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # 预滤波 + FFT
        res = self.pre_filter(x)
        fft_map = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2, -1)), dim=(-2, -1))
        mag = torch.abs(fft_map)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)

        # 获取环位置
        ring_pos = self.ring_positions

        # 计算采样范围
        # 嵌入范围: [r, r+r_range] (r_range=1, 由编码器决定)
        # 采样范围: Ring0=[r0-2, r1-3], Ring1=[r0+ring_width+2, r1+3]
        num_rings = len(ring_pos)
        dynamic_rings = []
        for i in range(num_rings):
            r = ring_pos[i]
            if i == 0:
                # 第一个环：采样范围 [r-2, ring1-3]
                min_r = r - 2
                max_r = ring_pos[i+1] - 3
            elif i == num_rings - 1:
                # 最后一个环：采样范围 [ring0+ring_width+2, r+3]
                min_r = ring_pos[i-1] + self.ring_width + 2
                max_r = r + 3
            else:
                # 中间环：采样范围 [前一个环嵌入结束+2, 下一个环嵌入开始-3]
                min_r = ring_pos[i-1] + self.ring_width + 2
                max_r = ring_pos[i+1] - 3
            dynamic_rings.append((min_r, max_r))

        # 解码所有环
        ring_embeds = []
        for i, (bit_num, patch_embed, transformer) in enumerate(zip(
                self.bits, self.patch_embeddings, self.ring_transformers)):

            min_r, max_r = dynamic_rings[i]
            polar = cartesian_to_polar(
                mag,
                output_shape=(self.radius_bins, self.angle_bins),
                min_radius=min_r,
                max_radius=max_r
            )

            polar_reshaped = polar.squeeze(1).permute(0, 2, 1)

            angles_per_patch = self.angle_bins // bit_num
            patches = []
            for j in range(bit_num):
                start = j * angles_per_patch
                end = (j + 1) * angles_per_patch
                patch = polar_reshaped[:, start:end, :].flatten(1)
                patches.append(patch)
            patches = torch.stack(patches, dim=1)

            x_embed = patch_embed(patches)
            x_transformer = transformer(x_embed)
            ring_repr = x_transformer[:, 0, :]  # CLS token
            ring_embeds.append(ring_repr)

        # 融合
        fused = torch.cat(ring_embeds, dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        fused = self.cross_ring_fusion(fused)
        logits = self.head(fused)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        return torch.sigmoid(logits), mag, None


if __name__ == "__main__":
    # 测试模型
    model = WatermarkDecoderV17(
        n_sectors=60,
        bits=[20, 20, 20],
        angle_bins=200,
        radius_bins=12,
        ring_positions_init=[8.0, 13.0, 18.0],
    )

    # 测试输入
    x = torch.randn(2, 1, 512, 512)
    logits, mag, _ = model(x)

    print(f"Logits shape: {logits.shape}")
    print(f"Mag shape: {mag.shape}")
    print(f"Ring positions: {model.ring_positions.tolist()}")
    print("Model test passed!")
