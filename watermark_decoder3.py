import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import os

# ==========================
# 极坐标函数增加 min_radius
# 支持：[min_radius, max_radius] 范围采样
# ==========================
def cartesian_to_polar(input_tensor, output_shape, min_radius, max_radius):
    """
    input_tensor: (B, 1, H, W) - 频谱图
    output_shape: (R_bins, T_bins) - 极坐标分辨率 (半径维, 角度维)
    min_radius: 最小采样半径
    max_radius: 最大采样半径
    """
    B, C, H, W = input_tensor.shape
    R_bins, T_bins = output_shape

    # 数值稳定性检查
    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        print(f"[WARNING] NaN/Inf found in cartesian_to_polar input!")
        print(f"  NaN count: {torch.isnan(input_tensor).sum()}")
        print(f"  Inf count: {torch.isinf(input_tensor).sum()}")
    input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

    # 核心修改：从 min ~ max 采样，不再从 0 开始
    rho = torch.linspace(min_radius, max_radius, R_bins, device=input_tensor.device)
    theta = torch.linspace(0, np.pi, T_bins, device=input_tensor.device)

    grid_rho, grid_theta = torch.meshgrid(rho, theta, indexing='ij')

    grid_x = grid_rho * torch.cos(grid_theta) / (W / 2)
    grid_y = grid_rho * torch.sin(grid_theta) / (H / 2)

    grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    polar_map = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)

    # 再次检查输出
    if torch.isnan(polar_map).any() or torch.isinf(polar_map).any():
        print(f"[WARNING] NaN/Inf found in cartesian_to_polar output!")
        print(f"  NaN count: {torch.isnan(polar_map).sum()}")
        print(f"  Inf count: {torch.isinf(polar_map).sum()}")
    polar_map = torch.nan_to_num(polar_map, nan=0.0, posinf=1e6, neginf=-1e6)
    return polar_map

# ==========================
# ResUNet 系列
# ==========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 添加缩放层，代替 Norm 层的功能
        self.scale1 = nn.Parameter(torch.ones(out_channels))
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.scale2 = nn.Parameter(torch.ones(out_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] NaN/Inf found in ResidualBlock input!")
            print(f"  NaN count: {torch.isnan(x).sum()}")
            print(f"  Inf count: {torch.isinf(x).sum()}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        identity = self.shortcut(x)

        # 简化版：Conv + 缩放 + ReLU
        out = self.conv1(x)
        # 应用缩放和偏移
        out = out * self.scale1.view(1, -1, 1, 1) + self.bias1.view(1, -1, 1, 1)
        out = self.relu(out)

        out = self.conv2(out)
        # 应用缩放和偏移
        out = out * self.scale2.view(1, -1, 1, 1) + self.bias2.view(1, -1, 1, 1)

        # 残差连接
        out += identity
        out = self.relu(out)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out

class ResUNetFilter(nn.Module):
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
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] NaN/Inf found in ResUNetFilter input!")
            print(f"  NaN count: {torch.isnan(x).sum()}")
            print(f"  Inf count: {torch.isinf(x).sum()}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[WARNING] NaN/Inf found in ResUNetFilter output!")
            print(f"  NaN count: {torch.isnan(out).sum()}")
            print(f"  Inf count: {torch.isinf(out).sum()}")
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

# ==========================
# 工具函数
# ==========================
def save_tensor_as_img(tensor, name, batch_idx=0):
    img = tensor[batch_idx].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    os.makedirs('debug_outputs', exist_ok=True)
    vutils.save_image(img, f'debug_outputs/{name}.png')

# ==========================
# 三圈独立范围采样 + ViT 风格解码器（按半径分块）
# ==========================
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        # 数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] NaN/Inf found in PatchEmbedding input!")
            print(f"  NaN count: {torch.isnan(x).sum()}")
            print(f"  Inf count: {torch.isinf(x).sum()}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        B = x.shape[0]
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # 再次检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] NaN/Inf found in PatchEmbedding after projection!")
            print(f"  NaN count: {torch.isnan(x).sum()}")
            print(f"  Inf count: {torch.isinf(x).sum()}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return x

class AdvancedWatermarkDecoder(nn.Module):
    def __init__(self, n_sectors=60,rings=None,bits=None):
        super().__init__()
        self.n_sectors = n_sectors
        self.pre_filter = ResUNetFilter()

        # 三圈配置：min, max
        self.rings = rings if rings is not None else [(4, 6), (8, 10), (11, 13)]
        self.bits = bits if bits is not None else [5, 15, 40]
        self.angle_bins = 180
        self.radius_bins = 12

        # ViT 配置
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 2

        # 每个 ring 的 Patch Embedding：根据 bits 数量分块，每个块保留完整角度和半径
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                in_dim=(self.angle_bins // bit_num) * self.radius_bins,
                embed_dim=self.embed_dim,
                num_patches=bit_num
            ) for bit_num in self.bits
        ])

        # 每个 ring 的 Transformer Encoder
        self.ring_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    batch_first=True,
                    dim_feedforward=self.embed_dim * 4
                ),
                num_layers=self.num_layers
            ) for _ in self.rings
        ])

        # 跨 ring 融合
        self.cross_ring_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * len(self.rings), self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        self.head = nn.Linear(self.embed_dim, n_sectors)

    def forward(self, x, save_debug=False):
        # 输入数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] NaN/Inf found in AdvancedWatermarkDecoder input!")
            print(f"  NaN count: {torch.isnan(x).sum()}")
            print(f"  Inf count: {torch.isinf(x).sum()}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        res = self.pre_filter(x)
        fft_map = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2,-1)), dim=(-2,-1))
        mag = torch.abs(fft_map)

        # mag 数值稳定性检查
        if torch.isnan(mag).any() or torch.isinf(mag).any():
            print(f"[WARNING] NaN/Inf found in mag after FFT!")
            print(f"  NaN count: {torch.isnan(mag).sum()}")
            print(f"  Inf count: {torch.isinf(mag).sum()}")
        mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)

        ring_embeds = []
        for (min_r, max_r), bit_num, patch_embed, transformer in zip(self.rings, self.bits, self.patch_embeddings, self.ring_transformers):
            polar = cartesian_to_polar(
                mag,
                output_shape=(self.radius_bins, self.angle_bins),
                min_radius=min_r,
                max_radius=max_r
            )

            # [B, 1, R, T] -> [B, T, R]
            polar_reshaped = polar.squeeze(1).permute(0, 2, 1)  # [B, 180, 8]

            # 根据 bit 数量分块：保留完整角度范围，不 mean 压缩
            angles_per_bit = self.angle_bins // bit_num
            patches = []
            for i in range(bit_num):
                start = i * angles_per_bit
                end = (i + 1) * angles_per_bit
                bit_patch = polar_reshaped[:, start:end, :].flatten(1)  # [B, angles_per_bit*R]
                patches.append(bit_patch)
            patches = torch.stack(patches, dim=1)  # [B, bit_num, angles_per_bit*R]

            # Patch Embedding
            x = patch_embed(patches)  # [B, bit_num+1, embed_dim]

            # Transformer 编码
            x = transformer(x)  # [B, bit_num+1, embed_dim]

            # 使用 CLS token 作为该 ring 的表示
            ring_repr = x[:, 0, :]  # [B, embed_dim]
            ring_embeds.append(ring_repr)

        # 融合所有 ring 的表示
        fused = torch.cat(ring_embeds, dim=-1)  # [B, embed_dim * 3]
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print(f"[WARNING] NaN/Inf found in fused ring embeddings!")
            print(f"  NaN count: {torch.isnan(fused).sum()}")
            print(f"  Inf count: {torch.isinf(fused).sum()}")
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        fused = self.cross_ring_fusion(fused)  # [B, embed_dim]
        logits = self.head(fused)

        # 输出检查
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[WARNING] NaN/Inf found in logits!")
            print(f"  NaN count: {torch.isnan(logits).sum()}")
            print(f"  Inf count: {torch.isinf(logits).sum()}")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        return torch.sigmoid(logits), mag, None
