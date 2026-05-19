import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import os

# ==========================
# 【你要的修改】极坐标函数增加 min_radius
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
    
    # 核心修改：从 min ~ max 采样，不再从 0 开始
    rho = torch.linspace(min_radius, max_radius, R_bins, device=input_tensor.device)
    theta = torch.linspace(0, np.pi, T_bins, device=input_tensor.device)
    
    grid_rho, grid_theta = torch.meshgrid(rho, theta, indexing='ij')
    
    grid_x = grid_rho * torch.cos(grid_theta) / (W / 2)
    grid_y = grid_rho * torch.sin(grid_theta) / (H / 2)
    
    grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    polar_map = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)
    return polar_map

# ==========================
# 【完全不动】ResUNet 系列
# ==========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# ==========================
# 工具函数
# ==========================
def save_tensor_as_img(tensor, name, batch_idx=0):
    img = tensor[batch_idx].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    os.makedirs('debug_outputs', exist_ok=True)
    vutils.save_image(img, f'debug_outputs/{name}.png')

# ==========================
# 三圈独立范围采样 + ViT 解码器
# ==========================
class AdvancedWatermarkDecoder(nn.Module):
    def __init__(self, n_sectors=60,rings=None,bits=None):
        super().__init__()
        self.n_sectors = n_sectors
        self.pre_filter = ResUNetFilter()

        # 三圈配置：min, max  （你可以自由调整）
        # self.rings = [
        #     (4, 6),   # 圈1：5±1
        #     (8, 10),  # 圈2：9±1
        #     (11, 13)  # 圈3：12±1
        # ]
        self.rings = rings if rings is not None else [(4, 6), (8, 10), (11, 13)]
        self.bits = bits if bits is not None else [5, 15, 40]
        self.angle_bins = 180

        # ViT
        self.embed_dim = 256
        self.ring_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.angle_bins, self.embed_dim),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True),
                    num_layers=2
                )
            ) for _ in self.rings
        ])

        self.fusion = nn.Sequential(
            nn.Linear(self.embed_dim*3, self.embed_dim*2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.head = nn.Linear(self.embed_dim*2, n_sectors)

    def forward(self, x, save_debug=False):
        # 完全不变
        res = self.pre_filter(x)
        fft_map = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2,-1)), dim=(-2,-1))
        mag = torch.abs(fft_map)

        # ==========================
        # 三圈独立范围采样
        # ==========================
        ring_feats = []
        for (min_r, max_r), bit_num in zip(self.rings, self.bits):
            polar = cartesian_to_polar(
                mag,
                output_shape=(8, self.angle_bins),  # 半径维度分辨率
                min_radius=min_r,
                max_radius=max_r
            )
            # 压缩半径维度，保留角度序列
            feat = polar.squeeze(1).mean(dim=1)  # [B, 180]
            ring_feats.append(feat)

        # ViT 编码
        embeds = []
        for tf, feat in zip(self.ring_transformers, ring_feats):
            e = tf(feat.unsqueeze(1))
            embeds.append(e.squeeze(1))

        # 融合输出
        fused = torch.cat(embeds, dim=-1)
        fused = self.fusion(fused)
        logits = self.head(fused)

        return torch.sigmoid(logits), mag, None