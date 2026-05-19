import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from scipy.signal import wiener

# 添加Transformer相关导入（如果需要）
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision



class PolarPositionSampler(nn.Module):
    """
    带圆形区域限制的直角坐标采样模块
    逻辑：在 (R, theta) 中心周围采集 roi_size * roi_size 的区域，
    但仅保留半径为 local_r 内的像素，圆外四个角置为 0。
    """
    def __init__(self, img_size=512, num_bits=64, roi_size=15, local_r=7.0, R1=None):
        super().__init__()
        self.img_size = img_size
        self.num_bits = num_bits
        self.roi_size = roi_size   # 最终输出张量的边长 (如 15)
        self.local_r = local_r     # 有效信号的圆形半径 (如 7.0)
        
        # 1. 设置频谱中心
        center = (img_size // 2) + 1
        
        if R1 is None: R1 = [25.0, 40.0, 55.0, 70.0]
        R1 = torch.tensor(R1, dtype=torch.float32)

        # 2. 生成矩形网格相对偏移
        # 确保中心点在 0, 0
        half_size = (roi_size - 1) / 2
        offsets = torch.linspace(-half_size, half_size, roi_size)
        # grid_dy, grid_dx 形状均为 [roi_size, roi_size]
        grid_dy, grid_dx = torch.meshgrid(offsets, offsets, indexing='ij')

        # 3. 预计算圆形掩码 (Mask)
        # 计算每个点到网格中心的欧几里得距离
        distances = torch.sqrt(grid_dx**2 + grid_dy**2)
        # 距离小于等于 local_r 的设为 1，其余为 0
        mask = (distances <= local_r).float()
        # 注册为 buffer，跟随模型移动到 GPU，且不作为参数更新
        self.register_buffer('circular_mask', mask.view(1, 1, roi_size, roi_size))

        # 4. 预计算所有 bit 的全局采样坐标
        coords = []
        for j in range(num_bits):
            theta_base = j / num_bits * np.pi
            r_base = R1[j % len(R1)].item()
            
            # 极坐标转直角坐标，确定 ROI 中心
            xc = center + r_base * np.cos(theta_base)
            yc = center + r_base * np.sin(theta_base)
            
            # 生成 15x15 的绝对像素坐标
            x_roi = xc + grid_dx
            y_roi = yc + grid_dy
            
            # 归一化到 [-1, 1] 供 grid_sample 使用
            x_norm = (x_roi / (img_size - 1)) * 2 - 1
            y_norm = (y_roi / (img_size - 1)) * 2 - 1
            
            coords.append(torch.stack([x_norm, y_norm], dim=-1))

        self.register_buffer('sample_grid', torch.stack(coords))

    def forward(self, dft_mag):
        B = dft_mag.size(0)
        N = self.num_bits
        
        # 准备网格 [B*N, roi_size, roi_size, 2]
        grid = self.sample_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid = grid.reshape(B * N, self.roi_size, self.roi_size, 2)
        
        # 扩展 DFT 幅度谱
        dft_expanded = dft_mag.repeat_interleave(N, dim=0) 
        
        # 采样 [B*N, 1, roi_size, roi_size]
        rois = F.grid_sample(dft_expanded, grid, mode='bilinear', 
                            padding_mode='zeros', align_corners=True)
        
        # 5. 应用圆形掩码：强制将圆外区域设为 0
        rois = rois * self.circular_mask
        
        # 返回 [B, 64, 1, roi_size, roi_size]
        return rois.view(B, N, 1, self.roi_size, self.roi_size)
class PolarPositionSampler2(nn.Module):
    """
    位置引导的局部采样模块
    支持可选的角度搜索范围和精确的中心偏移控制
    """
    def __init__(self, img_size=512, num_bits=64, roi_size=30, 
                 R1=None, search_r_range=5, search_angle_range=0):
        super().__init__()
        self.img_size = img_size
        self.num_bits = num_bits
        self.roi_size = roi_size
        
        # 1. 设置中心点 (依据需求 +1)
        # 在 512x512 图像中，传统的中心通常是 256，这里设为 257
        center = (img_size // 2) + 1
        self.register_buffer('center_val', torch.tensor([center], dtype=torch.float32))

        if R1 is None: R1 = [25.0, 40.0, 55.0, 70.0]
        R1 = torch.tensor(R1, dtype=torch.float32)

        # 2. 构建采样坐标
        coords = []
        angles = np.linspace(0, np.pi, self.num_bits + 1)
        for j in range(num_bits):
            theta_start = angles[j]
            theta_end = angles[j+1]

            r_base = R1[j % len(R1)].item()
            
            # 半径搜索范围
            dr = torch.linspace(-search_r_range, search_r_range, roi_size)
            
            dt=torch.linspace(theta_start, theta_end, roi_size)
            
            # 生成局部极坐标网格
            # indexing='ij' 表示 (row_idx, col_idx) -> (dt, dr)
            grid_dt, grid_dr = torch.meshgrid(dt, dr, indexing='ij')
            
            curr_r = r_base + grid_dr
            curr_theta = theta_start + grid_dt
            
            # 极坐标转直角坐标 (此处加入 center + 1 的偏移)
            # x 对应宽度方向 (Column)，y 对应高度方向 (Row)
            x = center + curr_r * torch.cos(curr_theta)
            y = center + curr_r * torch.sin(curr_theta)
            
            # 🔥 归一化到 [-1, 1] 供 grid_sample 使用
            # 注意：grid_sample 的坐标顺序是 (x, y) 即 (col, row)
            x_norm = (x / (img_size - 1)) * 2 - 1
            y_norm = (y / (img_size - 1)) * 2 - 1
            
            # 最终形状 [size_theta, size_r, 2]
            # 如果 search_angle_range 为 0，size_theta 为 1
            coords.append(torch.stack([y_norm, x_norm], dim=-1))

        # 将所有 bit 的采样点拼在一起 [num_bits, H_roi, W_roi, 2]
        self.register_buffer('sample_grid', torch.stack(coords))

    def forward(self, dft_mag):
        B = dft_mag.size(0)
        N = self.num_bits
        
        # 准备采样网格 [B*N, H_roi, W_roi, 2]
        grid = self.sample_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid = grid.reshape(B * N, *self.sample_grid.shape[1:])
        
        # 准备待采样的频谱图 [B*N, 1, 512, 512]
        # 注意：dft_mag 应该是经过 log 变换和归一化的
        dft_expanded = dft_mag.repeat_interleave(N, dim=0) 
        
        # 执行采样
        rois = F.grid_sample(dft_expanded, grid, mode='bilinear', 
                            padding_mode='zeros', align_corners=True)
        
        # 返回形状 [B, num_bits, 1, H_roi, W_roi]
        return rois.view(B, N, 1, *rois.shape[2:])

import torch
import torch.nn as nn
import torch.nn.functional as F

def cartesian_to_polar(input_tensor, output_shape, max_radius):
    """
    input_tensor: (B, 1, H, W) - 频谱图
    output_shape: (R_bins, T_bins) - 极坐标图分辨率 (半径维, 角度维)
    max_radius: 最大采样半径 (对应你的 r=12 附近)
    """
    B, C, H, W = input_tensor.shape
    R_bins, T_bins = output_shape
    
    # 1. 生成极坐标网格
    # rho: [0, max_radius], theta: [0, pi] (因为你的水印是 180 度共轭对称)
    rho = torch.linspace(0, max_radius, R_bins, device=input_tensor.device)
    theta = torch.linspace(0, np.pi, T_bins, device=input_tensor.device)
    
    grid_rho, grid_theta = torch.meshgrid(rho, theta, indexing='ij')
    
    # 2. 转换回笛卡尔坐标用于采样 (归一化到 [-1, 1])
    # 注意：FFT 移频后，中心点在 (H/2, W/2)
    grid_x = grid_rho * torch.cos(grid_theta) / (W / 2)
    grid_y = grid_rho * torch.sin(grid_theta) / (H / 2)
    
    grid = torch.stack([grid_y,grid_x], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # 3. 双线性插值采样
    polar_map = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)
    return polar_map
def scipy_wiener_filter(x, kernel_size=5):
    """
    使用 scipy.signal.wiener 对输入进行 Wiener 滤波。
    支持 [B, C, H, W] 输入，并保持 device/dtype。
    """
    is_2d = x.dim() == 2
    is_3d = x.dim() == 3
    is_4d = x.dim() == 4

    if not (is_2d or is_3d or is_4d):
        raise ValueError(f"Unsupported tensor dim {x.dim()}")

    orig_device = x.device
    orig_dtype = x.dtype
    x_cpu = x.detach().cpu().numpy()

    if is_4d:
        out = np.empty_like(x_cpu)
        for b in range(x_cpu.shape[0]):
            for c in range(x_cpu.shape[1]):
                out[b, c] = wiener(x_cpu[b, c], mysize=(kernel_size, kernel_size))
    else:
        out = wiener(x_cpu, mysize=(kernel_size, kernel_size))

    out = out.astype(np.float32)
    return torch.from_numpy(out).to(orig_device).to(orig_dtype)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels) # 依然建议使用 InstanceNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 如果输入输出维度不一致，增加一个 shortcut 映射
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
        
        # Encoder
        self.enc1 = ResidualBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(32, 64)
        
        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(32 + 16, 16)
        
        # Output layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 1, H, W)
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
class AdvancedWatermarkDecoder(nn.Module):
    def __init__(self, n_sectors=16):
        super().__init__()
        
        # A. 空间域预处理 (类似于学习一个自适应 Wiener 滤波)
        # self.pre_filter = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, padding=1),
        #     nn.InstanceNorm2d(32), # 替代 BN，对单张图做标准化
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 3, padding=1)
        # )
        self.pre_filter = ResUNetFilter()
        
        # B. 极坐标后的特征提取
        # 输入形状: (B, 1, R_bins, T_bins) -> 例如 (B, 1, 32, 128)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 2), padding=2),
            nn.GroupNorm(8, 64), # GroupNorm 在小 batch 或频域数据上更稳
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, n_sectors)) # 强制压缩到 num_bits个位置 b,128,1,n_sectors
        )
        
        # C. 序列解码 (处理 num_bits个扇区之间的关系)
        # 将 CNN 输出视为长度为 num_bits 的序列
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 每个扇区输出一个概率
            # nn.Sigmoid() # 最后输出 0-1 之间的概率
        )

    def forward(self, x, save_debug=False):
        # 1. 空间域残差提取
        res = self.pre_filter(x)
        # img_estimated = scipy_wiener_filter(x, kernel_size=5)
        # res = x - img_estimated
        if save_debug:
            save_tensor_as_img(res, "1_residual")
        
        # 2. 频域变换
        fft_map = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2, -1)), dim=(-2, -1))
        # mag = torch.log(torch.abs(fft_map) + 1e-6) 
        mag=torch.abs(fft_map)
        if save_debug:
            save_tensor_as_img(mag, "2_magnitude_spectrum")
        
        # 3. 极坐标拉直
        # 注意：max_radius=100 对于你 r=[5,12] 的设计来说太大了，
        # 你的信号会缩在图的最左边（很窄的一条），建议根据之前的建议设为 32 左右
        polar_feat = cartesian_to_polar(mag, output_shape=(24, 360), max_radius=25)
        if save_debug:
            # 保存前可以旋转一下方向，方便观察 16 个条带
            save_tensor_as_img(polar_feat, "3_polar_feature")
        
        # 4. 后续网络逻辑...
        feat = self.feature_extractor(polar_feat)
        feat = feat.squeeze(2).permute(0, 2, 1)
        
        attn_out, _ = self.attention(feat, feat, feat)
        logits = self.classifier(attn_out)
        
        return torch.sigmoid(logits.squeeze(-1)),mag,polar_feat
import torch
import torchvision.utils as vutils
import os

def save_tensor_as_img(tensor, name, batch_idx=0):
    """
    将 Tensor 归一化并保存为图片
    """
    # 1. 确保在 CPU 上并取第一个 batch
    img = tensor[batch_idx,batch_idx].detach().cpu()
    
    # 2. 线性归一化到 [0, 1] 之间，这样图片才看得见
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)
    
    # 3. 创建文件夹并保存
    os.makedirs('debug_outputs', exist_ok=True)
    vutils.save_image(img, f'debug_outputs/{name}.png')
class WatermarkDecoder(nn.Module):
    def __init__(self, block_size=512, num_bits=64, r=[25.0, 40.0, 55.0, 70.0], roi_size=15, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_bits = num_bits
        
        # 1. 残差提取（保持不变，但可加InstanceNorm）
        self.residual_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.InstanceNorm2d(32),  # 替代BatchNorm
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        
        # 2. 全局特征提取器（增强）
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(16*16, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim // 2)  # 输出嵌入维度
        )
        
        # 3. 采样器（保持PolarPositionSampler2）
        self.sampler = PolarPositionSampler2(img_size=block_size, num_bits=num_bits, roi_size=roi_size, R1=r, search_r_range=2, search_angle_range=0.02)
        
        # 4. 局部ROI特征提取器（更深，更复杂）
        self.roi_net = nn.Sequential(
            # 残差块1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 残差连接（简单版）
            nn.Conv2d(32, 32, 1),  # 跳跃连接
            
            # 残差块2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 压缩到1x1
            nn.Flatten(),
            nn.Linear(64, embed_dim // 2)  # 局部特征维度
        )
        
        # 5. Transformer编码器（处理64位序列）
        self.pos_embedding = nn.Parameter(torch.randn(1, num_bits, embed_dim))  # 位置嵌入
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6. 最终分类器（每位独立分类）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B = x.size(0)
        
        # A. 残差提取和FFT
        img_estimated = scipy_wiener_filter(x, kernel_size=5)
        res = x - img_estimated
       
        imge = (img_estimated[0, 0] - img_estimated[0, 0].min()) / (img_estimated[0, 0].max() - img_estimated[0, 0].min() + 1e-8)
        cv2.imwrite("tmp/debug_img_estimated.png", (imge * 255).detach().cpu().numpy().astype(np.uint8))
        imgr= (res[0, 0] - res[0, 0].min()) / (res[0, 0].max() - res[0, 0].min() + 1e-8)
        cv2.imwrite("tmp/debug_res.png", (imgr * 255).detach().cpu().numpy().astype(np.uint8))
        
        r_fft = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2, -1)), dim=(-2, -1))
        dft_mag = torch.abs(r_fft)
        save_spectrum(dft_mag[0, 0], filename="debug_dft_mag.png")
        r_fft1 = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        dft_mag1= torch.abs(r_fft1)
        dft_mag1 = torch.log1p(dft_mag1)
        save_spectrum(dft_mag1[0, 0], filename="debug_dft_mag1.png")
        # B. 全局特征
        dft_norm = (dft_mag - dft_mag.mean(dim=(-2, -1), keepdim=True)) / (dft_mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        g_feat = self.global_context(dft_norm)  # [B, embed_dim]
        
        # C. 采样ROI
        rois = self.sampler(dft_mag)  # [B, num_bits, 1, roi_size, roi_size]
        save_rois_visualization(rois, batch_idx=0, filename="debug_rois.png")
        # D. 并行提取局部特征
        B_N, C, H, W = rois.shape[0] * rois.shape[1], *rois.shape[2:]  # [B*num_bits, 1, H, W]
        rois_flat = rois.view(B_N, C, H, W)
        roi_feat_flat = self.roi_net(rois_flat)  # [B*num_bits, embed_dim//2]
        roi_feat = roi_feat_flat.view(B, self.num_bits, -1)  # [B, num_bits, embed_dim//2]
        
        # E. 融合全局和局部特征
        combined_feat = torch.cat([roi_feat, g_feat.unsqueeze(1).expand(-1, self.num_bits, -1)], dim=-1)  # [B, num_bits, embed_dim]
        
        # F. 添加位置嵌入并通过Transformer
        seq_feat = combined_feat + self.pos_embedding  # [B, num_bits, embed_dim]
        attn_out = self.transformer(seq_feat)  # [B, num_bits, embed_dim]
        
        # G. 最终分类（每位独立）
        logits = self.classifier(attn_out)  # [B, num_bits, 1]
        return logits.squeeze(-1)  # [B, num_bits]

def save_rois_visualization( rois, batch_idx=0, filename="debug_rois.png"):
    """
    rois: [B, num_bits, 1, H, W] 的 Tensor
    """
    # 1. 提取第 batch_idx 个样本
    # shape: [64, 1, H, W]
    sample_rois = rois[batch_idx].detach().cpu()
    numbit=sample_rois.shape[0]
    
    # 2. 归一化到 0-1 之间以便显示
    # 这里的归一化很重要，因为 DFT 的值可能很小
    mi = sample_rois.min()
    ma = sample_rois.max()
    sample_rois = (sample_rois - mi) / (ma - mi + 1e-8)
    
    # 3. 使用 torchvision 将 64 个小图拼成 8x8 的网格
    # padding=1 会在小图之间加白边，方便区分
    grid = torchvision.utils.make_grid(sample_rois, nrow=int(np.sqrt(numbit)), padding=1)
    
    # 4. 转换为 HWC 格式并保存
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    cv2.imwrite(f"tmp/{filename}", grid_np)
    print(f"Saved ROI visualization to tmp/{filename}")


def save_spectrum(dft_mag, filename="dft_debug.png"):
    """
    将 DFT 幅度谱 Tensor 保存为可视化图片
    dft_mag shape: [B, 1, H, W] 或 [H, W]
    """
    # 1. 降维处理：只取 Batch 中的第一张图
    if dft_mag.dim() == 4:
        img = dft_mag[0, 0]
    elif dft_mag.dim() == 3:
        img = dft_mag[0]
    else:
        img = dft_mag
        
    # 搬到 CPU 并转为 NumPy
    img = img.detach().cpu().numpy()

    # 2. Log 变换 (这是能看到细节的核心)
    # 频谱值跨度极大，log1p 可以将数据压缩到肉眼可辨的范围
    # img_log = np.log1p(img)
    img_log = img

    # 3. 归一化到 0-255
    img_min = img_log.min()
    img_max = img_log.max()
    img_norm = (img_log - img_min) / (img_max - img_min + 1e-8)
    img_255 = (img_norm * 255).astype(np.uint8)

    # 4. 增强对比度 (可选)
    # 如果还是太暗，可以使用直方图均衡化
    # img_255 = cv2.equalizeHist(img_255)

    cv2.imwrite(f"tmp/{filename}", img_255)
    print(f"Spectrum saved to tmp/{filename}")