"""
v18 镂空水印训练 - 纯pair噪声版

支持每N轮保存一次权重，纯pair噪声训练

用法:
    python train_cb_v18_pair.py --config config/train_cb_v18_hollow_pair.yaml
"""

import os
import sys
import argparse
import logging
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import yaml
import json

from watermark_decoder_v17 import WatermarkDecoderV17
from dataset_v18 import WatermarkDatasetV18 as WatermarkDatasetV11

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_cb_v18_pair.log')
    ]
)
logger = logging.getLogger(__name__)


class MixedNoiseDataset(Dataset):
    """
    支持混合噪声的数据集
    """

    def __init__(self, base_dataset, noise_configs):
        self.base_dataset = base_dataset
        self.noise_configs = noise_configs
        self.weights = [c['weight'] for c in noise_configs]
        self.types = [c['type'] for c in noise_configs]

        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 获取基础数据
        data = self.base_dataset[idx]

        # 随机选择噪声模式
        noise_type = np.random.choice(self.types, p=self.weights)

        # 设置噪声模式
        if noise_type == 'none':
            self.base_dataset.noise_level = 'none'
        else:
            self.base_dataset.noise_level = noise_type

        # 重新获取数据（应用新的噪声模式）
        return self.base_dataset[idx]


def build_dataset(cfg, transform, noise_level='none', alpha_embed=0.016,
                  ring_positions=None, bits_per_ring=None):
    """构建训练数据集 (v18 镂空版)"""
    if ring_positions is None:
        ring_positions = cfg.get('ring_positions', [8, 15])
    if bits_per_ring is None:
        bits_per_ring = cfg.get('bits_per_ring', [20, 40])

    dataset = WatermarkDatasetV11(
        image_dir=cfg['train_dir'],
        transform=transform,
        block_size=cfg.get('block_size', 512),
        num_bits=sum(bits_per_ring),
        alpha_embed=alpha_embed,
        noise_level=noise_level,
        max_rotation=cfg.get('max_rotation', 5.0),
        max_shift=cfg.get('max_shift', 0.5),
        r_watermark=ring_positions,
        bitsf=bits_per_ring,
        max_images=cfg.get('train_length', 0),
        M_w=cfg.get('M_w', 200),
        M_b=cfg.get('M_b', 55),
        hollow_ratio=cfg.get('hollow_ratio', 0.3)
    )
    return dataset


def build_val_dataset(cfg, transform, noise_level='none', alpha_embed=0.016,
                      ring_positions=None, bits_per_ring=None):
    """构建验证数据集 (v18 镂空版)"""
    if ring_positions is None:
        ring_positions = cfg.get('ring_positions', [8, 15])
    if bits_per_ring is None:
        bits_per_ring = cfg.get('bits_per_ring', [20, 40])

    dataset = WatermarkDatasetV11(
        image_dir=cfg['val_dir'],
        transform=transform,
        block_size=cfg.get('block_size', 512),
        num_bits=sum(bits_per_ring),
        alpha_embed=alpha_embed,
        noise_level=noise_level,
        max_rotation=0,  # 验证时不旋转
        max_shift=cfg.get('max_shift', 0.5),
        r_watermark=ring_positions,
        bitsf=bits_per_ring,
        max_images=cfg.get('val_length', 0),
        M_w=cfg.get('M_w', 200),
        M_b=cfg.get('M_b', 55),
        hollow_ratio=cfg.get('hollow_ratio', 0.3)
    )
    return dataset


def validate(model, val_loader, device, num_bits, bits_per_ring=None):
    """验证模型，返回总准确率和每个环的准确率"""
    model.eval()
    total_correct = 0
    total_bits = 0
    all_accs = []

    # 每个环的统计
    if bits_per_ring is None:
        bits_per_ring = [num_bits]
    ring_correct = [0] * len(bits_per_ring)
    ring_total = [0] * len(bits_per_ring)

    with torch.no_grad():
        for images, bits_gt in val_loader:
            images = images.to(device)
            bits_gt = bits_gt.to(device)

            # 前向传播
            logits, _, _ = model(images)
            preds = (logits > 0.5).float()

            # 计算总准确率
            correct = (preds == bits_gt).sum().item()
            total = bits_gt.numel()
            total_correct += correct
            total_bits += total

            # 每个样本的准确率
            acc = (preds == bits_gt).float().mean(dim=1)
            all_accs.extend(acc.cpu().numpy())

            # 每个环的准确率
            start = 0
            for i, n_bits in enumerate(bits_per_ring):
                end = start + n_bits
                ring_correct[i] += (preds[:, start:end] == bits_gt[:, start:end]).sum().item()
                ring_total[i] += bits_gt[:, start:end].numel()
                start = end

    avg_acc = total_correct / total_bits
    ring_accs = [rc / rt if rt > 0 else 0.0 for rc, rt in zip(ring_correct, ring_total)]
    return avg_acc, all_accs, ring_accs


def main():
    parser = argparse.ArgumentParser(description='v18 镂空水印训练 - 纯pair噪声')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default=None, help='GPU编号(覆盖config)')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device or cfg.get('device', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    output_dir = cfg.get('output_dir', f'output/v18_pair_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    logger.info("=" * 60)
    logger.info("v18 镂空水印训练 - 纯pair噪声")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 获取参数
    alpha = cfg.get('alpha', 0.016)
    bits_per_ring = cfg.get('bits_per_ring', [20, 40])
    ring_positions = cfg.get('ring_positions', [8, 15])
    num_bits = sum(bits_per_ring)
    batch_size = cfg.get('batch_size', 32)
    lambda_bit = cfg.get('lambda_bit', 15.0)
    save_every = cfg.get('save_every', 10)  # 每N轮保存一次

    # 微调参数
    finetune_cfg = cfg.get('finetune', {})
    epochs = finetune_cfg.get('epochs', 30)
    lr = finetune_cfg.get('lr', 0.00002)
    weight_path = finetune_cfg.get('weight_path', None)

    # 噪声配置
    noise_config = cfg.get('noise_config', {})
    train_noise_configs = noise_config.get('train_noise', [{'type': 'none', 'weight': 1.0}])
    val_noise = noise_config.get('val_noise', 'none')

    # 镂空参数
    M_w = cfg.get('M_w', 200)
    M_b = cfg.get('M_b', 55)
    hollow_ratio = cfg.get('hollow_ratio', 0.0)

    logger.info(f"圆环位置: {ring_positions}")
    logger.info(f"Bits配置: {bits_per_ring} (共{num_bits}bit)")
    logger.info(f"镂空参数: M_w={M_w}, M_b={M_b}, hollow_ratio={hollow_ratio}")
    logger.info(f"嵌入强度: alpha={alpha}")
    logger.info(f"训练噪声配置: {train_noise_configs}")
    logger.info(f"验证噪声: {val_noise}")
    logger.info(f"学习率: {lr}")
    logger.info(f"训练轮数: {epochs}")
    logger.info(f"保存间隔: 每{save_every}轮")

    # 创建模型
    angle_bins = cfg.get('angle_bins', 200)
    model = WatermarkDecoderV17(
        n_sectors=num_bits,
        bits=bits_per_ring,
        angle_bins=angle_bins,
        radius_bins=12,
        ring_positions_init=[float(r) for r in ring_positions]
    )

    # 加载预训练权重
    if weight_path and os.path.exists(weight_path):
        logger.info(f"加载预训练权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 移除ring_positions，保留模型初始化时的值
        if 'ring_positions' in state_dict:
            del state_dict['ring_positions']

        # 检测v14格式并重映射
        is_v14_format = any(k.startswith('ring_transformers.2.') for k in state_dict.keys())
        if is_v14_format:
            logger.info("检测到v14格式权重，应用ring index重映射")
            mapped_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('ring_transformers.0.'):
                    new_k = k.replace('ring_transformers.0.', 'ring_transformers.0.')
                    mapped_state_dict[new_k] = v
                elif k.startswith('ring_transformers.1.'):
                    new_k = k.replace('ring_transformers.1.', 'ring_transformers.1.')
                    mapped_state_dict[new_k] = v
                elif k.startswith('ring_transformers.2.'):
                    continue  # 跳过第3个ring
                else:
                    mapped_state_dict[k] = v
            state_dict = mapped_state_dict

        model.load_state_dict(state_dict, strict=False)
        logger.info("权重加载完成")
    else:
        logger.warning("未找到预训练权重，从头训练")

    model = model.to(device)

    # 创建数据集
    # 训练数据集使用混合噪声
    if len(train_noise_configs) > 1:
        # 多种噪声模式混合
        base_dataset = build_dataset(cfg, transform, noise_level='none',
                                     alpha_embed=alpha, ring_positions=ring_positions,
                                     bits_per_ring=bits_per_ring)
        train_dataset = MixedNoiseDataset(base_dataset, train_noise_configs)
        logger.info(f"使用混合噪声训练: {train_noise_configs}")
    else:
        # 单一噪声模式
        noise_type = train_noise_configs[0]['type']
        train_dataset = build_dataset(cfg, transform, noise_level=noise_type,
                                      alpha_embed=alpha, ring_positions=ring_positions,
                                      bits_per_ring=bits_per_ring)
        logger.info(f"使用单一噪声训练: {noise_type}")

    # 验证数据集使用指定噪声
    val_dataset = build_val_dataset(cfg, transform, noise_level=val_noise,
                                    alpha_embed=alpha, ring_positions=ring_positions,
                                    bits_per_ring=bits_per_ring)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(val_dataset)}")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    best_val_acc = 0.0
    best_epoch = 0

    logger.info("\n开始训练...")
    logger.info("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_bits = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, bits_gt) in enumerate(pbar):
            images = images.to(device)
            bits_gt = bits_gt.to(device)

            # 前向传播
            logits, _, _ = model(images)
            loss = criterion(logits, bits_gt)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            preds = (logits > 0.5).float()
            total_correct += (preds == bits_gt).sum().item()
            total_bits += bits_gt.numel()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(preds == bits_gt).float().mean().item()*100:.1f}%'
            })

        # 计算训练指标
        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_bits

        # 验证
        val_acc, val_accs, ring_accs = validate(model, val_loader, device, num_bits, bits_per_ring)

        # 学习率调度
        scheduler.step()

        # 计算时间
        epoch_time = time.time() - start_time

        # 日志
        logger.info(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s]")
        logger.info(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        ring_info = " | ".join([f"Ring{i}:{a*100:.1f}%" for i, a in enumerate(ring_accs)])
        logger.info(f"  Val Acc: {val_acc*100:.2f}% ({ring_info})")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, save_path)
            logger.info(f"  ✓ 保存最佳模型 (val_acc={val_acc*100:.2f}%)")

        # 每N轮保存一次权重
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
            }, checkpoint_path)
            logger.info(f"  ✓ 保存checkpoint: checkpoint_epoch_{epoch+1}.pth")

    # 保存最终模型
    final_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
    }, final_path)

    # 保存训练结果
    results = {
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'final_val_acc': float(val_acc),
        'final_train_acc': float(train_acc),
        'total_epochs': epochs,
        'config': cfg,
        'ring_positions': ring_positions,
        'bits_per_ring': bits_per_ring,
        'noise_config': {
            'train': train_noise_configs,
            'val': val_noise,
        }
    }

    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳Epoch: {best_epoch}")
    logger.info(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    logger.info(f"最终验证准确率: {val_acc*100:.2f}%")
    logger.info(f"模型保存位置: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
