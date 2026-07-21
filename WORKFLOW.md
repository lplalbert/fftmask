# Cb通道频域水印系统 - 正确方案总结

## 系统概述

在YCrCb色彩空间的Cb通道中嵌入频域水印，通过FFT→极坐标变换→Transformer解码器提取60bit水印信号。

## 核心参数（必须一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_bits` | 60 | 水印总bit数 |
| `r` | [12, 25] | 两个频域环的半径 |
| `bitsf` | [15, 45] | 每个环承载的bit数（15+45=60） |
| `ring_width` | 5 | 环的半宽度，实际环为(r-5, r+5) |
| `radius_bins` | 12 | 极坐标径向采样数（硬编码在模型中） |
| `k1` | 30000 | 水印模板缩放系数 |
| `alpha_embed` | 0.016 | 嵌入强度 |
| `block_size` | 512 | 训练/解码的crop大小 |

## 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `watermark_decoder3.py` | 解码器模型定义（AdvancedWatermarkDecoder），radius_bins=12硬编码 |
| `encode2.py` | 水印模板生成（Watermark16Sector1） |
| `dataset.py` | 训练数据集，负责嵌入水印+加噪声 |
| `noise_utils.py` | 噪声增强工具 |
| `train_cb.py` | 训练脚本 |
| `decode_channel_watermark.py` | 单图解码脚本 |
| `sweep_decode_channel_watermark.py` | 多尺寸裁剪搜索解码 |
| `config/train_cb_v1_valnoise.yaml` | 训练配置文件 |

### 训练流程

```bash
python train_cb.py --config config/train_cb_v1_valnoise.yaml
```

关键配置：
- 数据源：COCO + document数据集
- 噪声策略：pair模式，alpha=0.016
- 损失函数：lambda_bit=15.0, lambda_shape=0.04

### 水印嵌入

```python
from encode2 import Watermark16Sector1

wm = Watermark16Sector1(
    seed=2026,
    r=[12, 25],
    bits=[15, 45],
    k1=30000,
    alpha=0.016
)
# 生成模板并嵌入到Cb通道
template = wm.make_template()
```

### 单图解码

```bash
python decode_channel_watermark.py \
  --input image.png \
  --model_path best_cb_decoder.pth \
  --bits_file bits.txt \
  --channel cb \
  --r 12 25 \
  --ring_width 5
```

### 多尺寸搜索解码

```bash
python sweep_decode_channel_watermark.py \
  --input_dir images/ \
  --model_path best_cb_decoder.pth \
  --bits_file bits.txt \
  --channels cb \
  --min_edges 1024 2048 \
  --tile_sizes 1024 2048 \
  --ring_width 5
```

**重要**：`--ring_width` 默认值必须为5，否则模型初始化参数错误导致解码失败。

## 解码流程

1. 读取BGR图像
2. 转换为YCrCb，提取Cb通道
3. Resize最短边到目标尺寸（如1024）
4. 裁剪512×512的crop
5. 归一化到[-1, 1]：`tensor / 127.5 - 1.0`
6. 输入AdvancedWatermarkDecoder
7. 输出60个logits，sigmoid后>0.5判为1

## 常见错误

1. **ring_width不匹配**：训练用5，解码必须用5，不能用默认的1
2. **radius_bins参数**：新版模型硬编码为12，不要传参
3. **通道选择**：必须用Cb通道（ycrcb[:,:,2]），不是Y或Cr
4. **归一化**：必须是`/127.5 - 1.0`，不是`/255`
