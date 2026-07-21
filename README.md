# FFT频域水印系统

基于FFT频域的图像水印嵌入与解码系统，在YCrCb色彩空间的Cb通道中嵌入60bit水印信号。

## 快速开始

### 训练

```bash
python train_cb.py --config config/train_cb_v1_valnoise.yaml
```

### 水印嵌入

```python
from encode2 import Watermark16Sector1

wm = Watermark16Sector1(seed=2026, r=[12, 25], bits=[15, 45], k1=30000)
template = wm.make_template()
# 嵌入到Cb通道
```

### 单图解码

```bash
python decode_channel_watermark.py \
  --input image.png \
  --model_path best_cb_decoder.pth \
  --bits_file bits.txt \
  --channel cb
```

### 批量搜索解码

```bash
python sweep_decode_channel_watermark.py \
  --input_dir images/ \
  --model_path best_cb_decoder.pth \
  --bits_file bits.txt \
  --channels cb \
  --min_edges 1024
```

## 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_bits | 60 | 水印总bit数 |
| r | [12, 25] | 频域环半径 |
| bitsf | [15, 45] | 每环bit数 |
| ring_width | 5 | 环半宽度 |
| k1 | 30000 | 模板缩放系数 |
| alpha_embed | 0.016 | 嵌入强度 |

## 文件结构

- `watermark_decoder3.py` - 解码器模型（AdvancedWatermarkDecoder）
- `encode2.py` - 水印模板生成（Watermark16Sector1）
- `dataset.py` - 训练数据集
- `noise_utils.py` - 噪声增强
- `train_cb.py` - 训练脚本
- `decode_channel_watermark.py` - 单图解码
- `sweep_decode_channel_watermark.py` - 多尺寸搜索解码
- `config/` - 训练配置
- `best_cb_decoder.pth` - 预训练模型

详细说明见 [WORKFLOW.md](WORKFLOW.md)
