# 水印训练和解码框架

## 概述

本框架实现了基于CWT（Continuous Wavelet Transform）方法的水印生成、叠加和神经网络解码。主要功能包括：

1. 使用 `generate_message_watermark_template2` 方法生成平滑的水印模板
2. 将水印模板与图像进行透明度叠加
3. 使用PyTorch神经网络模型解码水印

## 目录结构

```
fftmask/
├── cwt_method.py         # CWT水印方法实现
├── watermark_trainer.py   # 训练框架
├── watermark_decoder.py   # 解码脚本
├── test_watermark.py      # 测试脚本
└── README.md              # 本说明文件
```

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- OpenCV
- NumPy
- scikit-image
- scipy

## 安装依赖

```bash
pip install torch torchvision opencv-python numpy scikit-image scipy
```

## 数据集准备

1. 创建以下目录结构：

```
fftmask/
└── data/
    ├── train/    # 训练图像
    ├── val/      # 验证图像
    └── test/     # 测试图像
```

2. 在每个目录中放入足够的图像文件（建议至少100张图像用于训练）。

## 训练模型

```bash
python watermark_trainer.py
```

训练过程会自动生成水印并与图像叠加，然后训练神经网络模型来解码水印。训练完成后，最佳模型会保存在 `best_watermark_decoder.pth`。

## 解码水印

```bash
python watermark_decoder.py --image path/to/image.jpg --model best_watermark_decoder.pth
```

## 测试完整流程

```bash
python test_watermark.py
```

该脚本会：
1. 生成随机水印
2. 创建水印模板
3. 与测试图像叠加
4. 使用训练好的模型解码水印
5. 计算解码准确率

## 核心功能说明

### 水印模板生成

使用 `generate_message_watermark_template2` 方法生成平滑的水印模板。该方法：
- 在频域的低频环形条带中嵌入水印
- 通过逆DFT转换到空域
- 生成二值化的平滑水印模板

### 水印叠加

使用透明度叠加方法将水印模板与图像混合：
```python
alpha = 0.3  # 水印透明度
watermarked_image = (1 - alpha) * image_float + alpha * Tm_float[:,:,np.newaxis]
```

### 神经网络解码器

使用卷积神经网络提取图像特征并解码水印。网络结构：
- 4层卷积层用于特征提取
- 全连接层用于水印解码
- 输出64位二进制水印

## 性能评估

训练过程中会计算验证集上的损失和准确率。测试脚本会计算解码准确率，评估模型性能。

## 注意事项

- 所有图像会被调整为512x512大小
- 水印长度固定为64位
- 训练时间取决于数据集大小和硬件性能
