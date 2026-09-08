# watermark_tools

这是显示端的水印工具目录。支持v17和v18两种水印模板的生成和显示。

## 功能概述

- **v17**：实心环形水印，r=[12,25]，alpha=0.016，Cb通道
- **v18**：镂空环形水印，r=[12,25]，hollow_ratio=0.3，alpha=0.0228，B通道
- **二维码**：每个模板都包含版本、通道、bit序列信息的二维码
- **Qt透明叠加**：实时显示，无需修改原图，支持点击穿透

## Quick Start

```bash
# 1. 生成模板（10个v17 + 10个v18，含B/Cb通道和二维码）
python watermark_tools/generate_v17_v18_templates.py

# 2. 预览所有模板（含二维码）
python watermark_tools/preview_all_templates.py

# 3. Qt透明窗口显示（推荐）
python watermark_tools/qt_display.py --channel cb --alpha 0.016

# 或使用传统Tkinter方式
python watermark_tools/display_with_watermark.py
```

## 生成模板

```bash
python watermark_tools/generate_v17_v18_templates.py \
  --output_dir watermark_tools/generated_templates \
  --seed 2026 --count 10 --size 512
```

### 生成内容

每个模板生成5个文件：
- `{version}_{index}_{bits}.png` - 灰度模板（训练/推理用）
- `{version}_{index}_{bits}_on_cb.png` - BGR域模板（嵌入Cb通道）
- `{version}_{index}_{bits}_on_b.png` - BGR域模板（嵌入B通道）
- `{version}_{index}_{bits}_qr_cb.png` - 二维码（Cb通道版本）
- `{version}_{index}_{bits}_qr_b.png` - 二维码（B通道版本）

### 二维码内容格式

```
v17|cb|01|100100001011111011000111011101111000000011000110001000010010
```

格式：`{version}|{channel}|{index}|{bit_string}`

## Qt透明窗口显示

基于test6.py的实现方式：
1. 水印模板嵌入到灰度128画布的Cb/B通道（预生成）
2. Qt透明窗口以`setWindowOpacity(alpha)`叠加整个画布
3. Windows API设置`WS_EX_LAYERED|WS_EX_TRANSPARENT`实现点击穿透

### 参数

- `--channel cb|b`：选择嵌入通道（默认cb）
- `--alpha`：窗口透明度/水印强度（默认0.016）
- `--version v17|v18|both`：选择水印版本（默认both）
- `--index N`：指定显示第N个模板（-1=循环切换）
- `--interval`：循环切换间隔毫秒（默认5000）

### 使用示例

```bash
# 显示Cb通道模板
python watermark_tools/qt_display.py --channel cb --alpha 0.016

# 显示B通道模板
python watermark_tools/qt_display.py --channel b --alpha 0.016

# 只显示v18版本，指定模板索引
python watermark_tools/qt_display.py --version v18 --channel cb --index 0

# 循环切换所有模板
python watermark_tools/qt_display.py --version both --interval 3000
```

### 操作

- **ESC**：退出
- **左/右箭头**：手动切换模板
- **自动切换**：按`--interval`设置的间隔自动循环

## 参数说明

### v17参数

- `r_watermark=[12, 25]`：嵌入环的内外半径
- `bitsf=[15, 45]`：两个环的bit数（共60bit）
- `alpha=0.016`：等效嵌入强度
- `channel=cb`：嵌入Cb通道（蓝差色度，人眼最不敏感）

### v18参数

- `r_watermark=[12, 25]`：嵌入环的内外半径
- `bitsf=[15, 45]`：两个环的bit数（共60bit）
- `hollow_ratio=0.3`：镂空比例（30%白像素随机去除）
- `alpha=0.0228`：等效嵌入强度（因镂空提高）
- `channel=b`：嵌入B通道

### 等效强度计算

v17白像素数=129899（49.55%），v18白像素数=91099（34.75%）

```
alpha_v18 = alpha_v17 / (1 - hollow_ratio)
         = 0.016 / 0.7
         ≈ 0.0228
```

## 文件结构

```
watermark_tools/
├── generate_v17_v18_templates.py  # 生成v17/v18模板（含BGR和二维码）
├── qt_display.py                  # Qt透明窗口叠加显示
├── display_with_watermark.py      # 传统Tkinter显示（可选）
├── preview_all_templates.py       # 预览所有模板
├── preview_blend.py               # 预览嵌入效果
├── README.md                      # 本文档
├── clean_image/                   # 干净图片素材
└── generated_templates/           # 生成的模板目录
    ├── manifest.json              # JSON格式元数据
    ├── manifest.csv               # CSV格式元数据
    ├── v17/                       # v17模板
    │   ├── v17_00_*.png          # 灰度模板
    │   ├── v17_00_*_on_cb.png    # Cb通道模板
    │   ├── v17_00_*_on_b.png     # B通道模板
    │   ├── v17_00_*_qr_cb.png    # 二维码（Cb通道）
    │   └── v17_00_*_qr_b.png     # 二维码（B通道）
    └── v18/                       # v18模板
        └── ...
```

## 依赖

```bash
pip install numpy opencv-python qrcode[pil] PyQt5
```
