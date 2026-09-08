# 水印工具使用命令

## 1. 生成模板

```bash
python watermark_tools/generate_v17_v18_templates.py --count 10 --seed 2026
```

## 2. 同时运行：图片 + 二维码 + 水印叠加

```bash
# v17 + Cb通道 (alpha=0.016)
python watermark_tools/run_both.py --version v17 --channel cb

# v17 + B通道
python watermark_tools/run_both.py --version v17 --channel b

# v18 + Cb通道 (等效alpha=0.0228)
python watermark_tools/run_both.py --version v18 --channel cb

# v18 + B通道
python watermark_tools/run_both.py --version v18 --channel b

# 加四角回字形标记
python watermark_tools/run_both.py --version v17 --channel cb --corners
```

### 图层顺序（从下到上）

```
┌─────────────────────────────┐
│  水印透明层 (qt_display.py) │  ← 最上层，alpha融合
├─────────────────────────────┤
│  二维码 (左上角)             │  ← 贴在图片上
├─────────────────────────────┤
│  干净图片 (image_slideshow)  │  ← 底层
└─────────────────────────────┘
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--version` | v17 | v17/v18/both |
| `--channel` | cb | cb/b |
| `--alpha` | 0.016 | 水印强度（v18自动转0.0228） |
| `--interval` | 3000 | 图片切换间隔ms |
| `--wm_interval` | 5000 | 水印模板切换间隔ms |
| `--index` | 0 | 模板/二维码索引 |

## 3. 单独运行

```bash
# 只播放图片+二维码
python watermark_tools/image_slideshow.py --version v17 --index 0

# 只显示水印层
python watermark_tools/qt_display.py --version v17 --channel cb --alpha 0.016

# 嵌入水印到图片查看效果
python watermark_tools/embed_and_show.py
```
