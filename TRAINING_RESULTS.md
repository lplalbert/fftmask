# 训练结果对比

## 实验配置

所有实验使用相同配置：
- 嵌入半径：r=[12, 25]
- 每环bit数：bits=[15, 45]（共60bit）
- 嵌入强度：alpha=0.0191
- 训练噪声：pair（随机选择2种：identity/wechat/tile_crop/pimog）
- 预训练模型：best_cb_decoder.pth

## 通道对比

| 方案 | 通道 | 镂空 | 验证噪声 | Val Acc | 备注 |
|------|------|------|----------|---------|------|
| v17 | Cb | 无 | 无噪声 | **99.89%** | Cb通道，干净验证 |
| v18 | Cb | 0.3 | 无噪声 | **99.86%** | Cb通道，镂空模板 |
| v17 | B | 无 | pair噪声 | **97.3%** | B通道，带噪声验证 |
| v18 | B | 0.3 | pair噪声 | **97.5%** | B通道，镂空模板 |

## 关键发现

1. **Cb vs B通道**：Cb通道在干净验证下准确率极高（99.8%+），B通道在pair噪声下仍有97%+
2. **v17 vs v18**：两者在各自通道上表现接近，v18略优
3. **验证噪声影响**：B通道验证也加了pair噪声，Cb通道验证无噪声，因此不能直接比较绝对值

## 模型路径

- v17 Cb：`output/v17_pair/best_model.pth`
- v18 Cb：`output/v18_hollow_pair/best_model.pth`
- v17 B：`output/v17_b_pair/best_model.pth`
- v18 B：`output/v18_b_pair/best_model.pth`
