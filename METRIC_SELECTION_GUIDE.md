# 📊 预测任务评估指标选择指南

## 问题背景

在时间序列预测任务中，选择正确的评估指标对模型选择和调优至关重要。本项目涉及三个预测任务：

| 任务 | 输入 | 输出 | 难度 |
|------|------|------|------|
| singlestep | 8h | 1h | ⭐ 简单 |
| multistep_1h | 8h | 1h | ⭐ 简单 |
| multistep_16h | 24h | 16h | ⭐⭐⭐ 困难 |

---

## 🔬 MSE vs R² 的本质区别

### MSE (Mean Squared Error)
```
MSE = Σ(y_true - y_pred)² / n
```
- **含义**：预测误差的平方均值
- **范围**：0 到 +∞
- **特点**：绝对误差，数值与数据尺度强相关
- **越小越好**

### R² (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot) = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²
```
- **含义**：模型解释的方差比例
- **范围**：-∞ 到 1（通常0-1）
- **特点**：相对指标，不受数据尺度影响
- **越大越好**

---

## 📈 实际数据分析

从 `results/model_comparison.csv` 可以看到：

### 短期预测（singlestep/multistep_1h）
```
MSE ≈ 0.86 - 1.05  （范围窄，差异小）
R²  ≈ 0.86 - 0.88  （范围窄，差异小）
```
**结论**：MSE和R²都能有效区分模型，MSE更直观

### 长期预测（multistep_16h）
```
MSE ≈ 3.5 - 5.2   （范围大，波动剧烈）
R²  ≈ 0.30 - 0.52 （范围适中，更稳定）
```
**结论**：MSE波动太大且绝对值无参考意义，R²更能反映相对能力

---

## 🎯 为什么16h长期预测用R²更合理？

### 1. 误差累积效应
长期预测（16步）的误差会累积：
- 每一步的小误差会传播到下一步
- 16步后，总MSE天然比1步大3-5倍
- 这不一定说明模型"差"，只是任务本身难

### 2. MSE的不稳定性
```
# 假设两个模型在不同epoch的验证MSE：
Epoch 50: Model A MSE=3.8, Model B MSE=4.2  → A看起来更好
Epoch 51: Model A MSE=4.5, Model B MSE=4.0  → B看起来更好
```
长期预测中MSE波动剧烈，早停可能因噪声过早触发

### 3. R²的相对性优势
R² = 1 - MSE/Var(y)

R²自动除以目标变量的方差，消除了尺度影响：
- 短期预测 Var(y) 小 → MSE小也有意义
- 长期预测 Var(y) 大 → 需要用R²归一化

### 4. 业务解释性
- MSE=4.0 说明什么？很难直观理解
- R²=0.52 说明模型解释了52%的方差，直观明确

---

## ✅ 推荐的评估指标策略

| 任务 | 早停判断 | 原因 |
|------|----------|------|
| **singlestep** | MSE (mode='min') | 误差范围小，MSE敏感且稳定 |
| **multistep_1h** | MSE (mode='min') | 与singlestep类似 |
| **multistep_16h** | R² (mode='max') | 误差累积大，R²更稳定 |

---

## 🛠️ 如何使用微调脚本

### 短期预测微调（使用MSE）
```bash
# 微调所有短期任务的所有模型
python finetune_shortterm.py

# 仅微调singlestep任务
python finetune_shortterm.py --task singlestep

# 仅微调指定模型
python finetune_shortterm.py --models LSTM Transformer

# 自定义超参
python finetune_shortterm.py --lr 0.0002 --epochs 100 --patience 30
```

### 长期预测微调（使用R²）
```bash
# 微调所有模型（16h任务）
python finetune_longterm_r2.py

# 仅微调指定模型
python finetune_longterm_r2.py --models Attention_LSTM LSTM

# 自定义超参
python finetune_longterm_r2.py --lr 0.0001 --epochs 150 --patience 40
```

---

## 📋 微调超参配置

### 短期预测超参
```python
FINETUNE_CONFIG = {
    'singlestep': {
        'lr': 0.0003,      # 较低学习率
        'epochs': 80,      # 足够的轮数
        'patience': 25,    # 宽松早停
    },
    'multistep_1h': {
        'lr': 0.00025,     # 更低学习率
        'epochs': 100,     # 更多轮数
        'patience': 30,    # 更宽松
    }
}
```

### 长期预测超参
```python
FINETUNE_CONFIG = {
    'lr': 0.0002,          # 非常低的学习率
    'epochs': 120,         # 充足的轮数
    'patience': 35,        # 非常宽松的早停
    'optimization': 'R2',  # 使用R²作为判断标准
}
```

---

## 🔄 训练流程对比

### 标准训练（MSE）
```
EarlyStopping(mode='min')
↓
if val_loss < best_val_loss:
    save_model()
```

### R²优化训练
```
EarlyStopping(mode='max')
↓
if val_R2 > best_val_R2:
    save_model()
```

---

## 📊 预期效果

使用正确的评估指标后，模型选择会更准确：

### 之前（都用MSE）
- 16h任务可能选到MSE较低但R²较差的模型
- 早停可能因MSE波动过早触发

### 之后（16h用R²）
- 选择真正解释力更强的模型
- 训练更稳定，减少过早停止

---

## 📝 总结

1. **短期预测（1步/1h）**：MSE足够，保持现状
2. **长期预测（16h）**：改用R²作为判断标准
3. **微调时**：使用更低的学习率、更长的训练周期、更宽松的早停

这种差异化的指标选择策略是**合理且推荐的**，因为不同预测任务有本质不同的特性。
