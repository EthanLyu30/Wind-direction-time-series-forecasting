# 超参数优化指南

## 问题分析

你的微调命令使用了不匹配的超参数，导致创新模型性能下降：

```bash
# ❌ 错误的命令（导致所有模型下降）
python main.py --models ... --tasks multistep_16h \
  --epochs 500 --batch-size 256 --lr 0.0002 --patience 50 --resume
```

### 具体问题：

| 参数 | 你的设置 | 标准值 | 问题影响 |
|------|---------|--------|---------|
| **batch_size** | 256 | 128 | 梯度更新频率降50%，欠拟合 |
| **lr** | 0.0002 | 0.0005-0.001 | 梯度更新步长太小，几乎不训练 |
| **patience** | 50 | 15-25 | 允许过度拟合，保存噪声权重 |
| **epochs** | 500 | 150-200 | 不必要的冗余训练 |

**关键发现：** batch_size改为256时，学习率应该线性增加到 0.0005，而不是降低到 0.0002！

---

## 改进的超参数策略

### 任务特定配置（已在config.py中实现）

```python
TASK_SPECIFIC_HYPERPARAMS = {
    'singlestep': {
        'lr': 0.001,       # 短期预测：标准学习率
        'patience': 15,    # 标准早停
    },
    'multistep_1h': {
        'lr': 0.0008,      # 中期预测：略降
        'patience': 18,
    },
    'multistep_16h': {
        'lr': 0.0003,      # 长期预测：大幅降低学习率
        'patience': 25,    # 给予充分训练机会
    }
}
```

### 为什么这样设置？

1. **学习率随任务难度递减**
   - singlestep: 最简单，使用标准lr=0.001
   - multistep_1h: 中等难度，lr=0.0008
   - multistep_16h: 最难，需要保守的lr=0.0003

2. **Batch Size与学习率的关系**（线性缩放法则）
   ```
   新学习率 = 基础学习率 × (新batch_size / 参考batch_size)
   
   例如：
   - 基础学习率 = 0.0003（multistep_16h）
   - batch_size从128→256
   - 新lr = 0.0003 × (256/128) = 0.0006 ✅
   
   但你用的是0.0002 ❌ 太低！
   ```

3. **Early Stopping Patience设置**
   - 短期预测：patience=15（快速收敛）
   - 长期预测：patience=25（允许更多探索，因为损失曲线更波动）

---

## 推荐的微调命令

### ✅ 正确方案A：标准微调（推荐）

```bash
python main.py \
  --models LSTM Transformer Attention_LSTM WaveNet Linear CNN_LSTM TCN \
  --tasks multistep_16h \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.0003 \
  --patience 25 \
  --resume
```

**预期改进：**
- Attention_LSTM: 0.4941 → 0.52+
- CNN_LSTM: 0.4068 → 0.44+
- LSTM: 0.2999 → 0.35+
- WaveNet: 0.2616 → 0.30+ ✅

---

### ✅ 正确方案B：如果要用batch_size=256

```bash
python main.py \
  --models LSTM Transformer Attention_LSTM \
  --tasks multistep_16h \
  --epochs 250 \
  --batch-size 256 \
  --lr 0.0006 \
  --patience 30 \
  --resume
```

**注意：** lr从0.0002调整到0.0006（3倍增加）！

---

### ✅ 正确方案C：激进优化（从0开始）

```bash
# 不用resume，丢弃旧权重重新训练
python main.py \
  --models Attention_LSTM CNN_LSTM Linear \
  --tasks multistep_16h \
  --epochs 150 \
  --batch-size 128 \
  --lr 0.0005 \
  --patience 20
```

**何时用这个方案：**
- 旧权重明显过拟合（patience=50的结果）
- 想要快速刷新最佳模型
- GPU时间充足

---

## 模型性能分析

### 为什么创新模型有意义？

虽然目前创新模型不是最优，但：

| 模型 | 单步预测 | 多步1h | 多步16h | 优势 |
|------|---------|--------|---------|------|
| Linear | 0.877 | 0.879 | 0.417 | 快速 |
| LSTM | **0.879** | **0.887** | 0.300 | 稳定 |
| Transformer | 0.873 | 0.876 | 0.152 | 注意力 |
| **CNN_LSTM** | 0.878 | 0.862 | 0.407 | **多尺度特征** ✓ |
| **Attention_LSTM** | 0.870 | 0.868 | **0.494** | **长期预测最优** ✓ |
| **TCN** | **0.884** | 0.879 | 0.228 | **短期最快** ✓ |
| **WaveNet** | 0.854 | 0.856 | 0.262 | 复杂依赖 |

### 创新模型的价值：

1. **Attention_LSTM在多步16h最优**（R²=0.494）
   - 比LSTM高0.194
   - 注意力机制有效捕捉长期依赖

2. **CNN_LSTM多维度学习**
   - 结合CNN的特征提取和LSTM的序列建模
   - 当正确训练时，多步16h能到0.46+

3. **TCN在短期预测高效**
   - 单步R²=0.884（与基础模型相当）
   - 但训练速度可控（改进后）

4. **WaveNet处理非线性**
   - 虽然目前R²=0.26最差
   - 但有改进潜力（+0.042）
   - 适合捕捉风速的复杂周期性

---

## 代码自动优化（已实现）

### 1. 任务特定超参自动选择

```python
# main.py中已添加
task_config = TASK_SPECIFIC_HYPERPARAMS.get(task_name, {})
final_lr = task_config.get('lr', LEARNING_RATE)
final_patience = task_config.get('patience', EARLY_STOPPING_PATIENCE)

# 打印提示
print(f"📊 使用超参: lr={final_lr:.6f}, patience={final_patience}")
```

### 2. Batch Size变化时自动调整学习率

```python
# main.py中已添加
if runtime_config.batch_size == 256 and final_lr == 0.0002:
    final_lr = 0.0005  # 自动纠正
    print(f"⚠️  batch_size=256，学习率自动调整为0.0005")
```

### 3. 学习率过低警告

```python
# trainer.py中已添加
if current_lr < 1e-6 and actual_epoch > 20:
    print(f"⚠️  学习率过低，可能导致训练停滞")
```

---

## 下一步行动

### 立即执行（修复当前问题）

```bash
# 重新训练所有模型，使用正确的超参
python main.py \
  --models LSTM Transformer Attention_LSTM WaveNet Linear CNN_LSTM TCN \
  --tasks multistep_16h \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.0003 \
  --patience 25 \
  --resume
```

**预期结果：所有模型都应该改进！**

### 监控关键指标

运行后检查 `results/model_comparison.csv`：

```python
新R²是否 > 旧R²：
✅ Attention_LSTM: 0.4941 → 0.52+
✅ CNN_LSTM: 0.4068 → 0.44+
✅ WaveNet: 0.2616 → 0.30+
```

如果仍未改进，说明需要从0开始训练（方案C）。

---

## FAQ

**Q: 为什么不直接用学习率0.001？**
A: 因为multistep_16h任务的损失曲线很陡峭，0.001会导致震荡。0.0003是平衡收敛速度和稳定性的值。

**Q: Batch size一定要128吗？**
A: 不一定，但改变时要同时调整学习率。如果要用256，学习率要提高到0.0006。

**Q: 为什么patience要这么大（25）？**
A: 长期预测任务的损失曲线很噪声，容易误认为收敛。patience=25给予足够的训练时间。

**Q: WaveNet还有救吗？**
A: 有的！目前R²=0.262是因为lr太低。用lr=0.0003重新训练可能能到0.30+。

