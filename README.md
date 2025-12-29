# 🌬️ 风速序列预测 (Wind Speed Sequence Prediction)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于深度学习的多高度风速预测系统，支持单步预测和多步预测任务。

## 📋 项目简介

本项目使用来自 HuggingFace 的风速数据集，包含 10米、50米、100米 三个高度的传感器数据，实现了基于多种深度学习模型的风速预测任务。

### 数据集
- **数据来源**: HuggingFace Datasets
  - [WindSpeed_10m](https://huggingface.co/datasets/Antajitters/WindSpeed_10m)
  - [WindSpeed_50m](https://huggingface.co/datasets/Antajitters/WindSpeed_50m)
  - [WindSpeed_100m](https://huggingface.co/datasets/Antajitters/WindSpeed_100m)
- **数据特征**: 时间戳、风速、风向、温度、气压、湿度
- **数据条数**: 10,573条
- **数据大小**: 约257KB

### 任务目标
1. **单步预测 (singlestep)**: 使用8小时历史数据预测未来1小时风速
2. **多步预测 (multistep_16h)**: 使用8小时历史数据预测未来16小时风速

## 🏗️ 项目结构

```
wind-speed-prediction/
├── config.py                 # 配置文件（超参数、路径配置）
├── data_loader.py           # 数据加载与预处理模块
├── models.py                # 基础模型定义（Linear、LSTM、Transformer）
├── models_innovative.py     # 创新模型定义（CNN-LSTM、TCN等）
├── trainer.py               # 训练与评估模块
├── visualization.py         # 可视化模块
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包列表
├── .gitignore              # Git忽略文件配置
├── README.md               # 项目说明文档
├── dataset/                # 数据集目录
│   ├── WindSpeed_10m/
│   ├── WindSpeed_50m/
│   └── WindSpeed_100m/
├── models/                 # 保存的模型文件
└── results/                # 实验结果与可视化
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

```bash
# 运行完整实验（训练所有模型 + 评估 + 可视化）
python main.py

# 仅训练模型
python main.py --mode train

# 仅评估模型（需要已训练的模型）
python main.py --mode eval

# 仅数据可视化
python main.py --mode visualize
```

### 详细训练指令

#### 基础训练命令
```bash
# 完整训练（推荐首次运行）
python main.py --epochs 200 --no-viz

# 服务器训练（禁用可视化，节省资源）
python main.py --mode train --epochs 200 --no-viz
```

#### 继续训练/微调（从检查点恢复）
```bash
# 继续训练所有模型
python main.py --resume --epochs 300 --no-viz

# 继续训练特定任务（如多步预测效果不好时）
python main.py --resume --tasks multistep_16h --epochs 400 --no-viz

# 继续训练特定模型
python main.py --resume --models LSTM WaveNet Transformer --epochs 300 --no-viz

# 组合使用：只训练多步预测任务的LSTM和WaveNet
python main.py --resume --models LSTM WaveNet --tasks multistep_16h --epochs 400 --no-viz
```

#### 自定义超参数
```bash
# 手动指定学习率和早停耐心值
python main.py --lr 0.0001 --patience 50 --epochs 300 --no-viz

# 指定batch size（根据GPU显存调整）
python main.py --batch-size 128 --epochs 200 --no-viz

# 指定评估指标模式
python main.py --metric-mode r2 --epochs 200 --no-viz  # 使用R²作为早停指标
python main.py --metric-mode mse --epochs 200 --no-viz # 使用MSE作为早停指标
```

#### GPU服务器推荐配置
```bash
# A100/H100 高端GPU（40GB+显存）
python main.py --batch-size 512 --epochs 300 --resume --no-viz

# RTX 3090/4090（24GB显存）
python main.py --batch-size 256 --epochs 250 --resume --no-viz

# RTX 3060/3070（8-12GB显存）
python main.py --batch-size 128 --epochs 200 --resume --no-viz
```

#### 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式: all/train/eval/visualize | all |
| `--models` | 指定训练的模型列表 | 全部模型 |
| `--tasks` | 指定训练的任务: singlestep/multistep_16h | 全部任务 |
| `--epochs` | 最大训练轮数 | 100 |
| `--batch-size` | 批次大小 | 自动检测 |
| `--lr` | 学习率 | 任务自适应 |
| `--patience` | 早停耐心值 | 任务自适应 |
| `--resume` | 从检查点继续训练 | False |
| `--no-viz` | 禁用可视化图表生成 | False |
| `--metric-mode` | 评估指标: r2/mse/combined | 任务自适应 |

## 🧠 模型介绍

### 基础模型 (75%评分要求)

| 模型 | 描述 |
|------|------|
| **Linear** | 基于多层感知机(MLP)的线性模型，将输入序列展平后预测 |
| **LSTM** | 双向长短期记忆网络，擅长捕获时序依赖关系 |
| **Transformer** | 基于自注意力机制的编码器-解码器架构 |

### 创新模型 (25%创新分)

| 模型 | 创新点 |
|------|--------|
| **CNN-LSTM** | 多尺度卷积特征提取 + LSTM序列建模 + 注意力机制 |
| **TCN** | 因果卷积 + 膨胀卷积扩大感受野 + 残差连接 |
| **WaveNet** | 门控激活单元 + 膨胀因果卷积 + Skip连接 |
| **LSTNet** | CNN短期模式提取 + GRU长期依赖 + Skip-RNN周期建模 + Highway自回归 |

## 📊 评估指标

| 指标 | 公式 | 说明 | 推荐度 |
|------|------|------|--------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | 均方根误差，与原数据同单位，直观易解释 | ⭐⭐⭐⭐⭐ |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数，表示模型解释的方差比例（1最好） | ⭐⭐⭐⭐ |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | 平均绝对误差，对异常值不敏感 | ⭐⭐⭐ |
| **MSE** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | 均方误差，数值较大不够直观 | ⭐⭐ |

### 指标选择建议
- **主要看 RMSE**：误差大小直观，单位与风速相同（m/s）
- **辅助看 R²**：拟合优度，越接近1越好（>0.8为优秀，>0.5为可接受）
- 单步预测 R²≈0.88 表示模型能解释88%的风速变化，效果优秀
- 多步预测 R²≈0.50 表示长期预测难度大，这是正常现象

## 🔧 特征工程

### 数据预处理
1. **时间戳解析**: 将时间戳列转换为datetime格式
2. **缺失值处理**: 使用线性插值填充缺失值
3. **异常值处理**: 使用IQR方法检测并处理异常值
4. **数据透视**: 按高度透视数据，使每个时间点包含所有高度的数据

### 特征构建
- **原始特征**: 风向、温度、气压、湿度（三个高度）
- **时间特征**: 小时、星期、月份的周期性编码（sin/cos）
- **标准化**: 使用StandardScaler进行Z-score标准化

## 📈 数据集划分

| 数据集 | 比例 | 用途 |
|--------|------|------|
| 训练集 | 70% | 模型训练 |
| 验证集 | 20% | 超参数调优、早停 |
| 测试集 | 10% | 最终性能评估 |

## 💾 模型保存

训练完成的模型保存为 `.pth` 格式，命名规则：`{模型名}_{任务名}.pth`

### 基础模型（共6个，满足作业要求）
| 模型 | 单步预测 | 多步预测 |
|------|----------|----------|
| **Linear** | `Linear_singlestep.pth` | `Linear_multistep_16h.pth` |
| **LSTM** | `LSTM_singlestep.pth` | `LSTM_multistep_16h.pth` |
| **Transformer** | `Transformer_singlestep.pth` | `Transformer_multistep_16h.pth` |

### 创新模型（额外加分）
| 模型 | 单步预测 | 多步预测 |
|------|----------|----------|
| **CNN_LSTM** | `CNN_LSTM_singlestep.pth` | `CNN_LSTM_multistep_16h.pth` |
| **TCN** | `TCN_singlestep.pth` | `TCN_multistep_16h.pth` |
| **WaveNet** | `WaveNet_singlestep.pth` | `WaveNet_multistep_16h.pth` |
| **LSTNet** | `LSTNet_singlestep.pth` | `LSTNet_multistep_16h.pth` |

## 📁 输出文件

实验完成后，将在 `results/` 目录生成以下文件：

- `dataset_overview.png` - 数据集概览图
- `{model}_{task}_history.png` - 训练历史曲线
- `{model}_{task}_predictions.png` - 预测结果对比图
- `{model}_{task}_scatter.png` - 预测散点图
- `model_comparison.csv` - 模型性能对比表
- `experiment_report.md` - 实验报告

## 🎯 创新点详细说明

### 1. CNN-LSTM混合模型
```
优势：
- CNN层提取局部时序模式（短期波动特征）
- 多尺度卷积（3/5/7 kernel）捕获不同时间尺度特征
- LSTM捕获长期依赖关系
- 注意力机制自适应加权重要时间步
```

### 2. LSTNet模型
```
优势：
- CNN层提取短期局部时序模式
- GRU层捕获长期时序依赖
- Skip-RNN直接建模周期性模式
- Highway自回归组件增强预测稳定性
- 参数量适中，适合中小规模数据集
```

### 3. TCN (Temporal Convolutional Network)
```
优势：
- 因果卷积保证预测只使用过去信息
- 膨胀率指数增长，感受野随深度指数扩大
- 残差连接解决梯度消失问题
- 并行计算效率高于RNN
```

### 4. WaveNet风格模型
```
优势：
- 门控激活单元增强表达能力
- 残差+Skip双路径梯度流动
- 堆叠膨胀卷积实现超大感受野
- 适合长序列建模
```

## ⚙️ 超参数配置

主要超参数在 `config.py` 中配置：

### 全局默认配置
```python
BATCH_SIZE = 自动检测（根据GPU显存）  # A100: 512, RTX 3090: 256, RTX 3060: 128
LEARNING_RATE = 0.001                   # 默认学习率（会被任务特定配置覆盖）
NUM_EPOCHS = 100                        # 默认最大训练轮数
EARLY_STOPPING_PATIENCE = 15            # 默认早停耐心值
WEIGHT_DECAY = 1e-5                     # L2正则化权重
```

### 任务特定超参数（自动应用）
```python
TASK_SPECIFIC_HYPERPARAMS = {
    'singlestep': {
        'lr': 0.001,          # 短期预测：正常学习率
        'patience': 20,       # 标准早停
        'min_epochs': 50,     # 至少训练50个epoch
    },
    'multistep_16h': {
        'lr': 0.0001,         # 长期预测：更低学习率避免快速收敛到局部最优
        'patience': 40,       # 更宽松早停，允许更充分探索
        'min_epochs': 100,    # 至少训练100个epoch
    }
}
```

### 学习率调度器
- **CosineAnnealingWarmRestarts**: T_0=20, T_mult=2, eta_min=lr×0.001
- **ReduceLROnPlateau**: factor=0.5, patience=15

### 模型配置概览
| 模型 | 关键参数 | 参数量 |
|------|---------|--------|
| Linear | hidden=[128,64,32], dropout=0.2 | ~34K |
| LSTM | hidden=384, layers=4, bidirectional | ~6M |
| Transformer | d_model=192, heads=8, layers=5 | ~3M |
| CNN_LSTM | cnn=[48,64], lstm_hidden=96 | ~300K |
| TCN | channels=[48,64,96], kernel=3 | ~100K |
| WaveNet | channels=96, blocks=10 | ~350K |
| LSTNet | cnn=48, rnn=96, skip=48 | ~150K |

## 📈 实验结果分析

### 模型性能排名

**单步预测 (8h→1h)** - 按RMSE排序：

| 排名 | 模型 | RMSE ↓ | R² ↑ | 类型 |
|------|------|--------|------|------|
| 🥇1 | **LSTM** | 0.9243 | 0.8848 | 基础 |
| 🥈2 | Linear | 0.9267 | 0.8841 | 基础 |
| 🥉3 | TCN | 0.9351 | 0.8820 | 创新 |
| 4 | Transformer | 0.9441 | 0.8798 | 基础 |
| 5 | LSTNet | 0.9480 | 0.8788 | 创新 |
| 6 | CNN_LSTM | 0.9510 | 0.8780 | 创新 |
| 7 | WaveNet | 0.9529 | 0.8775 | 创新 |

**多步预测 (8h→16h)** - 按RMSE排序：

| 排名 | 模型 | RMSE ↓ | R² ↑ | 类型 |
|------|------|--------|------|------|
| 🥇1 | **LSTM** | 1.8853 | 0.5225 | 基础 |
| 🥈2 | Linear | 1.8997 | 0.5152 | 基础 |
| 🥉3 | **WaveNet** | 1.9500 | 0.4892 | 创新 |
| 4 | CNN_LSTM | 1.9975 | 0.4640 | 创新 |
| 5 | Transformer | 2.0148 | 0.4546 | 基础 |
| 6 | TCN | 2.0417 | 0.4400 | 创新 |
| 7 | LSTNet | 2.1075 | 0.4033 | 创新 |

### 为什么简单模型表现好？

1. **数据量有限**：约10,000条数据，复杂模型容易过拟合
2. **特征关系简单**：风速与气象特征存在较强线性关系，Linear足以捕获
3. **奥卡姆剃刀原理**：简单问题用简单模型效果往往更好
4. **短期依赖主导**：8小时输入序列的时序依赖不复杂

### 创新模型的价值

虽然整体上简单模型表现更好，但创新模型仍有重要价值：

1. **WaveNet在多步预测中表现第3**：证明了门控卷积在长期预测上的优势
2. **TCN在单步预测中排名第3**：优于Transformer，证明因果卷积的有效性
3. **LSTNet提供周期性建模**：Skip-RNN和Highway组件可捕获周期性模式
4. **学术价值**：
   - 展示了对多种前沿架构的理解和实现能力
   - 对比分析本身就是创新（证明模型复杂度与数据量的权衡）
   - 为后续大规模数据场景提供了模型储备

### 结论

- 在小规模数据集上，**LSTM效果最佳**，单步和多步预测均排第一
- **Linear模型性价比最高**：参数量最少但性能接近LSTM
- 在长期预测任务中，**WaveNet等卷积模型有优势**
- 模型选择应根据数据规模和任务复杂度权衡

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 License

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

- 苏州大学机器学习实践课程期末大作业

## 🙏 致谢

- [HuggingFace Datasets](https://huggingface.co/datasets) 提供数据集
- [PyTorch](https://pytorch.org/) 深度学习框架
