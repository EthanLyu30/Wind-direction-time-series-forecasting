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
| 任务类型 | 输入 | 输出 | 说明 |
|----------|------|------|------|
| **单步预测** | 8小时 | 1小时 | 短期风速预测 |
| **多步预测** | 8小时 | 16小时 | 长期风速预测 |

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
├── README.md               # 项目说明文档
├── dataset/                # 数据集目录
│   ├── WindSpeed_10m/
│   ├── WindSpeed_50m/
│   └── WindSpeed_100m/
├── models/                 # 保存的模型文件（共14个）
│   ├── Linear_singlestep.pth
│   ├── Linear_multistep.pth
│   ├── LSTM_singlestep.pth
│   ├── LSTM_multistep.pth
│   ├── Transformer_singlestep.pth
│   ├── Transformer_multistep.pth
│   └── ... (创新模型)
├── logs/                   # 训练日志
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

# 禁用可视化（服务器推荐）
python main.py --mode train --no-viz

# 从检查点继续训练（微调）
python main.py --mode train --resume --epochs 200

# 训练指定模型
python main.py --mode train --models LSTM Transformer

# 训练指定任务
python main.py --mode train --tasks singlestep

# 调整超参数
python main.py --mode train --lr 0.0005 --batch-size 128 --epochs 150
```

## 🧠 模型介绍

### 基础模型（3个，满足作业要求）

| 模型 | 描述 | 参数量 |
|------|------|--------|
| **Linear** | 基于多层感知机(MLP)的线性模型，将输入序列展平后预测 | ~30K |
| **LSTM** | 双向长短期记忆网络，擅长捕获时序依赖关系 | ~500K |
| **Transformer** | 基于自注意力机制的编码器-解码器架构 | ~800K |

### 创新模型（4个，额外创新分）

| 模型 | 创新点 |
|------|--------|
| **CNN-LSTM** | 多尺度卷积特征提取 + LSTM序列建模 + 注意力机制 |
| **Attention-LSTM** | 自注意力增强 + 时序注意力 + 多头并行处理 |
| **TCN** | 因果卷积 + 膨胀卷积扩大感受野 + 残差连接 |
| **WaveNet** | 门控激活单元 + 膨胀因果卷积 + Skip连接 |

## 📊 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **MSE** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | 均方误差 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | 均方根误差 |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ | 平均绝对误差 |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数 |

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

训练完成的模型保存为 `.pth` 格式，共 **14个模型文件**（7个模型 × 2个任务）。

命名规则：`{模型名}_{任务名}.pth`

### 基础模型（6个，满足作业要求）
| 模型 | 单步预测 | 多步预测 |
|------|----------|----------|
| Linear | `Linear_singlestep.pth` | `Linear_multistep.pth` |
| LSTM | `LSTM_singlestep.pth` | `LSTM_multistep.pth` |
| Transformer | `Transformer_singlestep.pth` | `Transformer_multistep.pth` |

### 创新模型（8个）
| 模型 | 单步预测 | 多步预测 |
|------|----------|----------|
| CNN_LSTM | `CNN_LSTM_singlestep.pth` | `CNN_LSTM_multistep.pth` |
| Attention_LSTM | `Attention_LSTM_singlestep.pth` | `Attention_LSTM_multistep.pth` |
| TCN | `TCN_singlestep.pth` | `TCN_multistep.pth` |
| WaveNet | `WaveNet_singlestep.pth` | `WaveNet_multistep.pth` |

## 📁 输出文件

实验完成后，将在 `results/` 目录生成以下文件：

- `model_comparison.csv` - 模型性能对比表
- `experiment_report.md` - 实验报告
- `{model}_{task}_history.png` - 训练历史曲线
- `{model}_{task}_predictions.png` - 预测结果对比图
- `{model}_{task}_scatter.png` - 预测散点图
- `comparison_{metric}.png` - 模型对比图

## 🎯 创新点详细说明

### 1. CNN-LSTM混合模型
```
优势：
- CNN层提取局部时序模式（短期波动特征）
- 多尺度卷积（3/5/7 kernel）捕获不同时间尺度特征
- LSTM捕获长期依赖关系
- 注意力机制自适应加权重要时间步
```

### 2. Attention-LSTM模型
```
优势：
- 自注意力增强输入特征表示
- 时序注意力聚焦关键历史时间点
- 多头注意力并行处理不同子空间信息
- 层归一化稳定训练
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

```python
# 序列配置
SINGLE_STEP_INPUT_LEN = 8    # 单步预测：8小时输入
SINGLE_STEP_OUTPUT_LEN = 1   # 单步预测：1小时输出
MULTI_STEP_INPUT_LEN = 8     # 多步预测：8小时输入
MULTI_STEP_OUTPUT_LEN = 16   # 多步预测：16小时输出

# 训练配置
BATCH_SIZE = 64              # 批次大小
LEARNING_RATE = 0.001        # 学习率
NUM_EPOCHS = 100             # 最大训练轮数
EARLY_STOPPING_PATIENCE = 15 # 早停耐心值

# 数据集划分
TRAIN_RATIO = 0.7            # 训练集比例
VAL_RATIO = 0.2              # 验证集比例
TEST_RATIO = 0.1             # 测试集比例
```

## 📊 实验结果

### 单步预测 (8h → 1h)
| 模型 | MSE | RMSE | MAE | R² |
|------|-----|------|-----|-----|
| LSTM | 0.84 | 0.92 | 0.69 | **0.887** |
| Linear | 0.86 | 0.93 | 0.71 | 0.884 |
| TCN | 0.87 | 0.94 | 0.70 | 0.882 |

### 多步预测 (8h → 16h)
| 模型 | MSE | RMSE | MAE | R² |
|------|-----|------|-----|-----|
| Linear | 3.73 | 1.93 | 1.54 | **0.499** |
| WaveNet | 3.84 | 1.96 | 1.55 | 0.484 |
| LSTM | 3.87 | 1.97 | 1.55 | 0.480 |

## 📄 License

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

- 苏州大学机器学习实践课程期末大作业

## 🙏 致谢

- [HuggingFace Datasets](https://huggingface.co/datasets) 提供数据集
- [PyTorch](https://pytorch.org/) 深度学习框架
