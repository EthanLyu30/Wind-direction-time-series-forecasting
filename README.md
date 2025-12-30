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
1. **单步预测**: 使用8小时历史数据预测未来1小时风速
2. **多步预测 (短期)**: 使用8小时历史数据预测未来1小时风速
3. **多步预测 (长期)**: 使用8小时历史数据预测未来16小时风速

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

训练完成的模型保存为 `.pth` 格式，命名规则：`{模型名}_{任务名}.pth`

示例：
- `Linear_singlestep.pth`
- `LSTM_multistep_16h.pth`
- `Transformer_multistep_1h.pth`
- `CNN_LSTM_singlestep.pth`
- ...

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
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
```

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
