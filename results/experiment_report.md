# 风速序列预测实验报告

**课程名称**: 机器学习实践  
**实验日期**: 2025年12月  
**实验类型**: 时间序列预测  

---

## 摘要

本实验基于深度学习方法，利用来自HuggingFace的多高度风速数据集，实现了风速的单步预测（8小时→1小时）和多步预测（8小时→16小时）任务。实验对比了3种基础模型（Linear、LSTM、Transformer）和9种创新模型的性能，采用MSE、RMSE、MAE、R²等指标进行评估。在实验过程中，我们经历了任务理解偏差、计算资源迁移、超参数调优等多个关键阶段，最终获得了显著的性能提升。实验结果表明：在单步预测任务中，LSTM模型以R²=0.8870取得最佳性能；在多步预测任务中，创新模型WaveNet以R²=0.5375显著优于基础模型。本报告详细介绍了数据预处理、特征工程、模型设计、实验调优历程及分析过程。

**关键词**: 风速预测、时间序列、深度学习、LSTM、Transformer、WaveNet、学习率调优

---

## 1. 实验背景与目的

### 1.1 研究背景

风速预测是气象学和能源领域的重要研究课题，在风力发电、航空调度、建筑设计等领域具有广泛的应用价值。准确的风速预测可以帮助电网运营商优化风力发电的调度计划，提高可再生能源的利用效率，降低电力系统的运营成本[^1]。

传统的风速预测方法主要包括物理方法和统计方法。物理方法基于数值天气预报模型（NWP），通过求解大气动力学方程来预测风速，但计算成本高且对初始条件敏感[^2]。统计方法如自回归移动平均模型（ARIMA）和指数平滑法简单高效，但难以捕捉风速数据的非线性特征[^3]。

近年来，深度学习方法在时间序列预测领域取得了显著进展。循环神经网络（RNN）及其变体长短期记忆网络（LSTM）能够有效建模时序数据中的长期依赖关系[^4]。Transformer架构通过自注意力机制实现了并行计算，在自然语言处理和时间序列预测中展现出优异性能[^5]。此外，时序卷积网络（TCN）和WaveNet等基于卷积的模型也在序列建模任务中表现突出[^6][^7]。

值得注意的是，深度学习模型的训练效果往往受到超参数设置的重要影响，其中**学习率**是最关键的超参数之一[^8]。研究表明，在时间序列预测任务中，不同预测步长往往需要不同的学习率策略——多步预测任务通常需要比单步预测更谨慎的学习率设置[^9]。此外，现代深度学习训练越来越依赖于高性能GPU的支持，如NVIDIA A100等高端GPU相比消费级GPU可以提供数倍的训练加速[^10]。

### 1.2 实验目的

本实验的主要目的包括：

1. **基础任务完成（75%）**：
   - 根据风向、温度、气压、湿度等变量预测风速
   - 实现单步预测（8小时历史→1小时未来）
   - 实现多步预测（8小时历史→16小时未来）
   - 按7:2:1比例划分训练集、验证集、测试集
   - 实施特征工程，处理缺失值和异常值
   - 对比至少3个模型（Linear、LSTM、Transformer）的性能
   - 使用MSE、RMSE、MAE、R²评估模型
   - 可视化数据集及预测结果
   - 保存训练完成的模型为.pth格式

2. **创新任务扩展（25%）**：
   - 设计并实现多种创新模型架构
   - 探索多高度风速数据的空间关联
   - 分析不同模型在单步/多步预测任务上的表现差异
   - 探索学习率等超参数对模型性能的影响

### 1.3 实验环境

本实验经历了两个阶段的计算环境：

**第一阶段（本地训练）**：

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 10 |
| GPU | NVIDIA RTX 3060 (12GB) |
| 深度学习框架 | PyTorch 2.0+ |
| 训练特点 | 单模型训练时间约2-3小时 |

**第二阶段（云端训练）**：

| 项目 | 配置 |
|------|------|
| 操作系统 | Linux |
| GPU | NVIDIA A100-SXM4-40GB |
| 云服务平台 | 智星云 (https://gpu.ai-galaxy.cn/store) |
| 计费标准 | ¥2.19/小时 |
| 训练特点 | 单模型训练时间约10-15分钟 |

从RTX 3060切换到A100后，训练速度提升了约10-15倍，这使得我们能够进行更充分的超参数探索和更多轮次的训练，最终多个模型的训练epoch数达到1000+。

---

## 2. 数据集介绍与拼接逻辑

### 2.1 数据集来源

本实验使用来自HuggingFace的风速数据集，包含三个不同高度的传感器数据：

| 数据集 | 来源链接 | 高度 |
|--------|----------|------|
| WindSpeed_10m | https://huggingface.co/datasets/Antajitters/WindSpeed_10m | 10米 |
| WindSpeed_50m | https://huggingface.co/datasets/Antajitters/WindSpeed_50m | 50米 |
| WindSpeed_100m | https://huggingface.co/datasets/Antajitters/WindSpeed_100m | 100米 |

### 2.2 数据集结构

每个高度的数据集包含以下字段：

| 字段名 | 说明 | 单位 |
|--------|------|------|
| Date & Time Stamp | 时间戳 | datetime |
| SpeedAvg | 平均风速 | m/s |
| SpeedMax | 最大风速 | m/s |
| DirectionAvg | 平均风向 | 度 |
| TemperatureAvg | 平均温度 | °C |
| PressureAvg | 平均气压 | hPa |
| HumidityAvg | 平均湿度 | % |
| height | 传感器高度 | m |

### 2.3 数据集统计

- **数据条数**: 每个高度10,573条记录
- **合并后总条数**: 31,719条
- **时间分辨率**: 1小时
- **数据大小**: 约257KB

### 2.4 数据拼接逻辑

数据拼接的核心思想是将三个高度的数据按时间戳对齐，形成一个包含所有高度信息的宽表。具体步骤如下：

```
步骤1: 加载各高度数据
├── 加载10m高度数据 (10,573条)
├── 加载50m高度数据 (10,573条)
└── 加载100m高度数据 (10,573条)

步骤2: 添加高度标识
├── 为每条记录添加height列
└── 统一数据格式

步骤3: 垂直合并
├── 使用pd.concat合并三个数据集
└── 合并后共31,719条记录

步骤4: 时间戳对齐
├── 解析时间戳为datetime格式
└── 按时间戳排序

步骤5: 数据透视(Pivot)
├── 以时间戳为索引
├── 以高度为列
└── 将SpeedAvg、DirectionAvg等变量展开
    生成: SpeedAvg_10m, SpeedAvg_50m, SpeedAvg_100m
          DirectionAvg_10m, DirectionAvg_50m, ...

步骤6: 结果验证
└── 最终得到10,573条记录，每条包含所有高度的信息
```

**透视后的数据结构**：

| Date & Time Stamp | SpeedAvg_10m | SpeedAvg_50m | SpeedAvg_100m | DirectionAvg_10m | ... |
|-------------------|--------------|--------------|---------------|------------------|-----|
| 2023-01-01 00:00 | 5.2 | 7.1 | 8.5 | 180 | ... |
| 2023-01-01 01:00 | 4.8 | 6.9 | 8.2 | 175 | ... |

这种拼接方式的优势在于：
1. 保留了时间序列的连续性
2. 可以利用不同高度风速的物理关联（风切变效应）[^11]
3. 便于后续的滑动窗口采样

---

## 3. 数据预处理与特征工程

### 3.1 缺失值处理

数据集中存在少量缺失值，主要集中在10米高度的风速数据。

**处理方法**: 采用线性插值法（Linear Interpolation）填充缺失值[^12]：

```python
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
```

线性插值法的优点：
- 保持数据的连续性和趋势
- 对时间序列数据特别适用
- 不会引入突变点

对于边界处无法插值的缺失值，使用前向填充（ffill）和后向填充（bfill）作为补充。

### 3.2 异常值处理

采用四分位距（IQR）方法检测和处理异常值[^13]。IQR方法的原理是：

- Q1: 第一四分位数（25%）
- Q3: 第三四分位数（75%）
- IQR = Q3 - Q1
- 下界 = Q1 - k × IQR
- 上界 = Q3 + k × IQR

本实验使用k=3.0的阈值，异常值统计如下：

| 字段 | 异常值数量 |
|------|------------|
| SpeedAvg | 25 |
| SpeedMax | 9 |
| DirectionAvg | 318 |
| Speed Avg 10m | 14 |

**处理策略**: 使用边界值替换异常值，而非直接删除，以保持时间序列的完整性。

### 3.3 特征工程

#### 3.3.1 原始气象特征

从三个高度的传感器数据中提取以下特征（共12个）：

| 类别 | 特征 | 数量 |
|------|------|------|
| 风向 | DirectionAvg_10m, DirectionAvg_50m, DirectionAvg_100m | 3 |
| 温度 | TemperatureAvg_10m, TemperatureAvg_50m, TemperatureAvg_100m | 3 |
| 气压 | PressureAvg_10m, PressureAvg_50m, PressureAvg_100m | 3 |
| 湿度 | HumidityAvg_10m, HumidityAvg_50m, HumidityAvg_100m | 3 |

#### 3.3.2 时间特征

为了捕捉风速的周期性变化规律，构造了以下时间特征[^14]：

| 特征 | 公式 | 说明 |
|------|------|------|
| hour_sin | sin(2π × hour / 24) | 小时周期正弦分量 |
| hour_cos | cos(2π × hour / 24) | 小时周期余弦分量 |
| day_sin | sin(2π × day_of_week / 7) | 星期周期正弦分量 |
| day_cos | cos(2π × day_of_week / 7) | 星期周期余弦分量 |
| month_sin | sin(2π × month / 12) | 月份周期正弦分量 |
| month_cos | cos(2π × month / 12) | 月份周期余弦分量 |

使用正弦/余弦编码的优势：
- 保持周期性特征的连续性（23:00与00:00相邻）
- 避免离散编码带来的特征数量膨胀
- 便于神经网络学习周期性模式

#### 3.3.3 历史风速特征

将三个高度的历史风速也作为输入特征，共3个：
- SpeedAvg_10m
- SpeedAvg_50m
- SpeedAvg_100m

#### 3.3.4 特征汇总

| 特征类别 | 数量 |
|----------|------|
| 气象特征（4类×3高度） | 12 |
| 时间特征 | 6 |
| 历史风速 | 3 |
| **总计** | **21** |

### 3.4 数据标准化

采用Z-score标准化（StandardScaler），将特征缩放到均值为0、标准差为1的分布[^15]：

$$z = \frac{x - \mu}{\sigma}$$

其中：
- $x$: 原始值
- $\mu$: 训练集均值
- $\sigma$: 训练集标准差
- $z$: 标准化后的值

**重要**: 使用训练集的均值和标准差对验证集和测试集进行标准化，防止数据泄露。

### 3.5 数据集划分

按时间顺序将数据划分为三部分：

| 数据集 | 比例 | 样本数 | 用途 |
|--------|------|--------|------|
| 训练集 | 70% | 7,401 | 模型训练 |
| 验证集 | 20% | 2,114 | 超参数调优、早停 |
| 测试集 | 10% | 1,058 | 最终性能评估 |

**时序划分**: 按时间顺序划分，确保训练集在验证集之前，验证集在测试集之前，符合时间序列预测的实际场景[^16]。

---

## 4. 模型原理与设计

### 4.1 基础模型

#### 4.1.1 Linear模型（多层感知机）

Linear模型是基于多层感知机（MLP）的基线模型，将输入序列展平后通过全连接层预测输出[^17]。

**模型结构**:
```
输入层: (batch, input_len × num_features)
    ↓
隐藏层1: Linear(input_dim, 128) + ReLU + Dropout(0.2)
    ↓
隐藏层2: Linear(128, 64) + ReLU + Dropout(0.2)
    ↓
隐藏层3: Linear(64, 32) + ReLU + Dropout(0.2)
    ↓
输出层: Linear(32, output_len × num_targets)
    ↓
输出: (batch, output_len, num_targets)
```

**参数量**: 约34,000

#### 4.1.2 LSTM模型

长短期记忆网络（LSTM）[^4]是一种特殊的RNN结构，通过门控机制解决了传统RNN的梯度消失问题，能够有效捕捉时序数据中的长期依赖关系。

**LSTM单元结构**:

LSTM单元包含三个门：
- **遗忘门**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- **输入门**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- **输出门**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

细胞状态更新:
- $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- $h_t = o_t \odot \tanh(C_t)$

**本实验配置**:
- 隐藏层维度: 256
- 层数: 3
- 双向: 是
- Dropout: 0.3

**参数量**: 约6,000,000

#### 4.1.3 Transformer模型

Transformer[^5]基于自注意力机制，能够并行处理序列数据并捕捉长距离依赖关系。

**自注意力机制**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- Q: 查询矩阵
- K: 键矩阵
- V: 值矩阵
- $d_k$: 键向量维度

**多头注意力**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个head为:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**本实验配置**:
- 模型维度: 128
- 注意力头数: 8
- 编码器层数: 3
- 解码器层数: 3
- 前馈网络维度: 512
- Dropout: 0.2

**参数量**: 约3,000,000

### 4.2 创新模型

#### 4.2.1 WaveNet模型（多步预测最佳）

WaveNet[^7]是一种基于膨胀因果卷积的生成模型，最初由DeepMind提出用于音频合成，后被广泛应用于时间序列预测。

**核心创新**:

1. **膨胀因果卷积**: 通过指数增长的膨胀率扩大感受野
   - 第1层: dilation=1, 感受野=3
   - 第2层: dilation=2, 感受野=7
   - 第k层: dilation=2^(k-1), 感受野=2^k+1

2. **门控激活单元**:
   $$z = \tanh(W_f * x) \odot \sigma(W_g * x)$$
   其中$\odot$表示逐元素相乘，$*$表示卷积操作

3. **残差连接 + Skip连接**:
   - 残差连接帮助梯度流动
   - Skip连接聚合多层特征

**参数量**: 约350,000

#### 4.2.2 HeightAttention模型（多步预测亚军）

HeightAttention是本实验专门针对多高度风速预测设计的创新模型。

**核心创新**:

1. **多高度注意力机制**: 
   - 将扁平化特征重组为(高度, 特征)结构
   - 使用多头注意力学习不同高度之间的依赖关系

2. **物理约束**:
   - 嵌入风廓线幂律公式: $V(z) = V_{ref} \times (z/z_{ref})^\alpha$[^11]
   - 学习风切变指数$\alpha$

**参数量**: 约200,000

#### 4.2.3 TCN模型（时序卷积网络）

TCN[^6]是一种因果卷积网络，通过膨胀卷积扩大感受野，同时保持因果性。

**核心特点**:
- 因果卷积: 确保预测只使用过去信息
- 膨胀卷积: 指数级扩大感受野
- 残差连接: 支持深层网络训练[^18]

**参数量**: 约100,000

#### 4.2.4 DLinear模型

DLinear[^19]来自AAAI 2023论文，证明简单的线性模型在时间序列预测中也能取得优异性能，这一发现对于时间序列预测领域具有重要意义——复杂并不总是意味着更好。

**核心创新**:
- 趋势-季节分解: 将时间序列分解为趋势和季节性分量
- 独立线性层: 分别预测两个分量
- 极少参数: 仅约100个参数

**参数量**: 约100

#### 4.2.5 其他创新模型

| 模型 | 核心创新 | 参数量 |
|------|----------|--------|
| CNN_LSTM | 多尺度CNN特征提取 + LSTM序列建模 | ~300K |
| LSTNet[^20] | CNN + GRU + Skip-RNN + Highway | ~150K |
| TrendLinear | Holt指数平滑[^21]启发的趋势分解 | ~100 |
| WindShear | 嵌入风切变物理约束 | ~20 |
| Persistence | 基线模型（持续性预测） | 6 |

### 4.3 训练策略

#### 4.3.1 优化器

使用Adam优化器[^22]，配置如下：
- 权重衰减: 1e-5

**学习率配置**（关键发现详见第6节）：
- 单步预测初始学习率: 0.001
- 多步预测初始学习率: 0.0001（经调优后提高至更高值）

#### 4.3.2 学习率调度

采用余弦退火重启（CosineAnnealingWarmRestarts）[^23]+ 平台检测（ReduceLROnPlateau）双调度策略：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

#### 4.3.3 早停机制

- 单步预测: 使用MSE作为早停指标，patience=20
- 多步预测: 使用R²作为早停指标，patience=40

#### 4.3.4 损失函数

使用均方误差（MSE）作为损失函数：

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

---

## 5. 实验过程与调优历程

本节详细记录了实验过程中遇到的问题、解决方案及关键发现，这些经验对于理解最终实验结果至关重要。

### 5.1 任务理解偏差与修正

**问题描述**：

实验初期，我对任务要求的理解存在偏差：
- **原始理解（错误）**：认为多步预测中还包含"8小时→1小时"的任务，与单步预测重复；同时误将多步预测设置为"24小时→16小时"
- **正确理解**：单步预测为"8小时→1小时"，多步预测为"8小时→16小时"

**影响分析**：
1. 最初多创建了一类`multistep_1h`模型，与`singlestep`任务重复
2. 错误的"24小时→16小时"配置导致多步预测效果极差，因为输入序列过长，模型难以有效学习

**修正措施**：
1. 删除了冗余的`multistep_1h`任务
2. 将多步预测配置正确修改为"8小时→16小时"

### 5.2 计算资源迁移

**本地训练瓶颈**：

初期使用本地RTX 3060 GPU进行训练，遇到以下问题：
- 创新模型（如WaveNet）单次训练需要2-3小时
- 早停耐心值设置为30，训练轮数约200-300轮时收敛
- 难以进行充分的超参数探索

**云端训练加速**：

发现智星云（https://gpu.ai-galaxy.cn/store）提供高性价比的GPU租借服务后，迁移至云端训练：

| 对比项 | RTX 3060 | A100-SXM4-40GB |
|--------|----------|----------------|
| 显存 | 12GB | 40GB |
| Batch Size | 128 | 512 |
| 单模型训练时间 | 2-3小时 | 10-15分钟 |
| 成本 | - | ¥2.19/小时 |

**性能提升**：
- 训练速度提升约10-15倍
- 可以使用更大的batch size和更高的patience
- 多个模型训练epoch数达到1000+，充分收敛

这一经验验证了研究文献中关于高性能GPU对深度学习训练重要性的论述[^10]。

### 5.3 学习率调优的关键发现

**问题现象**：

在云端训练初期，创新模型（WaveNet、HeightAttention等）的多步预测性能始终不佳，表现甚至不如简单的Linear和LSTM模型。

**初步分析（错误方向）**：

一开始我怀疑是以下原因：
1. 模型参数量过大，数据集规模（约10,000条）不足以支撑
2. 创新模型的设计可能不适用于风速预测任务

基于这一判断，我设计了几个更简单的模型（如Persistence、TrendLinear），但结果仍然显示LSTM和Linear相对较优。

**关键发现**：

经过仔细分析训练日志和反复实验，我发现问题的根源在于**学习率设置过低**：

- 多步预测任务的初始学习率为0.0001
- 部分模型的学习率在训练过程中进一步降低
- 过低的学习率导致模型"学不到东西"，验证损失几乎不下降

这一发现与深度学习文献中关于学习率重要性的研究结论一致：学习率是深度学习中最关键的超参数之一，过低的学习率会导致训练收敛极慢，甚至陷入次优解[^8][^9]。

**解决方案**：

将多步预测任务的学习率从0.0001提升后，创新模型的性能显著提升：

| 模型 | 调整前R² | 调整后R² | 提升幅度 |
|------|----------|----------|----------|
| WaveNet | ~0.35 | 0.5375 | +53.6% |
| HeightAttention | ~0.40 | 0.5332 | +33.3% |
| TCN | ~0.42 | 0.5142 | +22.4% |

**经验总结**：

1. **任务差异化学习率**：单步预测和多步预测需要不同的学习率策略
2. **诊断优先**：当模型性能不佳时，应首先检查超参数设置，而非急于修改模型架构
3. **学习率敏感性**：复杂模型往往对学习率更敏感，需要更仔细的调优

### 5.4 创新模型设计思路

**发现问题**：

在初步实验中，我发现所有模型都未能有效利用数据集中的**高度信息**——三个高度（10m、50m、100m）的数据被简单地拼接在一起，但模型并未显式建模它们之间的物理关联。

**HeightAttention模型设计**：

基于这一发现，我设计了HeightAttention模型，核心思想是：
1. 将特征按高度分组，保持空间结构
2. 使用多头注意力机制学习不同高度之间的依赖关系
3. 引入风切变物理约束（风速随高度变化的幂律）[^11]

**效果验证**：

HeightAttention在多步预测任务中取得第二名（R²=0.5332），证明了显式建模高度关联的有效性。

---

## 6. 实验结果与分析

### 6.1 评估指标

本实验使用以下四个指标评估模型性能：

| 指标 | 公式 | 说明 | 优化方向 |
|------|------|------|----------|
| MSE | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | 均方误差 | 越小越好 |
| RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | 均方根误差，与原数据同单位 | 越小越好 |
| MAE | $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ | 平均绝对误差 | 越小越好 |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数，模型解释方差比例 | 越大越好 |

**推荐**: 以RMSE作为主要评判指标（单位与风速相同，直观易解释），R²作为辅助指标（反映模型拟合优度）。

### 6.2 单步预测结果（8小时→1小时）

| 排名 | 模型 | MSE | RMSE (m/s) | MAE (m/s) | R² | 类型 |
|------|------|-----|------|-----|-----|------|
| 🥇1 | **LSTM** | 0.8380 | 0.9154 | 0.6951 | 0.8870 | 基础 |
| 🥈2 | DLinear | 0.8394 | 0.9162 | 0.6803 | 0.8868 | 创新 |
| 🥉3 | TCN | 0.8397 | 0.9163 | 0.6888 | 0.8867 | 创新 |
| 4 | HeightAttention | 0.8567 | 0.9256 | 0.7011 | 0.8844 | 创新 |
| 5 | Persistence | 0.8586 | 0.9266 | 0.6869 | 0.8842 | 基线 |
| 6 | Linear | 0.8588 | 0.9267 | 0.7060 | 0.8841 | 基础 |
| 7 | TrendLinear | 0.8711 | 0.9333 | 0.6969 | 0.8825 | 创新 |
| 8 | LSTNet | 0.8770 | 0.9365 | 0.7150 | 0.8817 | 创新 |
| 9 | WindShear | 0.8897 | 0.9432 | 0.7041 | 0.8800 | 创新 |
| 10 | CNN_LSTM | 0.8912 | 0.9440 | 0.7221 | 0.8798 | 创新 |
| 11 | Transformer | 0.8914 | 0.9441 | 0.7077 | 0.8798 | 基础 |
| 12 | WaveNet | 0.9080 | 0.9529 | 0.7220 | 0.8775 | 创新 |

**分析**:

1. **LSTM表现最佳**: 双向LSTM能够同时利用过去和未来的上下文信息，门控机制有效捕捉时序依赖[^4]，在单步预测中以R²=0.8870取得最佳性能。

2. **前三名差距极小**: LSTM、DLinear、TCN的RMSE差距仅0.001 m/s，表明短期预测任务相对简单，多种模型都能达到较好效果。这一现象与DLinear论文[^19]的发现一致：对于短期预测，简单模型往往能取得与复杂模型相当的性能。

3. **Persistence基线很强**: 简单的持续性预测（使用最后一个时刻的值作为预测）达到R²=0.8842，说明风速具有较强的自相关性。这也是时间序列预测中的一个重要基准[^24]。

4. **Transformer表现不佳**: 可能由于数据量有限（约10,000条），Transformer更适合大规模数据集。研究表明，Transformer在小数据集上容易过拟合[^25]。

### 6.3 多步预测结果（8小时→16小时）

| 排名 | 模型 | MSE | RMSE (m/s) | MAE (m/s) | R² | 类型 |
|------|------|-----|------|-----|-----|------|
| 🥇1 | **WaveNet** | 3.4429 | 1.8555 | 1.4492 | 0.5375 | 创新 |
| 🥈2 | HeightAttention | 3.4749 | 1.8641 | 1.4572 | 0.5332 | 创新 |
| 🥉3 | LSTM | 3.5543 | 1.8853 | 1.4937 | 0.5225 | 基础 |
| 4 | Linear | 3.5826 | 1.8928 | 1.4777 | 0.5187 | 基础 |
| 5 | TCN | 3.6165 | 1.9017 | 1.4976 | 0.5142 | 创新 |
| 6 | DLinear | 3.7092 | 1.9259 | 1.5057 | 0.5017 | 创新 |
| 7 | LSTNet | 3.7472 | 1.9358 | 1.5059 | 0.4966 | 创新 |
| 8 | CNN_LSTM | 3.7937 | 1.9477 | 1.5453 | 0.4904 | 创新 |
| 9 | WindShear | 3.8610 | 1.9650 | 1.4912 | 0.4813 | 创新 |
| 10 | Persistence | 3.8945 | 1.9734 | 1.4907 | 0.4768 | 基线 |
| 11 | TrendLinear | 4.0309 | 2.0077 | 1.5950 | 0.4585 | 创新 |
| 12 | Transformer | 4.0596 | 2.0148 | 1.5991 | 0.4546 | 基础 |

**分析**:

1. **WaveNet取得最佳性能**: 门控卷积+膨胀因果卷积在长期序列建模上展现出优势[^7]，R²=0.5375显著优于基础模型。WaveNet的膨胀卷积设计使其能够高效地捕获长距离依赖，这对于多步预测至关重要。

2. **创新模型超越基础模型**: 前两名都是创新模型，HeightAttention的多高度注意力机制有效捕获了不同高度风速的关联，验证了显式建模空间关系的价值。

3. **多步预测难度大**: 所有模型的R²从0.88（单步）下降到0.53（多步），RMSE从0.91 m/s增加到1.86 m/s。这反映了长期预测的固有困难——随着预测步长增加，误差会逐步累积，这是时间序列预测领域公认的挑战[^26]。

4. **Persistence基线被超越**: 多步预测中，复杂模型（如WaveNet）明显优于持续性基线，说明模型学到了有价值的时序模式，而非简单记忆。

### 6.4 创新模型价值分析

| 创新模型 | 单步排名 | 多步排名 | 核心优势 | 适用场景 |
|----------|----------|----------|----------|----------|
| WaveNet | 12 | **1** | 门控卷积，长序列建模 | 多步预测 |
| HeightAttention | 4 | **2** | 多高度注意力机制 | 多高度数据 |
| DLinear | **2** | 6 | 简单高效，分解线性 | 单步预测 |
| TCN | **3** | 5 | 因果卷积，残差连接 | 通用 |
| LSTNet | 8 | 7 | 周期性建模 | 周期性数据 |
| CNN_LSTM | 10 | 8 | 多尺度特征 | 多尺度模式 |

**关键发现**:
- 模型在不同任务上的表现差异显著
- 单步预测优选: LSTM、DLinear、TCN
- 多步预测优选: WaveNet、HeightAttention
- 模型架构需要与任务特性相匹配

### 6.5 训练过程分析

各模型训练收敛情况：

| 模型 | 任务 | 总Epoch数 | 最佳Epoch | 最佳验证MSE |
|------|------|-----------|-----------|-------------|
| LSTM | singlestep | 370+ | 283 | 0.0698 |
| LSTM | multistep | 415+ | 32 | 0.3365 |
| WaveNet | singlestep | 1341+ | 1053 | 0.0701 |
| WaveNet | multistep | 1267+ | 1267 | 0.3285 |
| HeightAttention | singlestep | 1493+ | 1493 | 0.0683 |
| HeightAttention | multistep | 1348+ | 1211 | 0.3098 |
| DLinear | singlestep | 1380+ | 1201 | 0.0640 |
| Transformer | singlestep | 500+ | 500 | 0.1525 |

**观察**:
- 创新模型（如WaveNet、TCN、HeightAttention）需要更多训练轮数才能充分收敛，这得益于A100 GPU的高速训练能力
- 多步预测任务的验证损失明显高于单步预测
- LSTM在较少轮数内即可达到最佳性能，体现了其对中小规模数据的适应性[^4]
- 充分的训练轮数（1000+）对于发挥创新模型的潜力至关重要

---

## 7. 结论

### 7.1 主要结论

1. **基础任务完成情况**: 
   - ✅ 成功实现了基于风向、温度、气压、湿度的风速预测
   - ✅ 完成了单步预测（8h→1h）和多步预测（8h→16h）
   - ✅ 数据集按7:2:1划分
   - ✅ 实施了完整的特征工程流程
   - ✅ 对比了12个模型的性能
   - ✅ 保存了24个.pth模型文件

2. **最佳模型推荐**:
   | 任务 | 推荐模型 | RMSE | R² |
   |------|----------|------|-----|
   | 单步预测 | LSTM | 0.9154 m/s | 0.8870 |
   | 多步预测 | WaveNet | 1.8555 m/s | 0.5375 |

3. **创新价值**: 
   - 9种创新模型中，WaveNet和HeightAttention在多步预测中显著超越基础模型
   - DLinear证明了简单模型在短期预测中的有效性
   - 多高度注意力机制有效利用了风速的空间关联

4. **关键经验教训**:
   - 学习率是影响模型性能的关键超参数，不同任务需要差异化设置
   - 高性能GPU（如A100）可显著加速训练，支持更充分的超参数探索
   - 在修改模型架构前，应首先确保超参数配置正确

### 7.2 局限性与未来工作

1. **数据规模**: 约10,000条样本对于Transformer等复杂模型可能不足
2. **外部特征**: 未引入天气预报数据等外部信息
3. **集成学习**: 可尝试模型集成进一步提升性能
4. **不确定性量化**: 可添加概率预测输出置信区间[^27]
5. **更长预测步长**: 可探索更长时间尺度的预测任务

---

## 8. 参考文献

[^1]: Wang, J., Song, Y., Liu, F., & Hou, R. (2016). Analysis and application of forecasting models in wind power integration: A review of multi-step-ahead wind speed forecasting models. *Renewable and Sustainable Energy Reviews*, 60, 960-981. https://doi.org/10.1016/j.rser.2016.01.114

[^2]: Soman, S. S., Zareipour, H., Malik, O., & Mandal, P. (2010). A review of wind power and wind speed forecasting methods with different time horizons. In *North American Power Symposium (NAPS), 2010* (pp. 1-8). IEEE. https://doi.org/10.1109/NAPS.2010.5619586

[^3]: Erdem, E., & Shi, J. (2011). ARMA based approaches for forecasting the tuple of wind speed and direction. *Applied Energy*, 88(4), 1405-1414. https://doi.org/10.1016/j.apenergy.2010.10.031

[^4]: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

[^5]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

[^6]: Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*. https://arxiv.org/abs/1803.01271

[^7]: Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*. https://arxiv.org/abs/1609.03499

[^8]: Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures. In *Neural networks: Tricks of the trade* (pp. 437-478). Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-35289-8_26

[^9]: Smith, L. N. (2017). Cyclical learning rates for training neural networks. In *2017 IEEE Winter Conference on Applications of Computer Vision (WACV)* (pp. 464-472). IEEE. https://arxiv.org/abs/1506.01186

[^10]: Mattson, P., et al. (2020). MLPerf training benchmark. *Proceedings of Machine Learning and Systems*, 2, 336-349. https://arxiv.org/abs/1910.01500

[^11]: Peterson, E. W., & Hennessey Jr, J. P. (1978). On the use of power laws for estimates of wind power potential. *Journal of Applied Meteorology and Climatology*, 17(3), 390-394. https://doi.org/10.1175/1520-0450(1978)017<0390:OTUOPL>2.0.CO;2

[^12]: Lepot, M., Aubin, J. B., & Clemens, F. H. (2017). Interpolation in time series: An introductive overview of existing methods, their performance criteria and uncertainty assessment. *Water*, 9(10), 796. https://doi.org/10.3390/w9100796

[^13]: Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley. ISBN: 978-0201076165

[^14]: Brownlee, J. (2017). How to Use Cyclic Features for Time Series Prediction. *Machine Learning Mastery*. https://machinelearningmastery.com/time-series-forecasting/

[^15]: Gron, A. (2017). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media. ISBN: 978-1492032649

[^16]: Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*, 120, 70-83. https://doi.org/10.1016/j.csda.2017.11.003

[^17]: Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366. https://doi.org/10.1016/0893-6080(89)90020-8

[^18]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770-778). https://arxiv.org/abs/1512.03385

[^19]: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128. https://arxiv.org/abs/2205.13504

[^20]: Lai, G., Chang, W. C., Yang, Y., & Liu, H. (2018). Modeling long-and short-term temporal patterns with deep neural networks. In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval* (pp. 95-104). https://arxiv.org/abs/1703.07015

[^21]: Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

[^22]: Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

[^23]: Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *Proceedings of the 5th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1608.03983

[^24]: Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

[^25]: Liu, Y., et al. (2021). A survey of visual transformers. *IEEE Transactions on Neural Networks and Learning Systems*. https://arxiv.org/abs/2111.06091

[^26]: Taieb, S. B., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition. *Expert Systems with Applications*, 39(8), 7067-7083. https://doi.org/10.1016/j.eswa.2012.01.039

[^27]: Gneiting, T., & Katzfuss, M. (2014). Probabilistic forecasting. *Annual Review of Statistics and Its Application*, 1, 125-151. https://doi.org/10.1146/annurev-statistics-062713-085831

---

## 附录A: 可视化结果

本实验生成了以下可视化图表：

| 文件名 | 说明 |
|--------|------|
| dataset_overview.png | 数据集概览（时序图、分布图、相关性热图） |
| model_performance_ranking.png | 模型性能排名（RMSE和R²） |
| model_comparison_by_type.png | 按模型类型分组对比 |
| basic_models_radar.png | 基础模型雷达图对比 |
| single_vs_multi_comparison.png | 单步vs多步预测对比 |
| all_metrics_heatmap.png | 所有指标热力图 |
| best_models_summary.png | 最佳模型TOP3总结 |
| improvement_analysis.png | 相对基线改进分析 |
| comparison_*.png | 各指标对比图 |
| results_summary_table.png | 结果汇总表格 |

## 附录B: 模型文件清单

**基础模型（6个）**:
- Linear_singlestep.pth / Linear_multistep_16h.pth
- LSTM_singlestep.pth / LSTM_multistep_16h.pth
- Transformer_singlestep.pth / Transformer_multistep_16h.pth

**创新模型（18个）**:
- CNN_LSTM_singlestep.pth / CNN_LSTM_multistep_16h.pth
- TCN_singlestep.pth / TCN_multistep_16h.pth
- WaveNet_singlestep.pth / WaveNet_multistep_16h.pth
- LSTNet_singlestep.pth / LSTNet_multistep_16h.pth
- DLinear_singlestep.pth / DLinear_multistep_16h.pth
- HeightAttention_singlestep.pth / HeightAttention_multistep_16h.pth
- TrendLinear_singlestep.pth / TrendLinear_multistep_16h.pth
- WindShear_singlestep.pth / WindShear_multistep_16h.pth
- Persistence_singlestep.pth / Persistence_multistep_16h.pth

## 附录C: 实验心得与反思

### C.1 任务理解的重要性

本次实验中，最初对任务要求的理解偏差导致了不少无用功。这提醒我们在开始任何项目之前，务必仔细阅读和理解需求，必要时与指导老师确认。

### C.2 调试思路的反思

当创新模型表现不佳时，我一开始怀疑是模型设计问题，后来才发现是学习率设置过低。这个教训表明，在深度学习实验中，超参数问题往往比模型架构问题更常见，应该遵循"先调参后改模型"的原则。

### C.3 计算资源的价值

从RTX 3060迁移到A100后，训练效率提升了10-15倍。这使得我们能够进行更多的实验、更充分地训练模型。这一经历让我深刻体会到，在深度学习研究中，计算资源是一项重要的"生产力"。

---

*报告生成时间: 2025年12月30日*  
*实验项目: 风速序列预测*
