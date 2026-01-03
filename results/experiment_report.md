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

本章详细介绍实验中使用的12个模型的设计原理、网络架构和关键实现细节。所有模型均基于PyTorch框架实现，输入为8个时间步的21维特征向量，输出为1个（单步）或16个（多步）时间步的3维风速预测（对应10m、50m、100m三个高度）。

### 4.1 基础模型

#### 4.1.1 Linear模型（多层感知机）

Linear模型是基于多层感知机（MLP）的基线模型，其核心思想是将时序预测问题转化为标准的回归问题[^17]。MLP是最基础的前馈神经网络，由输入层、隐藏层和输出层组成，每层神经元与下一层全连接。

**设计动机**：
- 作为基线模型，验证深度学习方法相对于简单方法的提升
- MLP的通用近似定理保证其可以拟合任意复杂的非线性函数
- 结构简单，便于调试和理解

**数学原理**：

对于输入$\mathbf{x} \in \mathbb{R}^{d_{in}}$，每一层的前向传播为：
$$\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$$

其中$\mathbf{W}^{(l)}$为权重矩阵，$\mathbf{b}^{(l)}$为偏置向量，$\sigma$为激活函数（本实验使用ReLU）。

**批归一化（BatchNorm）**：
本实验在每个隐藏层添加BatchNorm层，其计算过程为：
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

其中$\mu_B$和$\sigma_B^2$是mini-batch的均值和方差，$\gamma$和$\beta$是可学习的缩放和偏移参数。BatchNorm的优点包括：
- 加速训练收敛
- 允许使用更大的学习率
- 具有轻微的正则化效果

**模型结构**:
```
输入: (batch_size, 8, 21) — 8个时间步，每步21维特征
    ↓ Flatten展平
中间表示: (batch_size, 168) — 将时序展平为一维向量
    ↓
隐藏层1: Linear(168, 128) → BatchNorm1d(128) → ReLU → Dropout(0.2)
    ↓
隐藏层2: Linear(128, 64) → BatchNorm1d(64) → ReLU → Dropout(0.2)
    ↓
隐藏层3: Linear(64, 32) → BatchNorm1d(32) → ReLU → Dropout(0.2)
    ↓
输出层: Linear(32, output_len × 3)
    ↓ Reshape重塑
输出: (batch_size, output_len, 3) — 预测各高度风速
```

**Dropout正则化**：
训练时以概率$p=0.2$随机将神经元输出置零，防止过拟合：
$$\mathbf{h}_{dropout} = \mathbf{h} \odot \mathbf{m}, \quad m_i \sim \text{Bernoulli}(1-p)$$

**参数量计算**：
- 隐藏层1: $168 \times 128 + 128 + 2 \times 128 = 21,888$
- 隐藏层2: $128 \times 64 + 64 + 2 \times 64 = 8,384$
- 隐藏层3: $64 \times 32 + 32 + 2 \times 32 = 2,144$
- 输出层（单步）: $32 \times 3 + 3 = 99$
- **总计（单步）**: 约32,500；**（多步）**: 约34,000

#### 4.1.2 LSTM模型

长短期记忆网络（LSTM）[^4]是一种特殊的循环神经网络（RNN）结构，由Hochreiter和Schmidhuber于1997年提出。LSTM通过精心设计的门控机制解决了传统RNN的梯度消失/爆炸问题，能够有效捕捉时序数据中的长期依赖关系。

**设计动机**：
- 风速预测具有明显的时序依赖性，前一时刻的风速对后续时刻有重要影响
- LSTM能够选择性地记忆和遗忘信息，适合处理变长的时序依赖
- 双向LSTM可以同时利用过去和未来的上下文信息

**LSTM单元核心原理**：

LSTM引入了**细胞状态（Cell State）** $C_t$作为信息的载体，通过三个门控制信息的流动：

1. **遗忘门（Forget Gate）** — 决定丢弃哪些旧信息：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **输入门（Input Gate）** — 决定存储哪些新信息：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

3. **输出门（Output Gate）** — 决定输出哪些信息：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**细胞状态更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**隐藏状态更新**：
$$h_t = o_t \odot \tanh(C_t)$$

其中$\sigma$为Sigmoid函数，$\odot$表示逐元素乘法（Hadamard积）。

**双向LSTM（Bi-LSTM）**：

本实验采用双向LSTM，同时从前向和后向处理序列：
- **前向LSTM**：从$t=1$到$t=T$处理序列，得到$\overrightarrow{h_t}$
- **后向LSTM**：从$t=T$到$t=1$处理序列，得到$\overleftarrow{h_t}$
- **最终表示**：$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$（拼接）

双向LSTM的输出维度为单向的2倍，即$256 \times 2 = 512$。

**模型架构**：
```
输入: (batch_size, 8, 21) — 8个时间步，每步21维特征
    ↓
Bi-LSTM Layer 1: input_size=21 → hidden_size=256×2
    ↓ Dropout(0.3)
Bi-LSTM Layer 2: input_size=512 → hidden_size=256×2
    ↓ Dropout(0.3)
Bi-LSTM Layer 3: input_size=512 → hidden_size=256×2
    ↓
取最后时刻隐藏状态: 拼接前向h[-2]和后向h[-1] → (batch_size, 512)
    ↓
全连接层: Linear(512, 128) → ReLU → Dropout(0.3)
    ↓
输出层: Linear(128, output_len × 3)
    ↓ Reshape
输出: (batch_size, output_len, 3)
```

**本实验配置详解**:

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 256 | 每个方向的隐藏层维度 |
| num_layers | 3 | LSTM堆叠层数，增加模型深度 |
| bidirectional | True | 使用双向LSTM |
| dropout | 0.3 | 层间Dropout，防止过拟合 |
| batch_first | True | 输入格式为(batch, seq, feature) |

**参数量分析**：

LSTM层参数量计算公式（单向）：
$$\text{params} = 4 \times [(input\_size + hidden\_size) \times hidden\_size + hidden\_size]$$

- 第1层（双向）: $4 \times [(21+256) \times 256 + 256] \times 2 = 567,296$
- 第2层（双向）: $4 \times [(512+256) \times 256 + 256] \times 2 = 1,574,912$
- 第3层（双向）: $4 \times [(512+256) \times 256 + 256] \times 2 = 1,574,912$
- 全连接层: $512 \times 128 + 128 + 128 \times 48 + 48 = 71,856$
- **总计**: 约3,790,000参数

#### 4.1.3 Transformer模型

Transformer[^5]是Google于2017年在论文"Attention is All You Need"中提出的革命性架构，完全基于自注意力机制，摒弃了传统的循环结构。Transformer能够并行处理序列数据并捕捉长距离依赖关系，已成为现代深度学习的基石架构。

**设计动机**：
- 自注意力机制可以直接建模序列中任意两个位置的关系，无需像RNN那样逐步传递
- 并行计算效率高，训练速度快
- 多头注意力可以同时关注不同的特征子空间

**核心组件详解**：

**1. 缩放点积注意力（Scaled Dot-Product Attention）**：

给定查询矩阵$Q$、键矩阵$K$和值矩阵$V$，注意力计算为：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$: 查询矩阵，表示"我在找什么"
- $K \in \mathbb{R}^{m \times d_k}$: 键矩阵，表示"我有什么"
- $V \in \mathbb{R}^{m \times d_v}$: 值矩阵，表示"实际的内容"
- $\sqrt{d_k}$: 缩放因子，防止点积过大导致softmax梯度消失

**注意力权重的直观理解**：
$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{l=1}^{m} \exp(q_i \cdot k_l / \sqrt{d_k})}$$

$\alpha_{ij}$表示位置$i$对位置$j$的关注程度。

**2. 多头注意力（Multi-Head Attention）**：

多头注意力允许模型同时关注不同表示子空间的信息：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个注意力头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$: 输出投影矩阵
- 本实验: $h=8$头，$d_k = d_v = d_{model}/h = 128/8 = 16$

**3. 位置编码（Positional Encoding）**：

由于Transformer没有循环结构，需要显式注入位置信息。本实验使用正弦位置编码：
$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

位置编码的设计使得模型可以通过注意力学习相对位置关系。

**4. 前馈神经网络（Feed-Forward Network）**：

每个Transformer层包含一个位置级的前馈网络：
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

即两层线性变换，中间使用ReLU激活，隐藏层维度为512。

**5. 残差连接与层归一化**：

每个子层（注意力/FFN）都使用残差连接和层归一化：
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓
编码器输入嵌入: Linear(21, 128) — 将特征维度映射到模型维度
    ↓
位置编码: 添加正弦位置编码
    ↓
┌─────────────────────────────────────────────────┐
│ Transformer Encoder × 3层                        │
│  ├─ Multi-Head Self-Attention (8头)              │
│  ├─ Add & LayerNorm                              │
│  ├─ Feed-Forward Network (dim=512)              │
│  └─ Add & LayerNorm                              │
└─────────────────────────────────────────────────┘
    ↓
解码器输入: 零初始化 (batch_size, output_len, 3) → 嵌入为 (batch_size, output_len, 128)
    ↓
┌─────────────────────────────────────────────────┐
│ Transformer Decoder × 3层                        │
│  ├─ Masked Multi-Head Self-Attention (因果掩码)  │
│  ├─ Add & LayerNorm                              │
│  ├─ Multi-Head Cross-Attention (与编码器交互)    │
│  ├─ Add & LayerNorm                              │
│  ├─ Feed-Forward Network                        │
│  └─ Add & LayerNorm                              │
└─────────────────────────────────────────────────┘
    ↓
输出层: Linear(128, 3)
    ↓
输出: (batch_size, output_len, 3)
```

**因果注意力掩码（Causal Mask）**：

为防止解码器"看到未来"，使用上三角掩码：
$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

**本实验配置详解**:

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 128 | 模型隐藏维度 |
| nhead | 8 | 注意力头数 |
| num_encoder_layers | 3 | 编码器层数 |
| num_decoder_layers | 3 | 解码器层数 |
| dim_feedforward | 512 | FFN中间维度 |
| dropout | 0.2 | 各处Dropout率 |

**参数量**: 约1,390,000（主要来自多头注意力的投影矩阵和FFN层）

### 4.2 创新模型

本节介绍9种创新模型，这些模型针对风速预测任务的特点进行了专门设计或优化。

#### 4.2.1 WaveNet模型（多步预测最佳）

WaveNet[^7]是DeepMind于2016年提出的生成模型，最初用于原始音频波形生成，后被广泛应用于时间序列预测。其核心创新在于**膨胀因果卷积**，能够在保持因果性的同时指数级扩大感受野。

**设计动机**：
- 传统CNN感受野有限，难以捕捉长距离依赖
- RNN逐步处理，计算效率低
- 膨胀卷积可以用较少的层数覆盖很长的历史信息

**核心创新详解**：

**1. 因果卷积（Causal Convolution）**：

因果卷积确保时刻$t$的输出仅依赖于$t$及之前的输入，不会"看到未来"：
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

通过适当的padding和裁剪实现因果性。

**2. 膨胀卷积（Dilated Convolution）**：

膨胀卷积在卷积核元素之间插入空洞，扩大感受野：
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-d \cdot k}$$

其中$d$为膨胀率（dilation rate）。本实验采用循环膨胀策略：

| 层号 | 膨胀率 | 累计感受野 |
|------|--------|-----------|
| 1 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 4 | 8 |
| 4 | 8 | 16 |
| 5 | 1 | 17 |
| 6 | 2 | 19 |
| 7 | 4 | 23 |
| 8 | 8 | 31 |

8层即可覆盖31个时间步的感受野，远超输入序列长度8。

**3. 门控激活单元（Gated Activation Unit）**：

受LSTM门控机制启发，WaveNet使用门控激活：
$$\mathbf{z} = \tanh(W_{f} * \mathbf{x}) \odot \sigma(W_{g} * \mathbf{x})$$

其中：
- $W_f * \mathbf{x}$: 滤波器分支（filter），提取特征
- $W_g * \mathbf{x}$: 门控分支（gate），控制信息流
- $\tanh$: 将特征压缩到$[-1, 1]$
- $\sigma$: Sigmoid门控，输出$[0, 1]$
- $\odot$: 逐元素相乘

**4. 残差连接与Skip连接**：

```
输入 x ──────────────────────────────────┐
   │                                      │
   ↓                                      │
膨胀卷积 → BatchNorm → 门控激活           │
   │                                      │
   ├──→ 1×1卷积 (残差) ──────────────────→ + → 输出到下一层
   │
   └──→ 1×1卷积 (Skip) ──────────────────→ 累加到最终输出
```

- **残差连接**: 帮助梯度流动，解决深层网络退化问题
- **Skip连接**: 直接将各层信息汇聚到输出，聚合多尺度特征

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓ 转置为 (batch_size, 21, 8) — Conv1d需要(batch, channels, length)
    ↓
输入卷积: Conv1d(21, 64, kernel=1) — 通道映射
    ↓
┌─────────────────────────────────────────────────┐
│ WaveNet Block × 8                                │
│  ├─ Dilated Causal Conv1d (dilation=1,2,4,8循环) │
│  ├─ BatchNorm                                    │
│  ├─ Gated Activation (tanh ⊙ sigmoid)           │
│  ├─ Residual Conv1d(64, 64, kernel=1)           │
│  └─ Skip Conv1d(64, 64, kernel=1)               │
└─────────────────────────────────────────────────┘
    ↓
Skip连接求和 → ReLU → Conv1d(64, 64, kernel=1) → ReLU → Dropout(0.2)
    ↓ 展平为 (batch_size, 64×8)
    ↓
全连接层: Linear(512, output_len × 3)
    ↓ Reshape
输出: (batch_size, output_len, 3)
```

**参数量**: 单步预测约208,000；多步预测约231,000

**为何WaveNet在多步预测中表现最佳**：
1. 膨胀卷积的大感受野适合捕捉长期依赖
2. 门控机制有效控制信息流，避免信息过载
3. Skip连接保留了多尺度时序特征
4. 参数效率高，不易过拟合

#### 4.2.2 HeightAttention模型（多步预测亚军）

HeightAttention是本实验**原创设计**的创新模型，专门针对多高度风速预测任务。其核心思想是充分利用三个高度（10m、50m、100m）数据之间的物理关联，而非简单地将它们作为独立特征处理。

**设计动机**：
- 气象学中，不同高度的风速存在明确的物理关联（风切变效应）
- 传统模型将21维特征扁平化处理，丢失了空间结构信息
- 注意力机制可以自动学习高度之间的依赖权重

**物理背景——风切变（Wind Shear）**：

大气边界层中，风速随高度变化遵循幂律关系[^11]：
$$V(z) = V_{ref} \times \left(\frac{z}{z_{ref}}\right)^\alpha$$

其中：
- $V(z)$: 高度$z$处的风速
- $V_{ref}$: 参考高度$z_{ref}$处的风速
- $\alpha$: 风切变指数，通常在0.1~0.4之间，取决于地表粗糙度和大气稳定性

**核心创新详解**：

**1. 特征重组（Feature Reorganization）**：

将扁平化的21维特征按物理意义重组：

```
原始特征 (21维):
├─ 风向: DirectionAvg_10m[0], DirectionAvg_50m[1], DirectionAvg_100m[2]
├─ 温度: TemperatureAvg_10m[3], _50m[4], _100m[5]
├─ 气压: PressureAvg_10m[6], _50m[7], _100m[8]
├─ 湿度: HumidityAvg_10m[9], _50m[10], _100m[11]
├─ 时间: hour_sin[12], hour_cos[13], day_sin[14], day_cos[15], month_sin[16], month_cos[17]
└─ 风速: SpeedAvg_10m[18], SpeedAvg_50m[19], SpeedAvg_100m[20]

重组为:
├─ 高度特征: (3个高度, 5个特征/高度) = (3, 5)
│   ├─ 10m: [风向, 温度, 气压, 湿度, 风速]
│   ├─ 50m: [风向, 温度, 气压, 湿度, 风速]
│   └─ 100m: [风向, 温度, 气压, 湿度, 风速]
└─ 时间特征: (6维) — 作为全局上下文
```

**2. 多高度注意力机制（Height Attention）**：

使用多头注意力学习不同高度之间的依赖关系：

$$Q = H \cdot W_Q, \quad K = H \cdot W_K, \quad V = H \cdot W_V$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中$H \in \mathbb{R}^{3 \times d}$是3个高度的特征表示。

**注意力权重可视化示例**：
```
        10m   50m   100m
10m   [0.4   0.3   0.3 ]   ← 10m的预测主要依赖自身
50m   [0.25  0.5   0.25]   ← 50m受上下两层共同影响
100m  [0.2   0.3   0.5 ]   ← 100m的预测主要依赖自身和50m
```

**3. 时序建模（Temporal Modeling）**：

对每个时间步的高度注意力输出进行时序建模：
- 使用双向LSTM捕获时间依赖
- 高度特征与时间特征融合后输入LSTM

**4. 物理约束（可选）**：

可嵌入风廓线幂律公式作为归纳偏置：
$$\alpha = \sigma(\text{MLP}(h_{final})) \times 0.5$$

学习到的$\alpha$值可用于物理分析。

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓
特征重组: 
├─ 高度特征: (batch_size, 8, 3, 5) 
└─ 时间特征: (batch_size, 8, 6)
    ↓
对每个时间步 t = 1, ..., 8:
│
├─ 高度编码器: Linear(5, 64) → LayerNorm → ReLU → Dropout
│   输出: (batch_size, 3, 64)
│
├─ 多高度注意力: MultiHeadAttention(embed_dim=64, num_heads=2)
│   ├─ Q, K, V投影
│   ├─ 缩放点积注意力
│   └─ 输出: (batch_size, 3, 64)，保存注意力权重
│
├─ 时间编码器: Linear(6, 32) → ReLU → Linear(32, 64)
│   输出: (batch_size, 64)
│
└─ 特征融合: Concat([3个高度展平, 时间特征]) → (batch_size, 3×64+64=256)
    → Linear(256, 128) → LayerNorm → ReLU → Dropout
    输出: (batch_size, 128)
    ↓
堆叠8个时间步: (batch_size, 8, 128)
    ↓
双向LSTM: Bi-LSTM(input=128, hidden=64, layers=2)
    ↓
取最终隐藏状态: Concat([h_forward, h_backward]) → (batch_size, 128)
    ↓
输出层: Linear(128, 64) → ReLU → Dropout → Linear(64, output_len × 3)
    ↓ Reshape
输出: (batch_size, output_len, 3)
```

**本实验配置详解**:

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 64 | 高度特征编码维度 |
| num_heads | 2 | 注意力头数 |
| lstm_layers | 2 | LSTM层数 |
| dropout | 0.2 | 各处Dropout率 |
| use_physics_constraint | True | 是否使用物理约束 |

**参数量**: 单步预测约264,000；多步预测约267,000

**模型优势**：
1. 保留了数据的物理结构（高度维度）
2. 注意力权重可解释性强
3. 可嵌入领域知识（风切变公式）
4. 在多高度预测任务中表现优异

#### 4.2.3 TCN模型（时序卷积网络）

TCN（Temporal Convolutional Network）[^6]是Bai等人于2018年提出的时序建模架构，在多项序列建模基准测试中表现优于LSTM等循环网络。TCN结合了因果卷积、膨胀卷积和残差连接，实现了高效的长序列建模。

**设计动机**：
- 卷积操作可以并行计算，训练效率高于RNN
- 膨胀卷积可以用较少的参数覆盖长距离依赖
- 残差连接使得深层网络训练更加稳定

**核心特点详解**：

**1. 因果卷积（Causal Convolution）**：

通过单侧padding确保时刻$t$的输出仅依赖$t$及之前的输入：
```
padding_size = (kernel_size - 1) × dilation
```
卷积后裁剪右侧多余的输出，保持因果性。

**2. 膨胀卷积（Dilated Convolution）**：

本实验使用的膨胀率配置：

| 层号 | 膨胀率 | 感受野增量 |
|------|--------|-----------|
| 1 | 1 | 3 |
| 2 | 2 | 6 |
| 3 | 4 | 12 |

总感受野 = $1 + \sum_{i=1}^{L} 2 \times (K-1) \times d_i = 1 + 2 \times 2 \times (1+2+4) = 29$

**3. 残差块（Temporal Block）**：

每个残差块包含两层卷积和一个残差连接：
```
输入 x ─────────────────────────────────┐
   │                                     │
   ↓                                     │
Conv1d → BatchNorm → ReLU → Dropout      │
   ↓                                     │
Conv1d → BatchNorm → ReLU → Dropout      │
   ↓                                     │
裁剪（保持因果性）                        │
   ↓                                     │
   + ←───── 1×1 Conv（如维度不匹配）←────┘
   ↓
 ReLU
   ↓
输出
```

**4. 注意力池化（Attention Pooling）**：

本实验在TCN输出后添加了注意力机制，自适应地聚合时序特征：
$$\alpha_t = \frac{\exp(w^T h_t)}{\sum_{i=1}^{T} \exp(w^T h_i)}$$
$$\text{context} = \sum_{t=1}^{T} \alpha_t h_t$$

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓ 转置为 (batch_size, 21, 8)
    ↓
┌─────────────────────────────────────────────────┐
│ Temporal Block 1: dilation=1                     │
│  ├─ Conv1d(21, 32, kernel=3, dilation=1)        │
│  ├─ BatchNorm → ReLU → Dropout(0.3)             │
│  ├─ Conv1d(32, 32, kernel=3, dilation=1)        │
│  ├─ BatchNorm → ReLU → Dropout(0.3)             │
│  └─ Residual: Conv1d(21, 32, kernel=1)          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Temporal Block 2: dilation=2                     │
│  ├─ Conv1d(32, 64, kernel=3, dilation=2)        │
│  ├─ BatchNorm → ReLU → Dropout(0.3)             │
│  ├─ Conv1d(64, 64, kernel=3, dilation=2)        │
│  ├─ BatchNorm → ReLU → Dropout(0.3)             │
│  └─ Residual: Conv1d(32, 64, kernel=1)          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Temporal Block 3: dilation=4                     │
│  ├─ Conv1d(64, 64, kernel=3, dilation=4)        │
│  ├─ BatchNorm → ReLU → Dropout(0.3)             │
│  ├─ Conv1d(64, 64, kernel=3, dilation=4)        │
│  ├─ BatchNorm → ReLU → Dropout(0.3)             │
│  └─ Residual: Identity                          │
└─────────────────────────────────────────────────┘
    ↓ 转置为 (batch_size, seq_len, 64)
    ↓
注意力池化:
├─ Linear(64, 32) → Tanh → Linear(32, 1) — 计算注意力分数
├─ Softmax — 归一化权重
└─ 加权求和 — 得到上下文向量 (batch_size, 64)
    ↓
全连接层: Linear(64, 128) → ReLU → Dropout(0.3)
    ↓
输出层: Linear(128, output_len × 3)
    ↓ Reshape
输出: (batch_size, output_len, 3)
```

**本实验配置详解**:

| 参数 | 值 | 说明 |
|------|-----|------|
| num_channels | [32, 64, 64] | 各层通道数 |
| kernel_size | 3 | 卷积核大小 |
| dropout | 0.3 | Dropout率 |

**参数量**: 单步预测约63,000；多步预测约68,500

**TCN vs LSTM对比**：

| 特性 | TCN | LSTM |
|------|-----|------|
| 并行性 | 高（卷积可并行） | 低（逐步计算） |
| 感受野 | 固定（由网络深度决定） | 理论上无限 |
| 梯度流 | 稳定（残差连接） | 可能消失/爆炸 |
| 参数效率 | 高 | 中等 |

#### 4.2.4 DLinear模型

DLinear[^19]来自AAAI 2023论文"Are Transformers Effective for Time Series Forecasting?"。这篇论文的核心发现是：**对于时间序列预测，简单的线性模型可以达到甚至超越复杂Transformer模型的性能**。这一发现对深度学习时序预测领域具有重要的启示意义。

**设计动机**：
- Transformer在时序预测中可能存在过度参数化问题
- 时间序列的核心是趋势和周期性，线性模型可以有效捕捉
- 简单模型不易过拟合，泛化性更好

**核心创新详解**：

**1. 序列分解（Series Decomposition）**：

将输入时间序列分解为**趋势分量**和**季节性分量**：

**移动平均提取趋势**：
$$\text{Trend}_t = \frac{1}{k}\sum_{i=-(k-1)/2}^{(k-1)/2} x_{t+i}$$

其中$k=3$为移动平均窗口大小。

**季节性分量**：
$$\text{Seasonal}_t = x_t - \text{Trend}_t$$

这种分解方法称为**加法分解**，假设原序列 = 趋势 + 季节性。

**2. 独立线性预测**：

对趋势和季节性分量分别使用独立的线性层进行预测：

$$\hat{y}_{trend} = W_{trend} \cdot \text{Trend} + b_{trend}$$
$$\hat{y}_{seasonal} = W_{seasonal} \cdot \text{Seasonal} + b_{seasonal}$$

其中$W \in \mathbb{R}^{\text{output\_len} \times \text{input\_len}}$，直接将8个时间步映射到1/16个时间步。

**3. 特征投影**：

由于输入是21维特征，输出是3维风速，需要一个投影层：
$$\hat{y} = (\hat{y}_{trend} + \hat{y}_{seasonal}) \cdot W_{proj}$$

其中$W_{proj} \in \mathbb{R}^{21 \times 3}$。

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓
┌─────────────────────────────────────────────────┐
│ 序列分解 (Series Decomposition)                  │
│  ├─ 移动平均 (kernel_size=3)                    │
│  │   ├─ 前端填充: 复制第一个值                   │
│  │   ├─ 后端填充: 复制最后一个值                 │
│  │   └─ AvgPool1d → 趋势分量                    │
│  └─ 季节性 = 原序列 - 趋势                       │
└─────────────────────────────────────────────────┘
    ↓
趋势分量: (batch_size, 8, 21)
季节性分量: (batch_size, 8, 21)
    ↓ 转置为 (batch_size, 21, 8)
    ↓
┌─────────────────────────────────────────────────┐
│ 线性预测                                         │
│  ├─ Linear_Trend: (8 → output_len)              │
│  └─ Linear_Seasonal: (8 → output_len)           │
└─────────────────────────────────────────────────┘
    ↓
趋势预测 + 季节性预测 = 合并输出: (batch_size, 21, output_len)
    ↓ 转置为 (batch_size, output_len, 21)
    ↓
特征投影: Linear(21, 3) — 将21维特征映射到3维风速
    ↓
输出: (batch_size, output_len, 3)
```

**参数量计算**：

| 组件 | 单步预测 | 多步预测 |
|------|----------|----------|
| Linear_Trend | $8 \times 1 + 1 = 9$ | $8 \times 16 + 16 = 144$ |
| Linear_Seasonal | $8 \times 1 + 1 = 9$ | $8 \times 16 + 16 = 144$ |
| 特征投影 | $21 \times 3 + 3 = 66$ | $21 \times 3 + 3 = 66$ |
| 残差权重 | 1 | 1 |
| **总计** | **85** | **355** |

**为何DLinear如此有效**：

1. **归纳偏置正确**：时序预测的本质是外推趋势和周期，线性模型天然适合
2. **避免过拟合**：参数量极少（仅数十到数百），即使小数据集也能训练好
3. **计算高效**：无需复杂的注意力计算
4. **可解释性强**：权重直接对应时间步的重要性

**DLinear的局限性**：
- 对于高度非线性的模式捕捉能力有限
- 在长期复杂预测任务中可能表现不如深度模型

#### 4.2.5 LSTNet模型

LSTNet（Long- and Short-term Time-series Network）[^20]是Lai等人于2018年在SIGIR会议上提出的时序预测模型。LSTNet通过组合CNN、GRU、Skip-RNN和Highway组件，同时建模短期局部模式和长期依赖关系。

**设计动机**：
- CNN擅长提取局部特征模式
- GRU/LSTM擅长捕捉长期依赖
- 时间序列通常具有周期性（如日周期、周周期）
- 自回归组件可以增强预测稳定性

**核心组件详解**：

**1. CNN层 — 短期模式提取**：

使用1D卷积提取相邻时间步之间的局部模式：
$$\mathbf{h}_t^c = \text{ReLU}(\mathbf{W}_c * \mathbf{X}_{t-k+1:t} + \mathbf{b}_c)$$

配置：32通道，kernel_size=3。

**2. GRU层 — 长期依赖建模**：

GRU（门控循环单元）是LSTM的简化版本，使用更少的门：
$$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1-\mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

配置：hidden_size=64。

**3. Skip-GRU — 周期性建模**：

直接在周期间隔的时间步之间建立连接，捕捉周期性模式：
```
Skip步长 p = 4
原序列: [t1, t2, t3, t4, t5, t6, t7, t8]
Skip采样: [t1, t5] 或 [t4, t8] — 每隔4步采样
```

这使得模型可以直接学习周期性变化规律。

**4. Highway组件 — 自回归增强**：

类似于残差连接，直接使用最近几个时间步的值预测：
$$\hat{\mathbf{y}}_{ar} = \mathbf{W}_{hw} \cdot \mathbf{X}_{t-w+1:t}$$

配置：窗口大小=4。

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓
┌─────────────────────────────────────────────────┐
│ CNN层: Conv1d(21, 32, kernel=3, padding=1)      │
│        → ReLU → Dropout(0.2)                    │
│ 输出: (batch_size, 8, 32)                       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ GRU层: GRU(input=32, hidden=64)                 │
│ 输出: (batch_size, 8, 64)                       │
│ 取最后时刻: (batch_size, 64)                    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Skip-GRU层 (可选):                               │
│ ├─ 采样: 每隔skip=4步取一个时间点                │
│ ├─ Skip-GRU(input=32, hidden=32)                │
│ ├─ Linear(32, 64)                               │
│ └─ 与GRU输出相加                                 │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ 输出层:                                          │
│ Linear(64, 128) → ReLU → Dropout(0.2)           │
│ Linear(128, output_len × 3)                     │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Highway组件 (自回归):                            │
│ 取最后4个时刻: (batch_size, 4×21=84)            │
│ Linear(84, output_len × 3)                      │
│ 与神经网络输出相加                               │
└─────────────────────────────────────────────────┘
    ↓ Reshape
输出: (batch_size, output_len, 3)
```

**本实验配置**:

| 参数 | 值 | 说明 |
|------|-----|------|
| cnn_channels | 32 | CNN通道数 |
| cnn_kernel | 3 | 卷积核大小 |
| rnn_hidden | 64 | GRU隐藏层维度 |
| skip_hidden | 32 | Skip-GRU隐藏层维度 |
| skip | 4 | 跳跃步长 |
| highway_window | 4 | 自回归窗口 |

**参数量**: 单步预测约38,000；多步预测约48,000

#### 4.2.6 CNN_LSTM模型

CNN_LSTM是一种混合模型，结合CNN的局部特征提取能力和LSTM的长期依赖建模能力。

**核心创新**：

**1. 多尺度特征提取**：

使用三种不同kernel size的卷积同时提取特征：
```
输入 → Conv1d(kernel=3) → 捕捉短期模式
    → Conv1d(kernel=5) → 捕捉中期模式
    → Conv1d(kernel=7) → 捕捉长期模式
```

多尺度特征拼接后，可以同时感知不同时间跨度的变化。

**2. 注意力机制**：

在LSTM输出上应用注意力，自适应地关注重要时刻：
$$\alpha_t = \frac{\exp(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_t))}{\sum_i \exp(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_i))}$$

**模型架构**：
```
输入: (batch_size, 8, 21)
    ↓
主干CNN: Conv1d(21→32, k=3) → BN → ReLU → Dropout
         Conv1d(32→64, k=3) → BN → ReLU → Dropout
    ↓
多尺度分支:
├─ Conv1d(21, 32, kernel=3, padding=1)
├─ Conv1d(21, 32, kernel=5, padding=2)
└─ Conv1d(21, 32, kernel=7, padding=3)
    ↓ Concat
合并: (batch_size, 8, 64+96=160)
    ↓
Bi-LSTM: (input=160, hidden=64, layers=2)
    ↓
注意力: 计算每个时刻的权重 → 加权求和
    ↓
全连接: Linear(128, 128) → ReLU → Dropout
        Linear(128, output_len × 3)
```

**参数量**: 单步预测约259,000；多步预测约265,000

#### 4.2.7 其他简单创新模型

**Persistence模型（持续性预测）**：

最简单的基线模型，假设未来风速等于最后观测值：
$$\hat{y}_{t+h} = y_t \times \text{scale} + \text{bias}$$

仅有6个可学习参数（3个scale + 3个bias），用于验证复杂模型是否学到了有价值的模式。

**TrendLinear模型**：

灵感来自Holt指数平滑[^21]，显式建模**水平**和**趋势**：

$$\text{Level}_t = \sum_{i=1}^{T} \alpha_i \cdot x_i \quad (\text{时间加权平均})$$
$$\text{Trend} = \frac{\text{近期均值} - \text{早期均值}}{时间间隔}$$
$$\hat{y}_{t+h} = \text{Level}_t + h \times \text{Trend} \times \text{decay}^h$$

参数量：约80个

**WindShear模型**：

直接嵌入风切变物理公式：
$$V(z) = V_{10m} \times (z / 10)^\alpha$$

其中$\alpha$是可学习的风切变指数。参数量：仅17个。

#### 4.2.8 创新模型汇总

| 模型 | 核心创新 | 参数量（多步） | 适用场景 |
|------|----------|----------------|----------|
| WaveNet | 膨胀因果卷积 + 门控激活 | ~231K | 多步预测 |
| HeightAttention | 多高度注意力 + 物理约束 | ~267K | 多高度数据 |
| TCN | 因果卷积 + 残差连接 + 注意力 | ~68K | 通用 |
| DLinear | 趋势-季节分解 + 线性预测 | ~355 | 短期预测 |
| LSTNet | CNN + GRU + Skip-RNN + Highway | ~48K | 周期性数据 |
| CNN_LSTM | 多尺度CNN + Bi-LSTM + 注意力 | ~265K | 多尺度模式 |
| TrendLinear | Holt平滑启发的趋势建模 | ~80 | 趋势明显 |
| WindShear | 物理约束（幂律公式） | 17 | 物理可解释 |
| Persistence | 持续性预测基线 | 6 | 基准对比 |

### 4.3 训练策略

本节详细介绍模型训练中采用的各种策略和技巧，这些策略对于获得良好的模型性能至关重要。

#### 4.3.1 优化器

使用Adam（Adaptive Moment Estimation）优化器[^22]，这是深度学习中最流行的优化器之一。

**Adam算法原理**：

Adam结合了动量（Momentum）和RMSprop的优点，维护一阶矩（均值）和二阶矩（未中心化方差）的指数移动平均：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{（一阶矩估计）}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{（二阶矩估计）}$$

偏差校正：
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

参数更新：
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**本实验配置**:

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate ($\eta$) | 0.001（单步）/ 0.0001（多步） | 初始学习率 |
| $\beta_1$ | 0.9 | 一阶矩衰减率 |
| $\beta_2$ | 0.999 | 二阶矩衰减率 |
| $\epsilon$ | 1e-8 | 数值稳定性常数 |
| weight_decay | 1e-5 | L2正则化系数 |

**权重衰减（L2正则化）**：

在损失函数中添加参数范数惩罚：
$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda \sum_i \theta_i^2$$

其中$\lambda = 10^{-5}$，防止过拟合。

#### 4.3.2 学习率调度

采用**双调度策略**：余弦退火重启（主调度器）+ 平台检测（辅助调度器）。

**1. 余弦退火重启（CosineAnnealingWarmRestarts）**[^23]：

学习率按余弦曲线周期性变化：
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

其中：
- $\eta_t$: 当前学习率
- $\eta_{min}$: 最小学习率
- $\eta_{max}$: 最大学习率（初始学习率）
- $T_{cur}$: 自上次重启以来的epoch数
- $T_i$: 当前周期的总epoch数

**重启机制**：当学习率降到最低点时，突然重启回最高点，帮助模型跳出局部最优。

```
学习率变化示意图:
     ↑
  η_max ─┬─────╮    ╭─────╮    ╭─────
         │      ╲  ╱      ╲  ╱
         │       ╲╱        ╲╱
  η_min ─┴───────┴──────────┴─────→ epoch
              T_0    T_0×mult
```

**2. 平台检测（ReduceLROnPlateau）**：

当验证损失在patience个epoch内不再下降时，降低学习率：
$$\eta_{new} = \eta_{old} \times \text{factor}$$

配置：factor=0.5，patience=10。

#### 4.3.3 早停机制

早停（Early Stopping）是防止过拟合的重要技术，当验证集性能不再提升时提前终止训练。

**本实验的早停策略**：

根据任务类型使用不同的评估指标：

| 任务 | 评估指标 | patience | 理由 |
|------|----------|----------|------|
| 单步预测 | MSE（越小越好） | 20 | 短期预测追求精确度 |
| 多步预测 | R²（越大越好） | 40 | 长期预测关注解释能力 |

**三种评估模式**：

1. **MSE模式**: 监控MSE，适合短期预测
   $$\text{score} = -\text{MSE} \quad \text{（转为越大越好）}$$

2. **R²模式**: 监控R²，适合长期预测
   $$\text{score} = R^2$$

3. **Combined模式**: 综合考虑两者
   $$\text{score} = R^2 - 0.1 \times \min(\text{MSE}, 1.0)$$

**早停流程**：
```python
if current_score > best_score + min_delta:
    best_score = current_score
    save_best_model()
    counter = 0
else:
    counter += 1
    if counter >= patience:
        early_stop = True
```

#### 4.3.4 损失函数

使用均方误差（MSE）作为损失函数，这是回归任务最常用的损失函数：

$$\mathcal{L}_{MSE} = \frac{1}{N \times T \times C}\sum_{n=1}^{N}\sum_{t=1}^{T}\sum_{c=1}^{C}(y_{n,t,c} - \hat{y}_{n,t,c})^2$$

其中：
- $N$: 批次大小
- $T$: 输出时间步数（1或16）
- $C$: 输出通道数（3个高度）
- $y$: 真实值
- $\hat{y}$: 预测值

**MSE的特点**：
- 对大误差敏感（平方惩罚）
- 梯度随误差线性变化，训练稳定
- 最小化MSE等价于最大似然估计（假设高斯噪声）

#### 4.3.5 其他训练技巧

**1. 梯度裁剪（Gradient Clipping）**：

防止梯度爆炸，将梯度范数限制在阈值内：
$$\mathbf{g} \leftarrow \frac{\mathbf{g}}{\max(1, \|\mathbf{g}\|_2 / \text{max\_norm})}$$

配置：max_norm=1.0

**2. 混合精度训练（AMP）**：

在A100等支持Tensor Core的GPU上使用FP16/BF16加速训练：
- 前向传播使用FP16节省显存和计算
- 梯度累积和参数更新保持FP32精度
- 使用GradScaler防止梯度下溢

**3. 批归一化（Batch Normalization）**：

在Linear、TCN等模型中使用BatchNorm：
- 加速训练收敛
- 允许使用更大的学习率
- 具有正则化效果

**4. Dropout正则化**：

训练时随机丢弃神经元，测试时使用全部神经元：
- 各模型Dropout率：0.2~0.3
- 防止过拟合
- 相当于训练多个子网络的集成

**5. 权重初始化**：

- Linear层：Xavier初始化
- LSTM层：正交初始化（权重）+ 零初始化（偏置）
- 遗忘门偏置初始化为1（帮助LSTM记忆长期信息）

**6. 数据加载优化**：

- num_workers：根据CPU核心数设置（A100服务器使用8）
- pin_memory：True（加速GPU数据传输）
- 预取：使用PyTorch DataLoader的默认预取

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

**1. 均方误差（MSE, Mean Squared Error）**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- **说明**：预测误差的平方均值，对大误差更敏感
- **优化方向**：越小越好

**2. 均方根误差（RMSE, Root Mean Squared Error）**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
- **说明**：MSE的平方根，与原数据同单位（m/s），直观易解释
- **优化方向**：越小越好

**3. 平均绝对误差（MAE, Mean Absolute Error）**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left| y_i - \hat{y}_i \right|$$
- **说明**：预测误差绝对值的均值，对异常值更鲁棒
- **优化方向**：越小越好

**4. 决定系数（R², Coefficient of Determination）**
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$
- **说明**：模型解释方差的比例，$R^2=1$表示完美拟合，$R^2=0$表示模型效果等同于均值预测
- **优化方向**：越大越好（最大为1）

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| MSE | 对大误差敏感，数学性质好 | 作为损失函数优化 |
| RMSE | 与原数据同单位，易解释 | **主要评判指标** |
| MAE | 对异常值鲁棒 | 需要稳健估计时 |
| R² | 反映拟合优度，无量纲 | 跨数据集比较 |

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
