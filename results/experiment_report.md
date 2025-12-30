# 风速序列预测实验报告

**课程名称**: 机器学习实践  
**实验日期**: 2025年12月  
**实验类型**: 时间序列预测  

---

## 摘要

本实验基于深度学习方法，利用来自HuggingFace的多高度风速数据集，实现了风速的单步预测（8小时→1小时）和多步预测（8小时→16小时）任务。实验对比了3种基础模型（Linear、LSTM、Transformer）和9种创新模型的性能，采用MSE、RMSE、MAE、R²等指标进行评估。实验结果表明：在单步预测任务中，LSTM模型以R²=0.8870取得最佳性能；在多步预测任务中，创新模型WaveNet以R²=0.5375显著优于基础模型。本报告详细介绍了数据预处理、特征工程、模型设计及实验分析过程。

**关键词**: 风速预测、时间序列、深度学习、LSTM、Transformer、WaveNet

---

## 1. 实验背景与目的

### 1.1 研究背景

风速预测是气象学和能源领域的重要研究课题，在风力发电、航空调度、建筑设计等领域具有广泛的应用价值。准确的风速预测可以帮助电网运营商优化风力发电的调度计划，提高可再生能源的利用效率，降低电力系统的运营成本[1]。

传统的风速预测方法主要包括物理方法和统计方法。物理方法基于数值天气预报模型（NWP），通过求解大气动力学方程来预测风速，但计算成本高且对初始条件敏感[2]。统计方法如自回归移动平均模型（ARIMA）和指数平滑法简单高效，但难以捕捉风速数据的非线性特征[3]。

近年来，深度学习方法在时间序列预测领域取得了显著进展。循环神经网络（RNN）及其变体长短期记忆网络（LSTM）能够有效建模时序数据中的长期依赖关系[4]。Transformer架构通过自注意力机制实现了并行计算，在自然语言处理和时间序列预测中展现出优异性能[5]。此外，时序卷积网络（TCN）和WaveNet等基于卷积的模型也在序列建模任务中表现突出[6][7]。

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

### 1.3 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Linux |
| 编程语言 | Python 3.8+ |
| 深度学习框架 | PyTorch 2.0+ |
| 计算设备 | CUDA GPU |
| 批次大小 | 512 |
| 随机种子 | 42 |

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
2. 可以利用不同高度风速的物理关联（风切变效应）
3. 便于后续的滑动窗口采样

---

## 3. 数据预处理与特征工程

### 3.1 缺失值处理

数据集中存在少量缺失值，主要集中在10米高度的风速数据。缺失值统计如下：

| 字段 | 缺失数量 |
|------|----------|
| Speed Avg 10m | 10,573 |

**处理方法**: 采用线性插值法（Linear Interpolation）填充缺失值：

```python
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
```

线性插值法的优点：
- 保持数据的连续性和趋势
- 对时间序列数据特别适用
- 不会引入突变点

对于边界处无法插值的缺失值，使用前向填充（ffill）和后向填充（bfill）作为补充。

### 3.2 异常值处理

采用四分位距（IQR）方法检测和处理异常值。IQR方法的原理是：

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

为了捕捉风速的周期性变化规律，构造了以下时间特征：

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

采用Z-score标准化（StandardScaler），将特征缩放到均值为0、标准差为1的分布：

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

**时序划分**: 按时间顺序划分，确保训练集在验证集之前，验证集在测试集之前，符合时间序列预测的实际场景。

---

## 4. 模型原理与设计

### 4.1 基础模型

#### 4.1.1 Linear模型（多层感知机）

Linear模型是基于多层感知机（MLP）的基线模型，将输入序列展平后通过全连接层预测输出。

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

长短期记忆网络（LSTM）[4]是一种特殊的RNN结构，通过门控机制解决了传统RNN的梯度消失问题，能够有效捕捉时序数据中的长期依赖关系。

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
- 隐藏层维度: 384
- 层数: 4
- 双向: 是
- Dropout: 0.3

**参数量**: 约6,000,000

#### 4.1.3 Transformer模型

Transformer[5]基于自注意力机制，能够并行处理序列数据并捕捉长距离依赖关系。

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
- 模型维度: 192
- 注意力头数: 8
- 编码器层数: 5
- 前馈网络维度: 512
- Dropout: 0.1

**参数量**: 约3,000,000

### 4.2 创新模型

#### 4.2.1 WaveNet模型（多步预测最佳）

WaveNet[7]是一种基于膨胀因果卷积的生成模型，最初用于音频合成，后被广泛应用于时间序列预测。

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
   - 嵌入风廓线幂律公式: $V(z) = V_{ref} \times (z/z_{ref})^\alpha$
   - 学习风切变指数$\alpha$

**参数量**: 约200,000

#### 4.2.3 TCN模型（时序卷积网络）

TCN[6]是一种因果卷积网络，通过膨胀卷积扩大感受野，同时保持因果性。

**核心特点**:
- 因果卷积: 确保预测只使用过去信息
- 膨胀卷积: 指数级扩大感受野
- 残差连接: 支持深层网络训练

**参数量**: 约100,000

#### 4.2.4 DLinear模型

DLinear[8]来自AAAI 2023论文，证明简单的线性模型在时间序列预测中也能取得优异性能。

**核心创新**:
- 趋势-季节分解: 将时间序列分解为趋势和季节性分量
- 独立线性层: 分别预测两个分量
- 极少参数: 仅约100个参数

**参数量**: 约100

#### 4.2.5 其他创新模型

| 模型 | 核心创新 | 参数量 |
|------|----------|--------|
| CNN_LSTM | 多尺度CNN特征提取 + LSTM序列建模 | ~300K |
| LSTNet | CNN + GRU + Skip-RNN + Highway | ~150K |
| TrendLinear | Holt指数平滑启发的趋势分解 | ~100 |
| WindShear | 嵌入风切变物理约束 | ~20 |
| Persistence | 基线模型（持续性预测） | 6 |

### 4.3 训练策略

#### 4.3.1 优化器

使用Adam优化器，配置如下：
- 初始学习率: 0.001（单步）/ 0.0001（多步）
- 权重衰减: 1e-5

#### 4.3.2 学习率调度

采用余弦退火重启（CosineAnnealingWarmRestarts）+ 平台检测（ReduceLROnPlateau）双调度策略：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

#### 4.3.3 早停机制

- 单步预测: 使用MSE作为早停指标，patience=20
- 多步预测: 使用R²作为早停指标，patience=40

#### 4.3.4 损失函数

使用均方误差（MSE）作为损失函数：

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

---

## 5. 实验结果与分析

### 5.1 评估指标

本实验使用以下四个指标评估模型性能：

| 指标 | 公式 | 说明 | 优化方向 |
|------|------|------|----------|
| MSE | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | 均方误差 | 越小越好 |
| RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | 均方根误差，与原数据同单位 | 越小越好 |
| MAE | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | 平均绝对误差 | 越小越好 |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数，模型解释方差比例 | 越大越好 |

**推荐**: 以RMSE作为主要评判指标（单位与风速相同，直观易解释），R²作为辅助指标（反映模型拟合优度）。

### 5.2 单步预测结果（8小时→1小时）

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

1. **LSTM表现最佳**: 双向LSTM能够同时利用过去和未来的上下文信息，门控机制有效捕捉时序依赖，在单步预测中以R²=0.8870取得最佳性能。

2. **前三名差距极小**: LSTM、DLinear、TCN的RMSE差距仅0.001 m/s，表明短期预测任务相对简单，多种模型都能达到较好效果。

3. **Persistence基线很强**: 简单的持续性预测（使用最后一个时刻的值作为预测）达到R²=0.8842，说明风速具有较强的自相关性。

4. **Transformer表现不佳**: 可能由于数据量有限（约10,000条），Transformer更适合大规模数据集。

### 5.3 多步预测结果（8小时→16小时）

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

1. **WaveNet取得最佳性能**: 门控卷积+膨胀因果卷积在长期序列建模上展现出优势，R²=0.5375显著优于基础模型。

2. **创新模型超越基础模型**: 前两名都是创新模型，HeightAttention的多高度注意力机制有效捕获了不同高度风速的关联。

3. **多步预测难度大**: 所有模型的R²从0.88（单步）下降到0.53（多步），RMSE从0.91 m/s增加到1.86 m/s，反映了长期预测的固有困难。

4. **Persistence基线被超越**: 多步预测中，复杂模型（如WaveNet）明显优于持续性基线，说明模型学到了有价值的时序模式。

### 5.4 创新模型价值分析

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

### 5.5 训练过程分析

各模型训练收敛情况：

| 模型 | 任务 | 总Epoch数 | 最佳Epoch | 最佳验证MSE |
|------|------|-----------|-----------|-------------|
| LSTM | singlestep | 370+ | 283 | 0.0698 |
| LSTM | multistep | 415+ | 32 | 0.3365 |
| WaveNet | singlestep | 1341+ | 1053 | 0.0701 |
| WaveNet | multistep | 1267+ | 1267 | 0.3285 |
| Transformer | singlestep | 500+ | 500 | 0.1525 |

**观察**:
- 创新模型（如WaveNet、TCN）需要更多训练轮数才能收敛
- 多步预测任务的验证损失明显高于单步预测
- LSTM在较少轮数内即可达到最佳性能

---

## 6. 结论

### 6.1 主要结论

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

### 6.2 局限性与未来工作

1. **数据规模**: 约10,000条样本对于Transformer等复杂模型可能不足
2. **外部特征**: 未引入天气预报数据等外部信息
3. **集成学习**: 可尝试模型集成进一步提升性能
4. **不确定性量化**: 可添加概率预测输出置信区间

---

## 7. 参考文献

[1] Wang, J., Song, Y., Liu, F., & Hou, R. (2016). Analysis and application of forecasting models in wind power integration: A review of multi-step-ahead wind speed forecasting models. Renewable and Sustainable Energy Reviews, 60, 960-981.

[2] Soman, S. S., Zareipour, H., Malik, O., & Mandal, P. (2010). A review of wind power and wind speed forecasting methods with different time horizons. In North American Power Symposium (NAPS), 2010 (pp. 1-8). IEEE.

[3] Erdem, E., & Shi, J. (2011). ARMA based approaches for forecasting the tuple of wind speed and direction. Applied Energy, 88(4), 1405-1414.

[4] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.

[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.

[6] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

[7] Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). WaveNet: A generative model for raw audio. arXiv preprint arXiv:1609.03499.

[8] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121-11128.

[9] Lai, G., Chang, W. C., Yang, Y., & Liu, H. (2018). Modeling long-and short-term temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (pp. 95-104).

[10] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778).

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

---

*报告生成时间: 2025年12月30日*  
*实验项目: 风速序列预测*
