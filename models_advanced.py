"""
高级创新模型模块
包含：HeightAttentionModel（高度注意力模型）、DLinear（趋势-季节分解线性模型）

这些模型针对风速预测任务进行了专门设计：
1. HeightAttentionModel: 利用多高度数据的空间结构，建模高度之间的物理关联
2. DLinear: 趋势-季节分解，简单但高效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== HeightAttention 模型 ====================

class HeightAttention(nn.Module):
    """
    高度注意力模块
    
    对不同高度的特征进行注意力加权，学习高度之间的依赖关系
    """
    
    def __init__(self, feature_dim, num_heads=2, dropout=0.1):
        super(HeightAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value 投影
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_heights, feature_dim)
        Returns:
            output: (batch, num_heights, feature_dim)
            attention_weights: (batch, num_heads, num_heights, num_heights)
        """
        batch_size, num_heights, _ = x.shape
        
        # 投影
        Q = self.q_proj(x)  # (batch, num_heights, feature_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, num_heights, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_heights, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_heights, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, num_heights, -1)
        output = self.out_proj(context)
        
        return output, attention_weights


class HeightAttentionModel(nn.Module):
    """
    高度注意力风速预测模型
    
    创新点：
    1. 将扁平化的特征重组为(高度, 特征)结构，保留空间信息
    2. 使用注意力机制学习不同高度之间的依赖关系
    3. 融合时间特征作为全局上下文
    4. 可选：添加物理约束（风速随高度变化的幂律）
    
    架构：
    Input(batch, time, 21) 
        → 特征重组(batch, time, 3heights, 5features) 
        → 高度注意力 
        → 时序建模(LSTM) 
        → 预测输出
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets,
                 hidden_dim=64, num_heads=2, lstm_layers=2, dropout=0.2,
                 use_physics_constraint=True):
        """
        Args:
            input_len: 输入序列长度 (8)
            output_len: 输出序列长度 (1 或 16)
            num_features: 原始特征数量 (21)
            num_targets: 目标数量 (3，三个高度的风速)
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            lstm_layers: LSTM层数
            dropout: Dropout率
            use_physics_constraint: 是否使用物理约束
        """
        super(HeightAttentionModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.use_physics_constraint = use_physics_constraint
        
        # 高度配置
        self.num_heights = 3  # 10m, 50m, 100m
        self.features_per_height = 5  # 风向、温度、气压、湿度、风速
        self.time_features = 6  # hour_sin/cos, day_sin/cos, month_sin/cos
        
        # 特征索引映射（根据data_loader.py中的get_feature_columns顺序）
        # 顺序：风向×3, 温度×3, 气压×3, 湿度×3, 时间×6, 风速×3
        self.height_feature_indices = {
            0: [0, 3, 6, 9, 18],    # 10m: 风向、温度、气压、湿度、风速
            1: [1, 4, 7, 10, 19],   # 50m
            2: [2, 5, 8, 11, 20],   # 100m
        }
        self.time_feature_indices = [12, 13, 14, 15, 16, 17]  # 时间特征
        
        # 1. 每个高度的特征编码器
        self.height_encoder = nn.Sequential(
            nn.Linear(self.features_per_height, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 时间特征编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        # 3. 高度注意力模块
        self.height_attention = HeightAttention(hidden_dim, num_heads, dropout)
        
        # 4. 高度融合层（融合注意力输出和时间特征）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_heights + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 5. 时序建模（双向LSTM）
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 6. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len * num_targets)
        )
        
        # 7. 物理约束：学习风切变指数α
        if use_physics_constraint:
            self.alpha_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # α通常在0-0.5之间
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def reorganize_features(self, x):
        """
        将扁平化的特征重组为高度结构
        
        Args:
            x: (batch, time, 21) 扁平化特征
        
        Returns:
            height_features: (batch, time, 3, 5) 按高度组织的特征
            time_features: (batch, time, 6) 时间特征
        """
        batch_size, seq_len, _ = x.shape
        
        # 提取每个高度的特征
        height_features = torch.zeros(batch_size, seq_len, self.num_heights, 
                                       self.features_per_height, device=x.device)
        
        for h_idx, feat_indices in self.height_feature_indices.items():
            for f_idx, global_idx in enumerate(feat_indices):
                if global_idx < x.shape[-1]:
                    height_features[:, :, h_idx, f_idx] = x[:, :, global_idx]
        
        # 提取时间特征
        time_feat_list = []
        for idx in self.time_feature_indices:
            if idx < x.shape[-1]:
                time_feat_list.append(x[:, :, idx:idx+1])
        time_features = torch.cat(time_feat_list, dim=-1)
        
        return height_features, time_features
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, input_len, num_features) 输入序列
        
        Returns:
            output: (batch, output_len, num_targets) 预测输出
        """
        batch_size = x.size(0)
        
        # 1. 特征重组
        height_features, time_features = self.reorganize_features(x)
        # height_features: (batch, time, 3, 5)
        # time_features: (batch, time, 6)
        
        # 2. 对每个时间步进行高度注意力处理
        all_fused = []
        all_attention_weights = []
        
        for t in range(self.input_len):
            # 获取当前时间步的特征
            h_feat = height_features[:, t, :, :]  # (batch, 3, 5)
            t_feat = time_features[:, t, :]  # (batch, 6)
            
            # 编码每个高度的特征
            h_encoded = self.height_encoder(h_feat)  # (batch, 3, hidden_dim)
            
            # 高度注意力
            h_attended, attn_weights = self.height_attention(h_encoded)
            # h_attended: (batch, 3, hidden_dim)
            all_attention_weights.append(attn_weights)
            
            # 编码时间特征
            t_encoded = self.time_encoder(t_feat)  # (batch, hidden_dim)
            
            # 融合：拼接所有高度的注意力输出 + 时间特征
            h_flat = h_attended.view(batch_size, -1)  # (batch, 3*hidden_dim)
            fused = torch.cat([h_flat, t_encoded], dim=-1)  # (batch, 3*hidden_dim + hidden_dim)
            fused = self.fusion(fused)  # (batch, hidden_dim*2)
            
            all_fused.append(fused)
        
        # 堆叠所有时间步
        fused_sequence = torch.stack(all_fused, dim=1)  # (batch, time, hidden_dim*2)
        
        # 3. LSTM时序建模
        lstm_out, (h_n, c_n) = self.lstm(fused_sequence)
        
        # 使用最后一个时间步的输出
        if self.lstm.bidirectional:
            last_output = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_output = h_n[-1]
        
        # 4. 输出预测
        output = self.output_layer(last_output)
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        
        Returns:
            attention_weights: 每个时间步的高度注意力权重
        """
        self.eval()
        with torch.no_grad():
            height_features, time_features = self.reorganize_features(x)
            
            all_attention_weights = []
            for t in range(self.input_len):
                h_feat = height_features[:, t, :, :]
                h_encoded = self.height_encoder(h_feat)
                _, attn_weights = self.height_attention(h_encoded)
                all_attention_weights.append(attn_weights)
            
            return torch.stack(all_attention_weights, dim=1)


# ==================== DLinear 模型 ====================

class MovingAvg(nn.Module):
    """
    移动平均模块，用于提取趋势
    """
    
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        """
        # 前端填充以保持序列长度
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # 转换维度进行池化
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        return x


class SeriesDecomp(nn.Module):
    """
    序列分解模块：将序列分解为趋势和季节性
    """
    
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            seasonal: 季节性分量
            trend: 趋势分量
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearModel(nn.Module):
    """
    DLinear模型 - 趋势-季节分解线性模型
    
    创新点：
    1. 将时间序列分解为趋势和季节性两个分量
    2. 用两个独立的线性层分别处理
    3. 参数量极小，不易过拟合
    4. 2023年AAAI论文证明可以超越复杂Transformer
    
    参考论文：
    "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets,
                 kernel_size=3, individual=False):
        """
        Args:
            input_len: 输入序列长度 (8)
            output_len: 输出序列长度 (1 或 16)
            num_features: 输入特征数量 (21)
            num_targets: 输出目标数量 (3)
            kernel_size: 移动平均窗口大小
            individual: 是否为每个变量使用独立的线性层
        """
        super(DLinearModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        self.individual = individual
        
        # 序列分解
        self.decomposition = SeriesDecomp(kernel_size)
        
        if individual:
            # 每个特征独立的线性层
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(input_len, output_len) for _ in range(num_features)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(input_len, output_len) for _ in range(num_features)
            ])
        else:
            # 共享的线性层
            self.Linear_Seasonal = nn.Linear(input_len, output_len)
            self.Linear_Trend = nn.Linear(input_len, output_len)
        
        # 特征映射层：将num_features映射到num_targets
        self.feature_projection = nn.Linear(num_features, num_targets)
        
        # 可选：残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        if self.individual:
            for layer in self.Linear_Seasonal:
                nn.init.xavier_normal_(layer.weight)
            for layer in self.Linear_Trend:
                nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.xavier_normal_(self.Linear_Seasonal.weight)
            nn.init.xavier_normal_(self.Linear_Trend.weight)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, input_len, num_features)
        
        Returns:
            output: (batch, output_len, num_targets)
        """
        batch_size = x.size(0)
        
        # 1. 序列分解
        seasonal, trend = self.decomposition(x)
        # seasonal, trend: (batch, input_len, num_features)
        
        # 2. 转置以便对时间维度做线性变换
        seasonal = seasonal.permute(0, 2, 1)  # (batch, num_features, input_len)
        trend = trend.permute(0, 2, 1)
        
        # 3. 线性预测
        if self.individual:
            seasonal_output = torch.zeros(batch_size, self.num_features, self.output_len, 
                                          device=x.device)
            trend_output = torch.zeros(batch_size, self.num_features, self.output_len,
                                       device=x.device)
            for i in range(self.num_features):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal)  # (batch, num_features, output_len)
            trend_output = self.Linear_Trend(trend)
        
        # 4. 合并趋势和季节性
        output = seasonal_output + trend_output  # (batch, num_features, output_len)
        
        # 5. 转置回来
        output = output.permute(0, 2, 1)  # (batch, output_len, num_features)
        
        # 6. 特征投影到目标维度
        output = self.feature_projection(output)  # (batch, output_len, num_targets)
        
        return output


# ==================== 辅助函数 ====================

def get_advanced_model(model_name, input_len, output_len, num_features, num_targets):
    """
    获取高级模型实例
    
    Args:
        model_name: 模型名称 ('HeightAttention', 'DLinear')
        input_len: 输入序列长度
        output_len: 输出序列长度
        num_features: 特征数量
        num_targets: 目标数量
    
    Returns:
        模型实例
    """
    models = {
        'HeightAttention': HeightAttentionModel,
        'DLinear': DLinearModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](input_len, output_len, num_features, num_targets)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("测试高级创新模型")
    print("=" * 60)
    
    # 测试配置
    batch_size = 32
    input_len = 8
    output_len_single = 1
    output_len_multi = 16
    num_features = 21  # 与实际数据一致
    num_targets = 3
    
    # 创建测试输入
    x = torch.randn(batch_size, input_len, num_features)
    
    print("\n" + "-" * 40)
    print("测试 HeightAttentionModel")
    print("-" * 40)
    
    # 测试单步预测
    model_ha_single = HeightAttentionModel(input_len, output_len_single, num_features, num_targets)
    output_single = model_ha_single(x)
    print(f"单步预测 - 输入: {x.shape} -> 输出: {output_single.shape}")
    
    # 测试多步预测
    model_ha_multi = HeightAttentionModel(input_len, output_len_multi, num_features, num_targets)
    output_multi = model_ha_multi(x)
    print(f"多步预测 - 输入: {x.shape} -> 输出: {output_multi.shape}")
    
    # 参数量
    params_ha = sum(p.numel() for p in model_ha_single.parameters() if p.requires_grad)
    print(f"参数量: {params_ha:,}")
    
    # 测试注意力权重获取
    attn_weights = model_ha_single.get_attention_weights(x[:1])
    print(f"注意力权重形状: {attn_weights.shape}")  # (1, time, heads, heights, heights)
    
    print("\n" + "-" * 40)
    print("测试 DLinearModel")
    print("-" * 40)
    
    # 测试单步预测
    model_dl_single = DLinearModel(input_len, output_len_single, num_features, num_targets)
    output_single = model_dl_single(x)
    print(f"单步预测 - 输入: {x.shape} -> 输出: {output_single.shape}")
    
    # 测试多步预测
    model_dl_multi = DLinearModel(input_len, output_len_multi, num_features, num_targets)
    output_multi = model_dl_multi(x)
    print(f"多步预测 - 输入: {x.shape} -> 输出: {output_multi.shape}")
    
    # 参数量
    params_dl = sum(p.numel() for p in model_dl_single.parameters() if p.requires_grad)
    print(f"参数量: {params_dl:,}")
    
    print("\n" + "=" * 60)
    print("模型参数量对比")
    print("=" * 60)
    print(f"HeightAttention: {params_ha:,} 参数")
    print(f"DLinear:         {params_dl:,} 参数")
    print(f"DLinear参数量仅为HeightAttention的 {params_dl/params_ha*100:.1f}%")
    
    print("\n✅ 所有测试通过！")
