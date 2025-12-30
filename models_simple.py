"""
简单基线模型模块
包含3个精选的简单但有创新价值的模型：
1. Persistence - 持久性基线（最简单的基准）
2. WindShear - 物理约束模型（嵌入风切变公式）
3. TrendLinear - 趋势感知线性模型（捕获短期趋势）

设计原则：参数量极少，但有明确的创新点和物理/统计意义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistenceModel(nn.Module):
    """
    持久性模型（Naive Baseline）
    
    创新点：作为所有时间序列预测的基准线
    - 如果复杂模型打不过这个，说明复杂模型有问题
    - 在实际应用中，这个基线常常很难被超越
    
    原理：直接用最后一个时刻的风速值作为预测
    参数量：6（仅缩放和偏置）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(PersistenceModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 风速特征索引（根据data_loader.py：风向×3, 温度×3, 气压×3, 湿度×3, 时间×6, 风速×3）
        self.target_indices = [18, 19, 20]  # SpeedAvg_10m, 50m, 100m
        
        # 可学习的缩放和偏置（允许微调）
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 提取最后一个时刻的风速值
        if x.size(-1) > 20:
            last_speeds = x[:, -1, self.target_indices]
        else:
            last_speeds = x[:, -1, -self.num_targets:]
        
        # 缩放和偏置
        output = last_speeds * self.scale + self.bias
        
        # 重复 output_len 次
        output = output.unsqueeze(1).repeat(1, self.output_len, 1)
        
        return output


class WindShearModel(nn.Module):
    """
    风切变物理约束模型
    
    创新点：将气象学中的风廓线幂律公式直接嵌入神经网络
    
    物理背景：
    - 大气边界层内，风速随高度变化遵循幂律：V(z) = V_ref × (z/z_ref)^α
    - α 是风切变指数，通常在 0.1~0.4 之间
    - 取决于地表粗糙度、大气稳定性等因素
    
    模型设计：
    1. 学习风切变指数 α（而不是硬编码）
    2. 结合时间加权的历史观测
    3. 添加趋势预测组件
    4. 物理预测与数据驱动预测融合
    
    参数量：约 20 个（极少但有物理意义）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(WindShearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 高度配置（米）
        self.register_buffer('heights', torch.tensor([10.0, 50.0, 100.0]))
        self.ref_height = 10.0
        
        # 风速特征索引
        self.target_indices = [18, 19, 20]
        
        # ========== 物理参数 ==========
        # 可学习的风切变指数（典型值 0.1-0.4）
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        
        # ========== 时序参数 ==========
        # 时间加权（学习历史各时刻的重要性）
        self.time_weights = nn.Parameter(torch.zeros(input_len))
        
        # 趋势权重
        self.trend_weight = nn.Parameter(torch.tensor(0.0))
        
        # ========== 融合参数 ==========
        # 物理预测 vs 数据驱动预测的融合比例
        self.physics_ratio_logit = nn.Parameter(torch.tensor(0.0))
        
        # 输出调整
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # 提取风速历史
        if x.size(-1) > 20:
            speeds = x[:, :, self.target_indices]  # (batch, input_len, 3)
        else:
            speeds = x[:, :, -self.num_targets:]
        
        # ========== 1. 物理模型预测 ==========
        # 计算风切变指数 α（映射到 0.05-0.5 范围）
        alpha = 0.05 + 0.45 * torch.sigmoid(self.alpha_logit)
        
        # 使用10m风速作为参考（时间加权平均）
        time_weights = F.softmax(self.time_weights, dim=0)
        v_10m_avg = torch.einsum('bt,t->b', speeds[:, :, 0], time_weights)  # (batch,)
        
        # 根据幂律公式计算各高度风速
        height_ratios = self.heights / self.ref_height  # [1, 5, 10]
        physics_pred = v_10m_avg.unsqueeze(1) * (height_ratios.unsqueeze(0) ** alpha)  # (batch, 3)
        
        # ========== 2. 数据驱动预测 ==========
        # 时间加权的历史平均
        data_pred = torch.einsum('btc,t->bc', speeds, time_weights)  # (batch, 3)
        
        # ========== 3. 趋势预测 ==========
        # 最近趋势 = 最后2时刻均值 - 最初2时刻均值
        recent = speeds[:, -2:, :].mean(dim=1)
        early = speeds[:, :2, :].mean(dim=1)
        trend = (recent - early) * torch.sigmoid(self.trend_weight)
        
        # ========== 4. 融合 ==========
        physics_ratio = torch.sigmoid(self.physics_ratio_logit)
        fused = physics_ratio * physics_pred + (1 - physics_ratio) * data_pred + trend
        
        # 输出调整
        output = fused * self.scale + self.bias
        
        # 重复 output_len 次
        output = output.unsqueeze(1).repeat(1, self.output_len, 1)
        
        return output
    
    def get_learned_alpha(self):
        """获取学习到的风切变指数（用于分析）"""
        with torch.no_grad():
            return (0.05 + 0.45 * torch.sigmoid(self.alpha_logit)).item()


class TrendLinearModel(nn.Module):
    """
    趋势感知线性模型
    
    创新点：显式建模时间序列的趋势和水平
    
    灵感来源：Holt's Linear Exponential Smoothing
    - 分解为：水平(Level) + 趋势(Trend)
    - 用可学习的方式替代传统的固定公式
    
    与普通Linear的区别：
    1. 显式分离"当前水平"和"变化趋势"
    2. 趋势外推到未来多步
    3. 参数量更少但更有针对性
    
    参数量：约 100 个（比普通Linear少很多）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(TrendLinearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        self.target_indices = [18, 19, 20]
        
        # ========== 水平估计 ==========
        # 时间加权（用于估计当前水平）
        self.level_weights = nn.Parameter(torch.zeros(input_len))
        
        # ========== 趋势估计 ==========
        # 趋势计算的时间范围权重
        self.trend_start_weights = nn.Parameter(torch.zeros(input_len // 2))
        self.trend_end_weights = nn.Parameter(torch.zeros(input_len // 2))
        
        # 趋势衰减（越远的未来，趋势影响越小）
        self.trend_decay = nn.Parameter(torch.tensor(0.0))
        
        # ========== 其他特征的影响 ==========
        # 用其他气象特征（温度、气压等）调整预测
        # 只用最后时刻的其他特征
        other_features = num_features - num_targets  # 21 - 3 = 18
        self.feature_adjust = nn.Linear(other_features, num_targets)
        
        # 输出调整
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
        
        # 初始化
        nn.init.zeros_(self.feature_adjust.weight)
        nn.init.zeros_(self.feature_adjust.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 提取风速历史
        if x.size(-1) > 20:
            speeds = x[:, :, self.target_indices]  # (batch, input_len, 3)
            other_feats = torch.cat([
                x[:, -1, :18],  # 风向、温度、气压、湿度、时间特征
            ], dim=-1)  # (batch, 18)
        else:
            speeds = x[:, :, -self.num_targets:]
            other_feats = x[:, -1, :-self.num_targets]
        
        # ========== 1. 估计当前水平 ==========
        level_weights = F.softmax(self.level_weights, dim=0)
        level = torch.einsum('btc,t->bc', speeds, level_weights)  # (batch, 3)
        
        # ========== 2. 估计趋势 ==========
        half = self.input_len // 2
        
        # 早期平均
        start_weights = F.softmax(self.trend_start_weights, dim=0)
        start_avg = torch.einsum('btc,t->bc', speeds[:, :half, :], start_weights)
        
        # 近期平均
        end_weights = F.softmax(self.trend_end_weights, dim=0)
        end_avg = torch.einsum('btc,t->bc', speeds[:, -half:, :], end_weights)
        
        # 每时刻的趋势
        trend_per_step = (end_avg - start_avg) / half
        
        # ========== 3. 外推到未来 ==========
        # 趋势衰减：越远的未来，趋势影响越小
        decay = torch.sigmoid(self.trend_decay)
        
        # 生成每个未来时刻的预测
        outputs = []
        for t in range(self.output_len):
            # 趋势随时间衰减
            trend_effect = trend_per_step * (t + 1) * (decay ** t)
            pred_t = level + trend_effect
            outputs.append(pred_t)
        
        output = torch.stack(outputs, dim=1)  # (batch, output_len, 3)
        
        # ========== 4. 用其他特征微调 ==========
        feat_adjust = self.feature_adjust(other_feats)  # (batch, 3)
        output = output + feat_adjust.unsqueeze(1)
        
        # 输出调整
        output = output * self.scale + self.bias
        
        return output


# ==================== 辅助函数 ====================

def get_simple_model(model_name, input_len, output_len, num_features, num_targets):
    """
    获取简单模型实例
    
    可用模型：
    - Persistence: 持久性基线
    - WindShear: 物理约束模型
    - TrendLinear: 趋势感知线性模型
    """
    models = {
        'Persistence': PersistenceModel,
        'WindShear': WindShearModel,
        'TrendLinear': TrendLinearModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](input_len, output_len, num_features, num_targets)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("测试简单基线模型（精选3个）")
    print("=" * 60)
    
    batch_size = 32
    input_len = 8
    output_len_single = 1
    output_len_multi = 16
    num_features = 21
    num_targets = 3
    
    x = torch.randn(batch_size, input_len, num_features)
    
    model_names = ['Persistence', 'WindShear', 'TrendLinear']
    
    print("\n单步预测 (8h → 1h)：")
    print("-" * 50)
    for name in model_names:
        model = get_simple_model(name, input_len, output_len_single, num_features, num_targets)
        output = model(x)
        params = count_parameters(model)
        print(f"{name:15s} | 参数量: {params:>6,} | 输出: {tuple(output.shape)}")
    
    print("\n多步预测 (8h → 16h)：")
    print("-" * 50)
    for name in model_names:
        model = get_simple_model(name, input_len, output_len_multi, num_features, num_targets)
        output = model(x)
        params = count_parameters(model)
        print(f"{name:15s} | 参数量: {params:>6,} | 输出: {tuple(output.shape)}")
    
    # 测试WindShear的物理参数
    print("\n" + "-" * 50)
    ws_model = get_simple_model('WindShear', input_len, output_len_single, num_features, num_targets)
    print(f"WindShear 初始风切变指数 α: {ws_model.get_learned_alpha():.4f}")
    print("(训练后会学习到适合数据的 α 值)")
    
    print("\n✅ 所有模型测试通过！")
