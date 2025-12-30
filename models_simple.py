"""
简单基线模型模块
包含：持久性模型、移动平均模型、简单线性模型等
这些模型参数量极少，适合小数据集，常常效果出奇地好

原理：对于时间序列预测，最近的历史值往往是最好的预测器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistenceModel(nn.Module):
    """
    持久性模型（Naive Baseline）
    
    最简单的基线：直接用最后一个时刻的值作为预测
    实际应用中这个基线常常很难被超越！
    
    参数量：0（无可学习参数）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(PersistenceModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 目标特征的索引（风速在特征的最后3列：18, 19, 20）
        # 根据 data_loader.py 中的顺序：风向×3, 温度×3, 气压×3, 湿度×3, 时间×6, 风速×3
        self.target_indices = [18, 19, 20]  # SpeedAvg_10m, 50m, 100m
        
        # 虽然是持久性模型，但加一个可学习的缩放因子可能有帮助
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_features)
        Returns:
            output: (batch, output_len, num_targets)
        """
        batch_size = x.size(0)
        
        # 提取最后一个时刻的风速值
        # 注意：如果 target_indices 超出范围，使用最后3个特征
        if x.size(-1) > 20:
            last_speeds = x[:, -1, self.target_indices]  # (batch, 3)
        else:
            # 备选：使用最后3个特征
            last_speeds = x[:, -1, -self.num_targets:]
        
        # 应用可学习的缩放和偏置
        last_speeds = last_speeds * self.scale + self.bias
        
        # 重复 output_len 次
        output = last_speeds.unsqueeze(1).repeat(1, self.output_len, 1)
        
        return output


class MovingAverageModel(nn.Module):
    """
    移动平均模型
    
    用过去 window_size 个时刻的加权平均作为预测
    权重可学习，相当于一个简单的1D卷积
    
    参数量：约 window_size * num_targets（极少）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, window_size=None):
        super(MovingAverageModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        self.window_size = window_size if window_size else input_len
        
        # 目标特征索引
        self.target_indices = [18, 19, 20]
        
        # 可学习的时间权重（每个目标独立）
        self.time_weights = nn.Parameter(torch.ones(num_targets, self.window_size) / self.window_size)
        
        # 输出缩放
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 提取风速历史
        if x.size(-1) > 20:
            speeds = x[:, -self.window_size:, self.target_indices]  # (batch, window, 3)
        else:
            speeds = x[:, -self.window_size:, -self.num_targets:]
        
        # 归一化权重（确保和为1）
        weights = F.softmax(self.time_weights, dim=1)  # (3, window)
        
        # 加权平均
        # speeds: (batch, window, 3), weights: (3, window)
        weighted_avg = torch.einsum('bwt,tw->bt', speeds, weights)  # (batch, 3)
        
        # 缩放和偏置
        weighted_avg = weighted_avg * self.scale + self.bias
        
        # 重复 output_len 次
        output = weighted_avg.unsqueeze(1).repeat(1, self.output_len, 1)
        
        return output


class SimpleLinearModel(nn.Module):
    """
    最简单的线性模型
    
    直接将输入展平后通过一个线性层映射到输出
    没有任何隐藏层，参数量极少
    
    参数量：input_len * num_features * output_len * num_targets + bias
           ≈ 8 * 21 * 1 * 3 ≈ 500（单步）
           ≈ 8 * 21 * 16 * 3 ≈ 8000（多步）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(SimpleLinearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 单一线性层
        self.linear = nn.Linear(input_len * num_features, output_len * num_targets)
        
        # 初始化
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 展平
        x_flat = x.view(batch_size, -1)  # (batch, input_len * num_features)
        
        # 线性变换
        output = self.linear(x_flat)  # (batch, output_len * num_targets)
        
        # 重塑
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class LastValueLinearModel(nn.Module):
    """
    基于最后时刻的线性模型
    
    只使用最后一个时刻的特征进行预测
    参数量更少，但仍有一定学习能力
    
    参数量：num_features * output_len * num_targets ≈ 21 * 1 * 3 = 63（单步）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(LastValueLinearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 只用最后时刻的特征
        self.linear = nn.Linear(num_features, output_len * num_targets)
        
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 只取最后一个时刻
        x_last = x[:, -1, :]  # (batch, num_features)
        
        # 线性变换
        output = self.linear(x_last)  # (batch, output_len * num_targets)
        
        # 重塑
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class ExponentialSmoothingModel(nn.Module):
    """
    指数平滑模型
    
    给近期数据指数级更高的权重
    经典的时间序列预测方法
    
    参数量：仅 1 个 alpha 参数 + 缩放偏置
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(ExponentialSmoothingModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        self.target_indices = [18, 19, 20]
        
        # 可学习的平滑系数（每个目标独立）
        # alpha 越大，越重视近期数据
        self.alpha_logit = nn.Parameter(torch.zeros(num_targets))
        
        # 输出调整
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 提取风速历史
        if x.size(-1) > 20:
            speeds = x[:, :, self.target_indices]  # (batch, input_len, 3)
        else:
            speeds = x[:, :, -self.num_targets:]
        
        # 计算 alpha（使用 sigmoid 确保在 0-1 之间）
        alpha = torch.sigmoid(self.alpha_logit)  # (3,)
        
        # 指数平滑权重
        # weights[t] = alpha * (1-alpha)^(T-1-t)，最近的权重最大
        T = speeds.size(1)
        t = torch.arange(T, device=x.device).float()
        
        # 为每个目标计算权重
        weights = alpha.unsqueeze(0) * ((1 - alpha).unsqueeze(0) ** (T - 1 - t).unsqueeze(1))
        # weights: (T, 3)
        
        # 归一化
        weights = weights / weights.sum(dim=0, keepdim=True)
        
        # 加权平均
        smoothed = torch.einsum('btc,tc->bc', speeds, weights)  # (batch, 3)
        
        # 缩放和偏置
        smoothed = smoothed * self.scale + self.bias
        
        # 重复 output_len 次
        output = smoothed.unsqueeze(1).repeat(1, self.output_len, 1)
        
        return output


class WindShearModel(nn.Module):
    """
    风切变物理模型（基于幂律公式）
    
    利用风速与高度的物理关系：V(z) = V_ref * (z / z_ref)^alpha
    
    创新点：
    1. 直接嵌入物理先验知识
    2. 学习风切变指数 alpha
    3. 结合历史趋势预测
    
    参数量：极少（约10个参数）
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets):
        super(WindShearModel, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 高度值（米）
        self.heights = torch.tensor([10.0, 50.0, 100.0])
        self.ref_height = 10.0
        
        # 目标特征索引
        self.target_indices = [18, 19, 20]
        
        # 可学习的风切变指数（典型值 0.1-0.4）
        # 使用 sigmoid 映射到合理范围
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        
        # 时间加权（用于选择历史时刻的重要性）
        self.time_weights = nn.Parameter(torch.zeros(input_len))
        
        # 趋势系数
        self.trend_weight = nn.Parameter(torch.tensor(0.1))
        
        # 输出微调
        self.scale = nn.Parameter(torch.ones(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_features)
        """
        batch_size = x.size(0)
        device = x.device
        
        # 移动高度tensor到正确设备
        heights = self.heights.to(device)
        
        # 提取风速历史
        if x.size(-1) > 20:
            speeds = x[:, :, self.target_indices]  # (batch, input_len, 3)
        else:
            speeds = x[:, :, -self.num_targets:]
        
        # 计算风切变指数 alpha（映射到 0.05-0.5 范围）
        alpha = 0.05 + 0.45 * torch.sigmoid(self.alpha_logit)
        
        # 计算时间加权平均的基准风速
        time_weights = F.softmax(self.time_weights, dim=0)  # (input_len,)
        base_speeds = torch.einsum('btc,t->bc', speeds, time_weights)  # (batch, 3)
        
        # 计算趋势（最后2个时刻与最初2个时刻的差异）
        recent = speeds[:, -2:, :].mean(dim=1)  # (batch, 3)
        early = speeds[:, :2, :].mean(dim=1)    # (batch, 3)
        trend = (recent - early) * torch.sigmoid(self.trend_weight)
        
        # 应用物理约束：确保高度关系合理
        # 使用 10m 风速作为参考，计算其他高度
        v_10m = base_speeds[:, 0:1]  # (batch, 1)
        
        # 根据幂律计算各高度风速
        height_ratios = (heights / self.ref_height).unsqueeze(0)  # (1, 3)
        physics_pred = v_10m * (height_ratios ** alpha)  # (batch, 3)
        
        # 融合：物理预测 + 直接观测 + 趋势
        # 0.5 * 物理约束 + 0.5 * 实际观测 + 趋势
        final_pred = 0.5 * physics_pred + 0.5 * base_speeds + trend
        
        # 输出调整
        final_pred = final_pred * self.scale + self.bias
        
        # 重复 output_len 次
        output = final_pred.unsqueeze(1).repeat(1, self.output_len, 1)
        
        return output


# ==================== 辅助函数 ====================

def get_simple_model(model_name, input_len, output_len, num_features, num_targets):
    """
    获取简单模型实例
    
    Args:
        model_name: 模型名称
        input_len: 输入序列长度
        output_len: 输出序列长度
        num_features: 特征数量
        num_targets: 目标数量
    
    Returns:
        模型实例
    """
    models = {
        'Persistence': PersistenceModel,
        'MovingAvg': MovingAverageModel,
        'SimpleLinear': SimpleLinearModel,
        'LastValueLinear': LastValueLinearModel,
        'ExpSmoothing': ExponentialSmoothingModel,
        'WindShear': WindShearModel,
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
    print("测试简单基线模型")
    print("=" * 60)
    
    # 测试配置
    batch_size = 32
    input_len = 8
    output_len_single = 1
    output_len_multi = 16
    num_features = 21
    num_targets = 3
    
    # 创建测试输入
    x = torch.randn(batch_size, input_len, num_features)
    
    model_names = ['Persistence', 'MovingAvg', 'SimpleLinear', 
                   'LastValueLinear', 'ExpSmoothing', 'WindShear']
    
    print("\n单步预测模型对比：")
    print("-" * 50)
    for name in model_names:
        model = get_simple_model(name, input_len, output_len_single, num_features, num_targets)
        output = model(x)
        params = count_parameters(model)
        print(f"{name:20s} | 参数量: {params:>6,} | 输出: {tuple(output.shape)}")
    
    print("\n多步预测模型对比：")
    print("-" * 50)
    for name in model_names:
        model = get_simple_model(name, input_len, output_len_multi, num_features, num_targets)
        output = model(x)
        params = count_parameters(model)
        print(f"{name:20s} | 参数量: {params:>6,} | 输出: {tuple(output.shape)}")
    
    print("\n✅ 所有简单模型测试通过！")
