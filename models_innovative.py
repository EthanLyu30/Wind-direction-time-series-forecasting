"""
创新模型模块
包含：CNN-LSTM混合模型、TCN模型、WaveNet模型、N-BEATS模型、集成模型
用于获得额外的25%创新分数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import CNN_LSTM_CONFIG, TCN_CONFIG, ENSEMBLE_CONFIG


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM混合模型
    
    创新点：
    1. 使用CNN提取局部时序特征
    2. 使用LSTM捕获长期依赖
    3. 多尺度特征融合
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=CNN_LSTM_CONFIG):
        super(CNNLSTMModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        cnn_channels = config['cnn_channels']
        kernel_size = config['kernel_size']
        lstm_hidden_size = config['lstm_hidden_size']
        lstm_num_layers = config['lstm_num_layers']
        dropout = config['dropout']
        
        # 多尺度CNN层（不同kernel size捕获不同尺度的特征）
        self.conv_layers = nn.ModuleList()
        for i, out_channels in enumerate(cnn_channels):
            in_channels = num_features if i == 0 else cnn_channels[i-1]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # 多尺度卷积分支
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.Conv1d(num_features, 32, kernel_size=5, padding=2),
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
        ])
        
        # LSTM层
        lstm_input_size = cnn_channels[-1] + 32 * 3  # CNN输出 + 多尺度特征
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_len * num_targets)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 转换为CNN输入格式 (batch, channels, length)
        x_cnn = x.permute(0, 2, 1)
        
        # CNN特征提取
        cnn_out = x_cnn
        for conv in self.conv_layers:
            cnn_out = conv(cnn_out)
        
        # 多尺度特征
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            feat = F.relu(conv(x_cnn))
            multi_scale_features.append(feat)
        multi_scale_out = torch.cat(multi_scale_features, dim=1)
        
        # 合并特征
        combined = torch.cat([cnn_out, multi_scale_out], dim=1)
        combined = combined.permute(0, 2, 1)  # (batch, length, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(combined)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 输出
        output = self.fc(context)
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class Chomp1d(nn.Module):
    """用于因果卷积的裁剪层 - 优化版本"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """TCN的基本模块 - 优化版本"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 使用更高效的网络结构
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(n_outputs),  # 添加 BatchNorm 加速收敛
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(n_outputs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.chomp = Chomp1d(padding * 2)  # 两层卷积，需要裁剪两倍
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.net(x)
        # 裁剪以保持因果性
        out = self.chomp(out)
        res = x if self.downsample is None else self.downsample(x)
        # 确保维度匹配
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    
    创新点：
    1. 因果卷积保证时序性
    2. 膨胀卷积扩大感受野
    3. 残差连接稳定训练
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=TCN_CONFIG):
        super(TCNModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        num_channels = config['num_channels']
        kernel_size = config['kernel_size']
        dropout = config['dropout']
        
        # 构建TCN层
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))
        
        self.tcn = nn.Sequential(*layers)
        
        # 全局平均池化 + 注意力
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.Tanh(),
            nn.Linear(num_channels[-1] // 2, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_len * num_targets)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 转换为Conv1d输入格式
        x = x.permute(0, 2, 1)
        
        # TCN
        tcn_out = self.tcn(x)
        
        # 转换回来
        tcn_out = tcn_out.permute(0, 2, 1)
        
        # 注意力
        attention_weights = self.attention(tcn_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * tcn_out, dim=1)
        
        # 输出
        output = self.fc(context)
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class EnsembleModel(nn.Module):
    """
    集成学习模型
    
    创新点：
    1. 多模型集成提高鲁棒性
    2. 可学习的权重分配
    3. 不同模型捕获不同特征
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, 
                 models=None, config=ENSEMBLE_CONFIG):
        super(EnsembleModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 如果没有提供预训练模型，创建新模型
        if models is None:
            from models import LinearModel, LSTMModel, TransformerModel
            self.base_models = nn.ModuleList([
                LinearModel(input_len, output_len, num_features, num_targets),
                LSTMModel(input_len, output_len, num_features, num_targets),
                TransformerModel(input_len, output_len, num_features, num_targets),
            ])
        else:
            self.base_models = nn.ModuleList(models)
        
        num_models = len(self.base_models)
        
        # 可学习的权重
        if config['weights'] == 'learned':
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        elif config['weights'] == 'stacking':
            # 使用元学习器
            self.meta_learner = nn.Sequential(
                nn.Linear(output_len * num_targets * num_models, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_len * num_targets)
            )
            self.weights = None
        else:
            # 等权重
            self.register_buffer('weights', torch.ones(num_models) / num_models)
        
        self.weight_type = config['weights']
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 获取所有基模型的预测
        outputs = []
        for model in self.base_models:
            out = model(x)
            outputs.append(out)
        
        if self.weight_type == 'stacking':
            # Stacking：将所有输出拼接后通过元学习器
            stacked = torch.cat([o.view(batch_size, -1) for o in outputs], dim=1)
            output = self.meta_learner(stacked)
            output = output.view(batch_size, self.output_len, self.num_targets)
        else:
            # 加权平均
            weights = F.softmax(self.weights, dim=0) if isinstance(self.weights, nn.Parameter) else self.weights
            output = sum(w * o for w, o in zip(weights, outputs))
        
        return output


class WaveNetBlock(nn.Module):
    """WaveNet风格的残差块 - 优化版本"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        
        # 使用更简单的 padding 策略
        self.padding = (kernel_size - 1) * dilation
        
        self.dilated_conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size,
                                       padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.res_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)
        
        self.out_channels = out_channels
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length)
        """
        conv_out = self.dilated_conv(x)
        # 裁剪多余的padding以保持因果性
        if self.padding > 0:
            conv_out = conv_out[:, :, :-self.padding]
        conv_out = self.bn(conv_out)
        
        # 门控激活 - 使用 chunk 更高效
        filter_out, gate_out = conv_out.chunk(2, dim=1)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        
        # 残差连接
        res = self.res_conv(gated) + x
        
        # Skip连接
        skip = self.skip_conv(gated)
        
        return res, skip


class WaveNetModel(nn.Module):
    """
    WaveNet风格的序列预测模型
    
    创新点：
    1. 门控激活单元
    2. 膨胀因果卷积
    3. 残差和Skip连接
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, 
                 num_channels=64, num_blocks=8, kernel_size=2, dropout=0.2):
        super(WaveNetModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 输入投影
        self.input_conv = nn.Conv1d(num_features, num_channels, 1)
        
        # WaveNet块
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)
            self.blocks.append(WaveNetBlock(num_channels, num_channels, kernel_size, dilation))
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.fc = nn.Linear(num_channels * input_len, output_len * num_targets)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 转换格式
        x = x.permute(0, 2, 1)
        
        # 输入投影
        x = self.input_conv(x)
        
        # WaveNet块
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        
        # 输出处理
        out = self.output_layers(skip_sum)
        out = out.view(batch_size, -1)
        
        # 全连接
        output = self.fc(out)
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


# ==================== LSTNet 模型 ====================

class LSTNetModel(nn.Module):
    """
    LSTNet (Long- and Short-term Time-series Network)
    
    创新点：
    1. CNN层提取短期局部模式
    2. GRU层捕获长期依赖
    3. Skip-GRU层直接建模周期性模式
    4. 自回归组件增强预测稳定性
    5. 参数量适中，适合中小规模数据集
    
    参考论文：
    Lai et al. "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (SIGIR 2018)
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=None):
        """
        Args:
            input_len: 输入序列长度
            output_len: 输出序列长度
            num_features: 输入特征数量
            num_targets: 预测目标数量
            config: 模型配置字典
        """
        super(LSTNetModel, self).__init__()
        
        # 延迟导入避免循环依赖
        if config is None:
            from config import LSTNET_CONFIG
            config = LSTNET_CONFIG
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        # 配置参数
        self.cnn_channels = config.get('cnn_channels', 32)
        self.cnn_kernel = config.get('cnn_kernel', 3)
        self.rnn_hidden = config.get('rnn_hidden', 64)
        self.skip_hidden = config.get('skip_hidden', 32)
        self.skip = config.get('skip', 4)  # 跳跃步长（周期）
        self.highway_window = config.get('highway_window', 4)  # 自回归窗口
        self.dropout = config.get('dropout', 0.2)
        
        # 1. CNN层 - 提取短期局部模式
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, self.cnn_channels, self.cnn_kernel, padding=self.cnn_kernel//2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 2. GRU层 - 捕获长期依赖
        self.gru = nn.GRU(
            input_size=self.cnn_channels,
            hidden_size=self.rnn_hidden,
            batch_first=True
        )
        
        # 3. Skip-GRU层 - 直接建模周期性（如果序列足够长）
        self.skip_steps = max(1, input_len // self.skip)
        if self.skip_steps > 0 and self.skip > 1:
            self.skip_gru = nn.GRU(
                input_size=self.cnn_channels,
                hidden_size=self.skip_hidden,
                batch_first=True
            )
            self.skip_linear = nn.Linear(self.skip_hidden, self.rnn_hidden)
        else:
            self.skip_gru = None
            self.skip_linear = None
        
        # 4. 输出层
        self.linear_out = nn.Sequential(
            nn.Linear(self.rnn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, output_len * num_targets)
        )
        
        # 5. 自回归组件（Highway）- 增强预测稳定性
        self.highway_window = min(self.highway_window, input_len)  # 确保不超过输入长度
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window * num_features, output_len * num_targets)
        else:
            self.highway = None
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, input_len, num_features)
        
        Returns:
            output: (batch_size, output_len, num_targets)
        """
        batch_size = x.size(0)
        
        # 1. CNN层
        c = x.permute(0, 2, 1)  # (batch, features, seq_len)
        c = self.conv(c)  # (batch, cnn_channels, seq_len)
        c = c.permute(0, 2, 1)  # (batch, seq_len, cnn_channels)
        
        # 2. GRU层
        gru_out, _ = self.gru(c)  # (batch, seq_len, rnn_hidden)
        gru_last = gru_out[:, -1, :]  # (batch, rnn_hidden)
        
        # 3. Skip-GRU层（如果启用）
        if self.skip_gru is not None and self.skip_linear is not None:
            # 选择每隔skip步的时间点
            skip_indices = list(range(0, c.size(1), self.skip))[-self.skip_steps:]
            if len(skip_indices) > 0:
                skip_c = c[:, skip_indices, :]  # (batch, skip_steps, cnn_channels)
                skip_out, _ = self.skip_gru(skip_c)
                skip_last = skip_out[:, -1, :]  # (batch, skip_hidden)
                skip_feat = self.skip_linear(skip_last)  # (batch, rnn_hidden)
                gru_last = gru_last + skip_feat
        
        # 4. 线性输出
        out = self.linear_out(gru_last)  # (batch, output_len * num_targets)
        
        # 5. 自回归组件
        if self.highway is not None:
            hw_input = x[:, -self.highway_window:, :].contiguous()
            hw_input = hw_input.view(batch_size, -1)
            hw_out = self.highway(hw_input)
            out = out + hw_out
        
        # 重塑为目标形状
        output = out.view(batch_size, self.output_len, self.num_targets)
        
        return output


def get_innovative_model(model_name, input_len, output_len, num_features, num_targets):
    """
    获取创新模型实例
    
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
        'CNN_LSTM': CNNLSTMModel,
        'TCN': TCNModel,
        'Ensemble': EnsembleModel,
        'WaveNet': WaveNetModel,
        'LSTNet': LSTNetModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](input_len, output_len, num_features, num_targets)


if __name__ == "__main__":
    # 测试创新模型
    print("测试创新模型定义...")
    
    batch_size = 32
    input_len = 8
    output_len = 16
    num_features = 20
    num_targets = 3
    
    x = torch.randn(batch_size, input_len, num_features)
    
    for model_name in ['CNN_LSTM', 'TCN', 'WaveNet', 'LSTNet']:
        print(f"\n测试 {model_name} 模型:")
        model = get_innovative_model(model_name, input_len, output_len, num_features, num_targets)
        
        output = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数数量: {num_params:,}")
    
    print("\n创新模型测试完成！")
