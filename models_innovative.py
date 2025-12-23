"""
创新模型模块
包含：CNN-LSTM混合模型、Attention-LSTM模型、TCN模型、集成模型
用于获得额外的25%创新分数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import CNN_LSTM_CONFIG, ATTENTION_LSTM_CONFIG, TCN_CONFIG, ENSEMBLE_CONFIG


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


class AttentionLSTMModel(nn.Module):
    """
    Attention-LSTM模型
    
    创新点：
    1. 自注意力机制增强特征表示
    2. 时序注意力聚焦关键时间点
    3. 多头注意力并行处理
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=ATTENTION_LSTM_CONFIG):
        super(AttentionLSTMModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        attention_heads = config['attention_heads']
        dropout = config['dropout']
        
        # 输入投影
        self.input_proj = nn.Linear(num_features, hidden_size)
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 时序注意力
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 特征注意力
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 层归一化
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_len * num_targets)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_len, num_features)
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 自注意力
        attn_out, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm2(lstm_out)
        
        # 时序注意力
        temporal_weights = self.temporal_attention(lstm_out)
        temporal_weights = F.softmax(temporal_weights, dim=1)
        context = torch.sum(temporal_weights * lstm_out, dim=1)
        
        # 输出
        output = self.fc(context)
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class Chomp1d(nn.Module):
    """用于因果卷积的裁剪层"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN的基本模块"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
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
    """WaveNet风格的残差块"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        
        # 使用padding='same'风格，确保输出长度与输入一致
        padding = (kernel_size - 1) * dilation
        
        self.dilated_conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size,
                                       padding=padding, dilation=dilation)
        self.res_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.chomp_size = padding  # 需要裁剪的大小
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length)
        """
        conv_out = self.dilated_conv(x)
        # 裁剪多余的padding以保持因果性
        if self.chomp_size > 0:
            conv_out = conv_out[:, :, :-self.chomp_size]
        conv_out = self.bn(conv_out)
        
        # 门控激活
        tanh_out = torch.tanh(conv_out[:, :conv_out.size(1)//2, :])
        sigmoid_out = torch.sigmoid(conv_out[:, conv_out.size(1)//2:, :])
        gated = tanh_out * sigmoid_out
        
        # 残差连接
        res = self.res_conv(gated)
        res = res + x
        
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
        'Attention_LSTM': AttentionLSTMModel,
        'TCN': TCNModel,
        'Ensemble': EnsembleModel,
        'WaveNet': WaveNetModel,
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
    
    for model_name in ['CNN_LSTM', 'Attention_LSTM', 'TCN', 'Ensemble', 'WaveNet']:
        print(f"\n测试 {model_name} 模型:")
        model = get_innovative_model(model_name, input_len, output_len, num_features, num_targets)
        
        output = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数数量: {num_params:,}")
    
    print("\n创新模型测试完成！")
