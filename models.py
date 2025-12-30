"""
模型定义模块
包含：Linear、LSTM、Transformer三个基础模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import LINEAR_CONFIG, LSTM_CONFIG, TRANSFORMER_CONFIG


class LinearModel(nn.Module):
    """
    基于多层感知机(MLP)的线性模型
    将输入序列展平后通过全连接层预测
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=LINEAR_CONFIG):
        """
        初始化Linear模型
        
        Args:
            input_len: 输入序列长度
            output_len: 输出序列长度
            num_features: 特征数量
            num_targets: 目标数量
            config: 模型配置
        """
        super(LinearModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        input_size = input_len * num_features
        output_size = output_len * num_targets
        
        hidden_sizes = config['hidden_sizes']
        dropout = config['dropout']
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_len, num_features)
            
        Returns:
            output: 输出张量 (batch_size, output_len, num_targets)
        """
        batch_size = x.size(0)
        
        # 展平输入
        x = x.view(batch_size, -1)
        
        # 通过网络
        output = self.network(x)
        
        # 重塑输出
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class LSTMModel(nn.Module):
    """
    基于LSTM的序列预测模型
    使用双向LSTM捕获时序特征
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=LSTM_CONFIG):
        """
        初始化LSTM模型
        
        Args:
            input_len: 输入序列长度
            output_len: 输出序列长度
            num_features: 特征数量
            num_targets: 目标数量
            config: 模型配置
        """
        super(LSTMModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout']
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, output_len * num_targets)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_len, num_features)
            
        Returns:
            output: 输出张量 (batch_size, output_len, num_targets)
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 拼接前向和后向的最后hidden state
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # 全连接层
        output = self.fc(h_final)
        
        # 重塑输出
        output = output.view(batch_size, self.output_len, self.num_targets)
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    基于Transformer的序列预测模型
    使用标准的Encoder-Decoder架构
    """
    
    def __init__(self, input_len, output_len, num_features, num_targets, config=TRANSFORMER_CONFIG):
        """
        初始化Transformer模型
        
        Args:
            input_len: 输入序列长度
            output_len: 输出序列长度
            num_features: 特征数量
            num_targets: 目标数量
            config: 模型配置
        """
        super(TransformerModel, self).__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.num_targets = num_targets
        
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        
        # 输入嵌入
        self.encoder_embedding = nn.Linear(num_features, self.d_model)
        self.decoder_embedding = nn.Linear(num_targets, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc_out = nn.Linear(self.d_model, num_targets)
    
    def generate_square_subsequent_mask(self, sz, device):
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x, tgt=None):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_len, num_features)
            tgt: 目标张量（用于训练时的teacher forcing）
            
        Returns:
            output: 输出张量 (batch_size, output_len, num_targets)
        """
        batch_size = x.size(0)
        device = x.device
        
        # 编码器输入
        enc_input = self.encoder_embedding(x)
        enc_input = self.pos_encoder(enc_input)
        
        # 解码器输入（使用零初始化）
        dec_input = torch.zeros(batch_size, self.output_len, self.num_targets, device=device)
        dec_input = self.decoder_embedding(dec_input)
        dec_input = self.pos_encoder(dec_input)
        
        # 生成掩码
        tgt_mask = self.generate_square_subsequent_mask(self.output_len, device)
        
        # Transformer前向传播
        transformer_out = self.transformer(enc_input, dec_input, tgt_mask=tgt_mask)
        
        # 输出层
        output = self.fc_out(transformer_out)
        
        return output


def get_model(model_name, input_len, output_len, num_features, num_targets):
    """
    根据模型名称获取模型实例
    
    Args:
        model_name: 模型名称 ('Linear', 'LSTM', 'Transformer')
        input_len: 输入序列长度
        output_len: 输出序列长度
        num_features: 特征数量
        num_targets: 目标数量
        
    Returns:
        模型实例
    """
    models = {
        'Linear': LinearModel,
        'LSTM': LSTMModel,
        'Transformer': TransformerModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](input_len, output_len, num_features, num_targets)


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("测试模型定义...")
    
    batch_size = 32
    input_len = 8
    output_len = 16
    num_features = 20
    num_targets = 3
    
    # 创建测试输入
    x = torch.randn(batch_size, input_len, num_features)
    
    # 测试每个模型
    for model_name in ['Linear', 'LSTM', 'Transformer']:
        print(f"\n测试 {model_name} 模型:")
        model = get_model(model_name, input_len, output_len, num_features, num_targets)
        
        output = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  参数数量: {count_parameters(model):,}")
    
    print("\n模型测试完成！")
