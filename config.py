"""
配置文件：定义所有超参数和路径配置
"""
import os
import torch

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# 创建必要的目录
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== 数据配置 ====================
# 数据集路径
DATA_PATHS = {
    '10m': os.path.join(DATASET_DIR, 'WindSpeed_10m', 'data'),
    '50m': os.path.join(DATASET_DIR, 'WindSpeed_50m', 'data'),
    '100m': os.path.join(DATASET_DIR, 'WindSpeed_100m', 'data'),
}

# 特征列名（原始数据集中的列名）
FEATURE_COLS = [
    'DirectionAvg',      # 风向
    'TemperatureAvg',    # 温度
    'PressureAvg',       # 气压
    'HumidtyAvg',        # 湿度
]

# 目标列（我们要预测的）
TARGET_COL = 'SpeedAvg'  # 风速

# 时间戳列
TIMESTAMP_COL = 'Date & Time Stamp'

# 高度列
HEIGHT_COL = 'height'

# ==================== 序列配置 ====================
# 单步预测配置
SINGLE_STEP_INPUT_LEN = 8   # 输入序列长度（8小时）
SINGLE_STEP_OUTPUT_LEN = 1  # 输出序列长度（1小时）

# 多步预测配置 - 任务1：8小时预测1小时
MULTI_STEP_1_INPUT_LEN = 8
MULTI_STEP_1_OUTPUT_LEN = 1

# 多步预测配置 - 任务2：8小时预测16小时
MULTI_STEP_2_INPUT_LEN = 8
MULTI_STEP_2_OUTPUT_LEN = 16

# ==================== 数据集划分配置 ====================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ==================== 训练配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5

# ==================== 模型配置 ====================
# Linear模型配置
LINEAR_CONFIG = {
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.2,
}

# LSTM模型配置
LSTM_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': True,
}

# Transformer模型配置
TRANSFORMER_CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dim_feedforward': 256,
    'dropout': 0.1,
}

# ==================== 创新模型配置 ====================
# CNN-LSTM混合模型配置
CNN_LSTM_CONFIG = {
    'cnn_channels': [32, 64],
    'kernel_size': 3,
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'dropout': 0.2,
}

# Attention-LSTM模型配置
ATTENTION_LSTM_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'attention_heads': 4,
    'dropout': 0.2,
}

# TCN (Temporal Convolutional Network) 配置
TCN_CONFIG = {
    'num_channels': [64, 128, 256],
    'kernel_size': 3,
    'dropout': 0.2,
}

# 集成模型配置
ENSEMBLE_CONFIG = {
    'models': ['Linear', 'LSTM', 'Transformer'],
    'weights': 'learned',  # 'equal', 'learned', 'stacking'
}

# ==================== 随机种子 ====================
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保可复现性"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
