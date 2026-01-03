"""
配置文件：定义所有超参数和路径配置
支持自动检测GPU，适配本地Windows和远程Linux服务器训练
"""
import os
import sys
# ==================== 解决服务器无图形界面问题（必须最先执行）====================
# 在无头Linux服务器上设置环境变量，避免Qt插件错误
if sys.platform.startswith('linux'):
    if not os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        # 设置matplotlib后端为Agg（如果还没设置）
        import matplotlib
        matplotlib.use('Agg')
import torch

# ==================== 设备自动检测 ====================
def get_device():
    """自动检测并返回最佳可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 检测到GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device
    else:
        print("💻 使用CPU训练（建议在GPU服务器上运行以加速）")
        return torch.device('cpu')

# ==================== 平台检测 ====================
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')
print(f"📍 运行平台: {'Windows' if IS_WINDOWS else 'Linux' if IS_LINUX else sys.platform}")

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
# 单步预测配置：8小时历史数据 → 预测1小时
SINGLE_STEP_INPUT_LEN = 8   # 输入序列长度（8小时）
SINGLE_STEP_OUTPUT_LEN = 1  # 输出序列长度（1小时）

# 多步预测配置：8小时历史数据 → 预测16小时
MULTI_STEP_INPUT_LEN = 8    # 输入序列长度（8小时）
MULTI_STEP_OUTPUT_LEN = 16  # 输出序列长度（16小时）

# ==================== 数据集划分配置 ====================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ==================== 训练配置 ====================
DEVICE = get_device()

# 根据设备自动调整batch_size
# GPU可以使用更大的batch_size加速训练
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # A100/H100等高端GPU的优化配置
    if 'a100' in gpu_name or 'h100' in gpu_name or gpu_memory >= 35:
        BATCH_SIZE = 512       # A100 40G可以使用很大的batch
        USE_AMP = True         # 启用混合精度训练
        NUM_WORKERS = 8        # 更多数据加载线程
    elif gpu_memory >= 20:     # RTX 3090/4090等
        BATCH_SIZE = 256
        USE_AMP = True
        NUM_WORKERS = 4
    elif gpu_memory >= 10:     # RTX 3060 12GB等
        BATCH_SIZE = 128
        USE_AMP = True
        NUM_WORKERS = 4
    elif gpu_memory >= 6:
        BATCH_SIZE = 64
        USE_AMP = False
        NUM_WORKERS = 2
    else:
        BATCH_SIZE = 32
        USE_AMP = False
        NUM_WORKERS = 2
else:
    BATCH_SIZE = 64  # CPU默认
    USE_AMP = False
    NUM_WORKERS = 0

LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5

# ==================== 任务特定的超参优化 ====================
# 不同任务需要不同的学习率和早停耐心值
TASK_SPECIFIC_HYPERPARAMS = {
    'singlestep': {
        'lr': 0.001,          # 短期预测：正常学习率
        'patience': 20,       # 标准早停
        'min_epochs': 50,     # 至少训练50个epoch
    },
    'multistep_16h': {
        'lr': 0.0001,         # 长期预测：更低学习率避免快速收敛到局部最优
        'patience': 40,       # 更宽松早停，允许更充分探索
        'min_epochs': 100,    # 至少训练100个epoch
    }
}

# 根据batch_size自动调整学习率（线性缩放）
def get_adjusted_lr(base_lr, batch_size):
    """
    根据batch_size调整学习率
    线性缩放法则: lr = base_lr * (batch_size / 128)
    """
    reference_batch = 128
    return base_lr * (batch_size / reference_batch)

# 注意：实际batch_size可能被命令行参数覆盖，这里只是默认值
# print(f"⚙️  默认 Batch Size: {BATCH_SIZE}")  # 移到main.py中打印实际使用的值

# ==================== 模型配置 ====================
# Linear模型配置
LINEAR_CONFIG = {
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.2,
}

# LSTM模型配置
LSTM_CONFIG = {
    'hidden_size': 256,      # 隐藏层大小
    'num_layers': 3,         # 层数
    'dropout': 0.3,          # dropout率
    'bidirectional': True,
}

# Transformer模型配置
TRANSFORMER_CONFIG = {
    'd_model': 128,            # 模型维度
    'nhead': 8,                # 注意力头数
    'num_encoder_layers': 3,   # 编码器层数
    'num_decoder_layers': 3,   # 解码器层数
    'dim_feedforward': 512,    # 前馈层维度
    'dropout': 0.2,            # dropout率
}

# ==================== 创新模型配置 ====================
# CNN-LSTM混合模型配置
CNN_LSTM_CONFIG = {
    'cnn_channels': [32, 64],      # CNN通道数
    'kernel_size': 3,
    'lstm_hidden_size': 64,        # LSTM隐藏层大小
    'lstm_num_layers': 2,          # LSTM层数
    'dropout': 0.3,                # dropout率
}

# TCN (Temporal Convolutional Network) 配置
TCN_CONFIG = {
    'num_channels': [32, 64, 64],  # 各层通道数
    'kernel_size': 3,
    'dropout': 0.3,                # dropout率
}

# WaveNet模型配置
WAVENET_CONFIG = {
    'num_channels': 64,            # 通道数
    'num_blocks': 8,               # 残差块数量
    'kernel_size': 2,              # 卷积核大小
    'dropout': 0.3,                # dropout率
}

# 集成模型配置
ENSEMBLE_CONFIG = {
    'models': ['Linear', 'LSTM', 'Transformer'],
    'weights': 'learned',  # 'equal', 'learned', 'stacking'
}

# LSTNet模型配置（轻量级，适合小数据集）
LSTNET_CONFIG = {
    'cnn_channels': 32,         # CNN通道数
    'cnn_kernel': 3,            # CNN卷积核大小
    'rnn_hidden': 64,           # GRU隐藏层
    'skip_hidden': 32,          # Skip-GRU隐藏层
    'skip': 4,                  # 跳跃步长（用于捕获周期性）
    'highway_window': 4,        # 自回归窗口
    'dropout': 0.2,             # dropout率
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
