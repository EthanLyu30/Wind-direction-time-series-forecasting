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
# 单步预测配置
SINGLE_STEP_INPUT_LEN = 8   # 输入序列长度（8小时）
SINGLE_STEP_OUTPUT_LEN = 1  # 输出序列长度（1小时）

# 多步预测配置 - 任务1：8小时预测1小时
MULTI_STEP_1_INPUT_LEN = 8
MULTI_STEP_1_OUTPUT_LEN = 1

# 多步预测配置 - 任务2：8小时预测16小时（符合作业要求）
MULTI_STEP_2_INPUT_LEN = 8   # 作业要求：8小时
MULTI_STEP_2_OUTPUT_LEN = 16

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
    if gpu_memory >= 10:  # RTX 3060 有12GB显存
        BATCH_SIZE = 128
    elif gpu_memory >= 6:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 32
else:
    BATCH_SIZE = 64  # CPU默认

LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5

# ==================== 任务特定的超参优化 ====================
# 不同任务需要不同的学习率和早停耐心值
TASK_SPECIFIC_HYPERPARAMS = {
    'singlestep': {
        'lr': 0.001,          # 短期预测：正常学习率
        'patience': 15,       # 标准早停
        'min_epochs': 50,     # 至少训练50个epoch
    },
    'multistep_1h': {
        'lr': 0.0008,         # 多步1h：略降学习率
        'patience': 18,       # 略宽松
        'min_epochs': 60,
    },
    'multistep_16h': {
        'lr': 0.0003,         # 长期预测：显著降低学习率（关键！）
        'patience': 25,       # 宽松早停，允许更多探索
        'min_epochs': 80,     # 至少训练80个epoch
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

print(f"⚙️  Batch Size: {BATCH_SIZE}")

# ==================== 模型配置 ====================
# Linear模型配置
LINEAR_CONFIG = {
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.2,
}

# LSTM模型配置
LSTM_CONFIG = {
    'hidden_size': 256,      # 增大隐藏层（原128）
    'num_layers': 3,         # 增加层数（原2）
    'dropout': 0.3,          # 增加dropout防止过拟合
    'bidirectional': True,
}

# Transformer模型配置
TRANSFORMER_CONFIG = {
    'd_model': 128,            # 增大模型维度（原64）
    'nhead': 8,                # 增加注意力头数（原4）
    'num_encoder_layers': 4,   # 增加层数（原3）
    'num_decoder_layers': 4,   # 增加层数（原3）
    'dim_feedforward': 512,    # 增大前馈层（原256）
    'dropout': 0.2,            # 增加dropout
}

# ==================== 创新模型配置 ====================
# CNN-LSTM混合模型配置
CNN_LSTM_CONFIG = {
    'cnn_channels': [32, 32],      # 减少通道数（原[32,64]）
    'kernel_size': 3,
    'lstm_hidden_size': 64,        # 减少隐藏单元（原64，已合理）
    'lstm_num_layers': 2,          # 减少层数（原2，已合理）
    'dropout': 0.3,                # 增加dropout防止过拟合（原0.2）
}

# Attention-LSTM模型配置
ATTENTION_LSTM_CONFIG = {
    'hidden_size': 96,             # 减少隐藏单元（原128）
    'num_layers': 2,               # 保持2层（足够了）
    'attention_heads': 4,          # 减少头数（原4，已合理）
    'dropout': 0.3,                # 增加dropout（原0.2）
}

# TCN (Temporal Convolutional Network) 配置
# 优化版本：减少通道数以加快训练速度
TCN_CONFIG = {
    'num_channels': [32, 64, 64],  # 保持通道数（已优化）
    'kernel_size': 3,
    'dropout': 0.3,                # 增加dropout（原0.2）
}

# WaveNet模型配置
WAVENET_CONFIG = {
    'num_channels': 64,            # 减少通道数（原64，已合理）
    'num_blocks': 8,               # 保持块数（8个已足够）
    'kernel_size': 2,              # 保持卷积核（标准设置）
    'dropout': 0.3,                # 增加dropout防止过拟合
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
