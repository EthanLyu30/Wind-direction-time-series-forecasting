#!/bin/bash
# =================================================================
# 远程服务器/GPU机器训练脚本 (Linux)
# 使用方法: bash train_remote.sh
# =================================================================

echo "========================================"
echo "风速序列预测 - 远程训练脚本"
echo "========================================"

# 检测是否有GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_AVAILABLE=1
else
    echo "未检测到GPU，将使用CPU训练"
    GPU_AVAILABLE=0
fi

# 检查Python环境
echo ""
echo "Python环境信息:"
python3 --version
pip3 --version

# 安装依赖（如果需要）
echo ""
echo "检查并安装依赖..."
pip3 install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip3 install -q torch torchvision

pip3 install -q pandas pyarrow numpy scikit-learn matplotlib seaborn tqdm

# 检查PyTorch CUDA支持
echo ""
echo "PyTorch配置:"
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CPU模式')"

# 开始训练
echo ""
echo "========================================"
echo "开始训练..."
echo "========================================"

# 运行主程序
python3 main.py

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"
echo "模型保存位置: ./models/"
echo "结果保存位置: ./results/"
