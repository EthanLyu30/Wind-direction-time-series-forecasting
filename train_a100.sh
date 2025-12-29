#!/bin/bash
# =================================================================
# A100 40G GPU 优化训练脚本
# 分阶段微调所有模型
# =================================================================

echo "=============================================="
echo "A100 GPU 微调训练 - $(date)"
echo "=============================================="

# 检查GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ==================== 第1阶段：新模型LSTNet ====================
echo ""
echo ">>> 第1阶段：训练新模型 LSTNet"
echo "=============================================="
python3 main.py --models LSTNet \
    --batch-size 512 \
    --epochs 500 \
    --lr 0.001 \
    --patience 40 \
    --no-viz

# ==================== 第2阶段：创新模型多步预测优化 ====================
echo ""
echo ">>> 第2阶段：微调创新模型（多步预测）"
echo "=============================================="
python3 main.py --models CNN_LSTM TCN WaveNet \
    --tasks multistep_16h \
    --batch-size 512 \
    --epochs 1500 \
    --lr 0.0001 \
    --patience 60 \
    --metric-mode r2 \
    --resume \
    --no-viz

# ==================== 第3阶段：创新模型单步预测优化 ====================
echo ""
echo ">>> 第3阶段：微调创新模型（单步预测）"
echo "=============================================="
python3 main.py --models CNN_LSTM WaveNet LSTNet \
    --tasks singlestep \
    --batch-size 512 \
    --epochs 2000 \
    --lr 0.0002 \
    --patience 50 \
    --resume \
    --no-viz

# ==================== 第4阶段：基础模型微调 ====================
echo ""
echo ">>> 第4阶段：微调基础模型"
echo "=============================================="
python3 main.py --models Linear LSTM Transformer \
    --batch-size 512 \
    --epochs 1500 \
    --lr 0.0002 \
    --patience 50 \
    --resume \
    --no-viz

# ==================== 生成最终报告和可视化 ====================
echo ""
echo ">>> 生成最终报告和可视化"
echo "=============================================="
python3 main.py --mode visualize

echo ""
echo "=============================================="
echo "训练完成！$(date)"
echo "结果保存在: results/"
echo "模型保存在: models/"
echo "=============================================="
