#!/bin/bash
# Git仓库初始化脚本
# 在项目根目录运行此脚本

echo "=== 初始化Git仓库 ==="

# 初始化git仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: 风速序列预测项目

功能特性:
- 支持单步预测和多步预测
- 实现Linear、LSTM、Transformer基础模型
- 实现CNN-LSTM、Attention-LSTM、TCN、WaveNet创新模型
- 完整的数据预处理和特征工程
- MSE、RMSE、MAE、R²评估指标
- 丰富的可视化功能
- 模型保存为pth格式"

echo ""
echo "=== Git仓库初始化完成 ==="
echo ""
echo "接下来请执行以下步骤:"
echo "1. 在GitHub上创建新仓库"
echo "2. 运行以下命令关联远程仓库:"
echo "   git remote add origin https://github.com/你的用户名/仓库名.git"
echo "   git branch -M main"
echo "   git push -u origin main"
