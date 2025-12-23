@echo off
REM =================================================================
REM Windows本地训练脚本 (使用 lxy6032 conda环境)
REM 使用方法: 双击运行 或 在命令行执行 train_local.bat
REM =================================================================

echo ========================================
echo 风速序列预测 - 本地训练脚本
echo 使用conda环境: lxy6032
echo ========================================

REM 激活conda环境
call conda activate lxy6032

REM 检查Python环境
echo.
echo Python环境信息:
python --version

REM 检查PyTorch
echo.
echo PyTorch配置:
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

REM 开始训练
echo.
echo ========================================
echo 开始训练...
echo ========================================

python main.py

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo 模型保存位置: .\models\
echo 结果保存位置: .\results\

pause
