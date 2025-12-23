"""
GPU服务器专用训练脚本
- 针对云服务器GPU训练优化
- 更长的训练轮数
- 更大的batch size
- 支持断点续训
"""
import os
import sys
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='GPU服务器训练脚本')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--models', type=str, default='all', 
                        help='训练的模型,逗号分隔: Linear,LSTM,Transformer,CNN_LSTM,Attention_LSTM,TCN,WaveNet')
    parser.add_argument('--tasks', type=str, default='all',
                        help='训练任务: singlestep,multistep_1h,multistep_16h')
    args = parser.parse_args()
    
    # 动态修改config
    import config
    config.NUM_EPOCHS = args.epochs
    config.EARLY_STOPPING_PATIENCE = args.patience
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    print("=" * 60)
    print("GPU训练配置")
    print("=" * 60)
    print(f"📍 设备: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"📊 Batch Size: {args.batch_size}")
    print(f"🔄 训练轮数: {args.epochs}")
    print(f"⏱️  早停耐心值: {args.patience}")
    print(f"📈 学习率: {args.lr}")
    print("=" * 60)
    
    # 解析模型和任务
    if args.models == 'all':
        models = ['Linear', 'LSTM', 'Transformer', 'CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']
    else:
        models = args.models.split(',')
    
    if args.tasks == 'all':
        tasks = ['singlestep', 'multistep_1h', 'multistep_16h']
    else:
        tasks = args.tasks.split(',')
    
    print(f"🤖 模型: {models}")
    print(f"📝 任务: {tasks}")
    print("=" * 60)
    
    # 运行主程序
    from main import main as run_main
    
    class Args:
        def __init__(self):
            self.skip_data_viz = True  # 跳过数据可视化节省时间
            self.quick_test = False
    
    run_main(Args())

if __name__ == '__main__':
    main()
