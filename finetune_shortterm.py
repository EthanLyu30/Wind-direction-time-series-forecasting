"""
短期预测模型精细微调脚本
目标：针对 singlestep 和 multistep_1h 任务进行精细化调优

策略：
1. 使用较小的学习率进行微调
2. 增加训练轮数，降低早停敏感度
3. 使用MSE作为判断标准（短期预测MSE敏感）
4. 添加学习率预热和更精细的调度

用法：
    python finetune_shortterm.py                          # 微调所有短期任务的所有模型
    python finetune_shortterm.py --task singlestep        # 仅微调singlestep
    python finetune_shortterm.py --models LSTM Transformer  # 仅微调指定模型
    python finetune_shortterm.py --lr 0.0001 --epochs 80  # 自定义超参
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 设置无头模式
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import matplotlib
    matplotlib.use('Agg')

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, set_seed, RANDOM_SEED,
    SINGLE_STEP_INPUT_LEN, SINGLE_STEP_OUTPUT_LEN,
    MULTI_STEP_1_INPUT_LEN, MULTI_STEP_1_OUTPUT_LEN,
)
from data_loader import load_all_data, preprocess_data, create_dataloaders
from models import get_model, count_parameters
from models_innovative import get_innovative_model
from trainer import train_model, test_model, print_test_results, load_model


# ==================== 短期任务微调超参 ====================
# 这些参数专门针对短期预测优化
FINETUNE_CONFIG = {
    'singlestep': {
        'lr': 0.0003,           # 较低学习率，精细调整
        'epochs': 80,           # 足够的训练轮数
        'patience': 25,         # 宽松早停，允许更多探索
        'warmup_epochs': 5,     # 学习率预热
        'description': '单步预测微调 (8h→1h)',
    },
    'multistep_1h': {
        'lr': 0.00025,          # 更低的学习率
        'epochs': 100,          # 更多训练轮数
        'patience': 30,         # 更宽松的早停
        'warmup_epochs': 8,     # 更长的预热
        'description': '多步1h预测微调 (8h→1h)',
    }
}

# 所有可用模型
ALL_MODELS = ['Linear', 'LSTM', 'Transformer', 'CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']
INNOVATIVE_MODELS = ['CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']


def get_task_config(task_name):
    """获取任务配置"""
    if task_name == 'singlestep':
        return SINGLE_STEP_INPUT_LEN, SINGLE_STEP_OUTPUT_LEN
    elif task_name == 'multistep_1h':
        return MULTI_STEP_1_INPUT_LEN, MULTI_STEP_1_OUTPUT_LEN
    else:
        raise ValueError(f"此脚本仅支持短期任务: singlestep, multistep_1h")


def finetune_model(model_name, task_name, df, config, batch_size=64, verbose=True):
    """
    微调单个模型
    
    Args:
        model_name: 模型名称
        task_name: 任务名称
        df: 预处理后的数据
        config: 微调配置
        batch_size: 批次大小
        verbose: 是否打印详细信息
    
    Returns:
        测试指标, 训练历史
    """
    input_len, output_len = get_task_config(task_name)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler_features, scaler_targets, feature_cols, target_cols = \
        create_dataloaders(df, input_len, output_len, batch_size)
    
    num_features = len(feature_cols)
    num_targets = len(target_cols)
    
    # 创建模型
    is_innovative = model_name in INNOVATIVE_MODELS
    if is_innovative:
        model = get_innovative_model(model_name, input_len, output_len, num_features, num_targets)
    else:
        model = get_model(model_name, input_len, output_len, num_features, num_targets)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"微调 {model_name} - {config['description']}")
        print(f"{'='*60}")
        print(f"参数量: {count_parameters(model):,}")
        print(f"学习率: {config['lr']}")
        print(f"训练轮数: {config['epochs']}")
        print(f"早停耐心: {config['patience']}")
    
    # 尝试加载已有模型作为起点
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{task_name}.pth")
    if os.path.exists(model_path):
        try:
            model, prev_history = load_model(model, model_path)
            if verbose:
                best_loss = prev_history.get('best_val_loss', 'N/A')
                if isinstance(best_loss, (int, float)):
                    print(f"✅ 加载已有模型，之前最佳损失: {best_loss:.4f}")
                else:
                    print(f"✅ 加载已有模型")
        except Exception as e:
            if verbose:
                print(f"⚠️ 无法加载已有模型，从头训练: {e}")
    
    # 微调训练
    history = train_model(
        model, train_loader, val_loader,
        model_name=model_name,
        task_name=task_name,
        num_epochs=config['epochs'],
        learning_rate=config['lr'],
        patience=config['patience'],
        device=DEVICE,
        save_best=True,
        verbose=verbose,
        resume=True  # 始终尝试继续训练
    )
    
    # 测试
    metrics, metrics_per_target, predictions, targets = test_model(
        model, test_loader, scaler_targets, device=DEVICE
    )
    
    if verbose:
        print_test_results(model_name, task_name, metrics, metrics_per_target, target_cols)
    
    return metrics, history


def main():
    parser = argparse.ArgumentParser(description='短期预测模型精细微调')
    parser.add_argument('--task', type=str, default=None, choices=['singlestep', 'multistep_1h'],
                        help='指定要微调的任务（不指定则全部微调）')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='指定要微调的模型（不指定则全部微调）')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖默认学习率')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认训练轮数')
    parser.add_argument('--patience', type=int, default=None,
                        help='覆盖默认早停耐心值')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    print("=" * 70)
    print("🔧 短期预测模型精细微调")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    print("\n📊 加载数据...")
    raw_df = load_all_data()
    df = preprocess_data(raw_df)
    print(f"数据形状: {df.shape}")
    
    # 确定要微调的任务和模型
    tasks = [args.task] if args.task else ['singlestep', 'multistep_1h']
    models = args.models if args.models else ALL_MODELS
    
    # 验证模型名称
    invalid_models = [m for m in models if m not in ALL_MODELS]
    if invalid_models:
        print(f"⚠️ 未知模型: {invalid_models}")
        print(f"可用模型: {ALL_MODELS}")
        models = [m for m in models if m in ALL_MODELS]
    
    print(f"\n📋 微调计划:")
    print(f"  任务: {tasks}")
    print(f"  模型: {models}")
    
    # 微调结果收集
    all_results = {}
    
    for task_name in tasks:
        config = FINETUNE_CONFIG[task_name].copy()
        
        # 应用命令行覆盖
        if args.lr is not None:
            config['lr'] = args.lr
        if args.epochs is not None:
            config['epochs'] = args.epochs
        if args.patience is not None:
            config['patience'] = args.patience
        
        print(f"\n{'#'*70}")
        print(f"# 任务: {config['description']}")
        print(f"# 超参: lr={config['lr']}, epochs={config['epochs']}, patience={config['patience']}")
        print(f"{'#'*70}")
        
        task_results = {}
        
        for model_name in models:
            try:
                metrics, history = finetune_model(
                    model_name, task_name, df, config,
                    batch_size=args.batch_size,
                    verbose=not args.quiet
                )
                task_results[model_name] = {
                    'metrics': metrics,
                    'history': history
                }
            except Exception as e:
                print(f"❌ 微调 {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[task_name] = task_results
    
    # 打印最终结果汇总
    print("\n" + "=" * 70)
    print("📊 微调结果汇总")
    print("=" * 70)
    
    for task_name, task_results in all_results.items():
        print(f"\n【{FINETUNE_CONFIG[task_name]['description']}】")
        print("-" * 50)
        print(f"{'模型':<20} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print("-" * 50)
        
        # 按R²排序
        sorted_results = sorted(
            task_results.items(),
            key=lambda x: x[1]['metrics']['R2'],
            reverse=True
        )
        
        for i, (model_name, result) in enumerate(sorted_results):
            m = result['metrics']
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
            print(f"{medal}{model_name:<18} {m['MSE']:>10.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 微调完成！")
    print(f"   模型已保存至: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
