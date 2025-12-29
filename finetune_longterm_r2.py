"""
长期预测模型精细微调脚本 - 使用R²作为最优判断标准
目标：针对 multistep_16h 任务进行精细化调优

关键区别：
1. 使用 R² 而非 MSE 作为早停判断标准（mode='max'，越大越好）
2. 更低的学习率，更长的训练周期
3. 更宽松的早停条件

为什么16h任务用R²更合理？
- 16步预测的MSE天然很大（3.5-5.2），波动剧烈
- R²表示模型解释的方差比例，范围0-1，更稳定
- R²能更好反映模型的"相对预测能力"

用法：
    python finetune_longterm_r2.py                     # 微调所有模型
    python finetune_longterm_r2.py --models LSTM       # 仅微调指定模型
    python finetune_longterm_r2.py --lr 0.0001         # 自定义学习率
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import r2_score

# 设置无头模式
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import matplotlib
    matplotlib.use('Agg')

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, set_seed, RANDOM_SEED, WEIGHT_DECAY,
    MULTI_STEP_2_INPUT_LEN, MULTI_STEP_2_OUTPUT_LEN,
)
from data_loader import load_all_data, preprocess_data, create_dataloaders
from models import get_model, count_parameters
from models_innovative import get_innovative_model
from trainer import calculate_metrics, test_model, print_test_results, load_model


# ==================== 长期任务微调超参 ====================
FINETUNE_CONFIG = {
    'lr': 0.0002,              # 非常低的学习率
    'epochs': 120,             # 充足的训练轮数
    'patience': 35,            # 非常宽松的早停
    'warmup_epochs': 10,       # 学习率预热
    'lr_min_factor': 0.01,     # 最小学习率比例
    'description': '长期预测微调 (24h→16h) - R²优化',
}

# 所有可用模型
ALL_MODELS = ['Linear', 'LSTM', 'Transformer', 'CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']
INNOVATIVE_MODELS = ['CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']


class EarlyStoppingR2:
    """基于R²的早停机制（mode='max'，越大越好）"""
    
    def __init__(self, patience=30, min_delta=0.001):
        """
        初始化早停
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_r2 = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, r2, model):
        if self.best_r2 is None:
            self.best_r2 = r2
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif r2 > self.best_r2 + self.min_delta:
            # R²提升了
            self.best_r2 = r2
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def load_best_model(self, model):
        """加载最佳模型状态"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_with_r2_criterion(model, train_loader, val_loader, model_name, task_name,
                            num_epochs, learning_rate, patience, device=DEVICE,
                            save_best=True, verbose=True, resume=False):
    """
    使用R²作为最优判断标准的训练函数
    
    关键区别：
    - 早停基于R²（越大越好）
    - 同时记录MSE和R²
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=learning_rate * FINETUNE_CONFIG['lr_min_factor']
    )
    
    # 使用R²早停（mode='max'）
    early_stopping = EarlyStoppingR2(patience=patience, min_delta=0.001)
    
    # 尝试加载已有模型
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{task_name}.pth")
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_metrics': [],
        'best_epoch': 0,
        'best_val_r2': -float('inf'),
        'best_val_loss': float('inf'),
    }
    
    if resume and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            checkpoint_state = checkpoint['model_state_dict']
            current_state = model.state_dict()
            
            # 检查兼容性
            compatible = all(
                key in checkpoint_state and current_state[key].shape == checkpoint_state[key].shape
                for key in current_state.keys()
            )
            
            if compatible:
                model.load_state_dict(checkpoint_state)
                prev_history = checkpoint.get('history', {})
                
                # 恢复历史
                history['train_loss'] = prev_history.get('train_loss', [])
                history['val_loss'] = prev_history.get('val_loss', [])
                history['val_r2'] = prev_history.get('val_r2', [])
                history['val_metrics'] = prev_history.get('val_metrics', [])
                
                # 关键：获取历史最佳R²
                if 'val_metrics' in prev_history and len(prev_history['val_metrics']) > 0:
                    historical_r2_values = [m.get('R2', -1) for m in prev_history['val_metrics']]
                    history['best_val_r2'] = max(historical_r2_values) if historical_r2_values else -float('inf')
                    history['best_val_loss'] = prev_history.get('best_val_loss', float('inf'))
                    history['best_epoch'] = prev_history.get('best_epoch', 0)
                
                start_epoch = len(history['train_loss'])
                if verbose:
                    print(f"✅ 从检查点恢复: 已完成 {start_epoch} epochs")
                    print(f"   历史最佳 R²: {history['best_val_r2']:.4f}")
            else:
                if verbose:
                    print("⚠️ 模型结构不兼容，从头训练")
        except Exception as e:
            if verbose:
                print(f"⚠️ 无法加载检查点: {e}")
    
    remaining_epochs = max(0, num_epochs - start_epoch)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"训练 {model_name} - {task_name} (R²优化)")
        print(f"{'='*60}")
        print(f"设备: {device}")
        print(f"学习率: {learning_rate}")
        print(f"剩余轮数: {remaining_epochs}")
        print(f"早停耐心: {patience} (基于R²)")
    
    if remaining_epochs == 0:
        if verbose:
            print("✅ 已完成指定轮数训练")
        return history
    
    progress_bar = tqdm(range(remaining_epochs), desc=f"Training {model_name}")
    
    for epoch_idx in progress_bar:
        actual_epoch = start_epoch + epoch_idx
        
        # 训练
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
                all_preds.append(output.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算R²
        val_metrics = calculate_metrics(all_targets, all_preds)
        val_r2 = val_metrics['R2']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_metrics'].append(val_metrics)
        
        # 更新最佳记录（基于R²）
        if val_r2 > history['best_val_r2']:
            history['best_val_r2'] = val_r2
            history['best_val_loss'] = val_loss
            history['best_epoch'] = actual_epoch + 1
        
        # 学习率调度
        scheduler.step(actual_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新进度条
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_R²': f'{val_r2:.4f}',
            'best_R²': f'{history["best_val_r2"]:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # R²早停检查
        early_stopping(val_r2, model)
        if early_stopping.early_stop:
            if verbose:
                print(f"\n⏹️ 早停触发于 epoch {actual_epoch + 1} (R²连续{patience}轮未改进)")
            break
    
    # 加载最佳模型
    early_stopping.load_best_model(model)
    
    # 保存模型
    if save_best:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'task_name': task_name,
            'history': history,
            'total_epochs': len(history['train_loss']),
            'optimization_target': 'R2',  # 标记优化目标
        }, model_path)
        if verbose:
            print(f"✅ 模型已保存 (最佳R²: {history['best_val_r2']:.4f})")
    
    return history


def finetune_model(model_name, task_name, df, config, batch_size=64, verbose=True):
    """微调单个模型"""
    input_len = MULTI_STEP_2_INPUT_LEN
    output_len = MULTI_STEP_2_OUTPUT_LEN
    
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
    
    # 使用R²优化的训练函数
    history = train_with_r2_criterion(
        model, train_loader, val_loader,
        model_name=model_name,
        task_name=task_name,
        num_epochs=config['epochs'],
        learning_rate=config['lr'],
        patience=config['patience'],
        device=DEVICE,
        save_best=True,
        verbose=verbose,
        resume=True
    )
    
    # 测试
    metrics, metrics_per_target, predictions, targets = test_model(
        model, test_loader, scaler_targets, device=DEVICE
    )
    
    if verbose:
        print_test_results(model_name, task_name, metrics, metrics_per_target, target_cols)
    
    return metrics, history


def main():
    parser = argparse.ArgumentParser(description='长期预测模型精细微调 (R²优化)')
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
    print("🎯 长期预测模型精细微调 (R²优化)")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 关键区别: 使用R²(而非MSE)作为最优判断标准")
    print(f"   原因: 16h长期预测的MSE波动大，R²更稳定")
    
    # 加载数据
    print("\n📊 加载数据...")
    raw_df = load_all_data()
    df = preprocess_data(raw_df)
    print(f"数据形状: {df.shape}")
    
    # 确定要微调的模型
    models = args.models if args.models else ALL_MODELS
    
    # 验证模型名称
    invalid_models = [m for m in models if m not in ALL_MODELS]
    if invalid_models:
        print(f"⚠️ 未知模型: {invalid_models}")
        models = [m for m in models if m in ALL_MODELS]
    
    # 配置
    config = FINETUNE_CONFIG.copy()
    if args.lr is not None:
        config['lr'] = args.lr
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.patience is not None:
        config['patience'] = args.patience
    
    print(f"\n📋 微调计划:")
    print(f"  任务: multistep_16h ({MULTI_STEP_2_INPUT_LEN}h → {MULTI_STEP_2_OUTPUT_LEN}h)")
    print(f"  模型: {models}")
    print(f"  超参: lr={config['lr']}, epochs={config['epochs']}, patience={config['patience']}")
    
    # 微调
    all_results = {}
    
    for model_name in models:
        try:
            metrics, history = finetune_model(
                model_name, 'multistep_16h', df, config,
                batch_size=args.batch_size,
                verbose=not args.quiet
            )
            all_results[model_name] = {
                'metrics': metrics,
                'history': history
            }
        except Exception as e:
            print(f"❌ 微调 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 结果汇总
    print("\n" + "=" * 70)
    print("📊 微调结果汇总 (按R²排序)")
    print("=" * 70)
    print(f"\n【{config['description']}】")
    print("-" * 60)
    print(f"{'模型':<20} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 60)
    
    # 按R²排序
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['metrics']['R2'],
        reverse=True
    )
    
    for i, (model_name, result) in enumerate(sorted_results):
        m = result['metrics']
        medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        print(f"{medal}{model_name:<18} {m['MSE']:>10.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 长期预测微调完成！")
    print(f"   模型已保存至: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
