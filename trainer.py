"""
训练与评估模块
功能：
1. 训练循环
2. 评估指标计算（MSE, RMSE, MAE, R²）
3. 早停机制
4. 模型保存与加载
"""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY, MODELS_DIR, LOGS_DIR
)


class EarlyStopping:
    """早停机制 - 使用R²作为选择标准"""

    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0.001, mode='max'):
        """
        初始化早停

        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量 (对于R²，0.001是合理的改进阈值)
            mode: 'max'表示R²越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 固定为'max'，因为R²越大越好
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_metrics = None  # 保存最佳的完整指标

    def __call__(self, metrics, model):
        """
        检查是否应该早停

        Args:
            metrics: 包含R2、MSE、RMSE、MAE的字典
            model: 当前模型
        """
        current_score = metrics['R2']  # 使用R²作为主要指标

        if self.best_score is None:
            # 第一次调用
            self.best_score = current_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_metrics = metrics.copy()
        elif self._is_improvement(current_score):
            # 有改进
            self.best_score = current_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_metrics = metrics.copy()
            self.counter = 0
        else:
            # 没有改进
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        """检查是否有足够改进"""
        return score > self.best_score + self.min_delta

    def load_best_model(self, model):
        """加载最佳模型状态"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

    def get_best_metrics(self):
        """获取最佳模型的指标"""
        return self.best_metrics


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实值 (numpy array)
        y_pred: 预测值 (numpy array)
        
    Returns:
        包含各指标的字典
    """
    # 确保是numpy数组
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def train_epoch(model, dataloader, criterion, optimizer, device=DEVICE):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, dataloader, criterion, device=DEVICE):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        平均损失, 所有预测值, 所有真实值
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, all_predictions, all_targets


def train_model(model, train_loader, val_loader, model_name, task_name,
                num_epochs=NUM_EPOCHS, learning_rate=None, patience=None,
                lr=None, device=DEVICE, save_best=True, verbose=True,
                resume=False):
    """
    完整的模型训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model_name: 模型名称
        task_name: 任务名称 ('singlestep', 'multistep_1h', 'multistep_16h')
        num_epochs: 训练轮数
        learning_rate: 学习率（新参数名）
        patience: 早停耐心值
        lr: 学习率（向后兼容）
        device: 设备
        save_best: 是否保存最佳模型
        verbose: 是否打印详细信息
        resume: 是否从检查点继续训练
        
    Returns:
        训练历史字典
    """
    # 处理学习率参数（优先使用learning_rate，然后是lr，最后是默认值）
    actual_lr = learning_rate if learning_rate is not None else (lr if lr is not None else LEARNING_RATE)
    # 处理早停耐心值
    actual_patience = patience if patience is not None else EARLY_STOPPING_PATIENCE
    
    # 尝试从检查点恢复
    start_epoch = 0
    previous_history = None
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{task_name}.pth")
    
    if resume and os.path.exists(model_path):
        try:
            # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含 numpy 数组的检查点
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # 检查模型结构是否兼容
            checkpoint_state = checkpoint['model_state_dict']
            current_state = model.state_dict()
            
            # 比较参数形状是否匹配
            compatible = True
            for key in current_state.keys():
                if key in checkpoint_state:
                    if current_state[key].shape != checkpoint_state[key].shape:
                        compatible = False
                        if verbose:
                            print(f"⚠️ 模型结构已更改，参数 '{key}' 形状不匹配:")
                            print(f"   检查点: {checkpoint_state[key].shape} → 当前模型: {current_state[key].shape}")
                        break
                else:
                    compatible = False
                    if verbose:
                        print(f"⚠️ 模型结构已更改，缺少参数: '{key}'")
                    break
            
            if compatible:
                model.load_state_dict(checkpoint_state)
                if 'history' in checkpoint:
                    previous_history = checkpoint['history']
                    # 从上次训练结束的epoch继续
                    start_epoch = len(previous_history.get('train_loss', []))
                if verbose:
                    prev_best_loss = previous_history.get('best_val_loss', 'N/A') if previous_history else 'N/A'
                    prev_best_epoch = previous_history.get('best_epoch', 'N/A') if previous_history else 'N/A'
                    # 格式化最佳损失值
                    loss_str = f"{prev_best_loss:.4f}" if isinstance(prev_best_loss, (int, float)) else str(prev_best_loss)
                    print(f"✅ 从检查点恢复: {model_path}")
                    print(f"   已完成 {start_epoch} 个epoch")
                    print(f"   之前最佳验证损失: {loss_str} (epoch {prev_best_epoch})")
            else:
                if verbose:
                    print(f"⚠️ 配置已更改，检查点不兼容，从头开始训练新模型")
                start_epoch = 0
                previous_history = None
                
        except Exception as e:
            if verbose:
                print(f"⚠️ 无法加载检查点，从头开始训练: {e}")
            start_epoch = 0
            previous_history = None
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr, weight_decay=WEIGHT_DECAY)
    
    # 使用 CosineAnnealingWarmRestarts 学习率调度器（更适合长期训练）
    # 结合 ReduceLROnPlateau 作为后备
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=actual_lr * 0.01
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    early_stopping = EarlyStopping(patience=actual_patience, mode='max')
    
    # 初始化历史记录（如果是继续训练，合并之前的历史）
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'training_time': 0
    }
    
    if previous_history is not None:
        history['train_loss'] = previous_history.get('train_loss', [])
        history['val_loss'] = previous_history.get('val_loss', [])
        history['val_metrics'] = previous_history.get('val_metrics', [])
        history['best_epoch'] = previous_history.get('best_epoch', 0)
        history['best_val_loss'] = previous_history.get('best_val_loss', float('inf'))  # 关键：保留历史最佳
        history['training_time'] = previous_history.get('training_time', 0)
        
        # 重要：记录历史最佳损失，用于后续对比
        history['_historical_best_val_loss'] = history['best_val_loss']
    
    start_time = time.time()
    
    # 计算剩余需要训练的轮数
    remaining_epochs = max(0, num_epochs - start_epoch)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"训练 {model_name} - {task_name}")
        print(f"{'='*60}")
        print(f"设备: {device}")
        print(f"学习率: {actual_lr}")
        if start_epoch > 0:
            print(f"继续训练: 从 epoch {start_epoch + 1} 到 {num_epochs}")
        else:
            print(f"训练轮数: {num_epochs}")
        print(f"早停耐心值: {actual_patience}")
    
    if remaining_epochs == 0:
        if verbose:
            print("✅ 模型已完成指定轮数的训练，无需继续")
        return history
    
    progress_bar = tqdm(range(remaining_epochs), desc=f"Training {model_name}")
    
    for epoch_idx in progress_bar:
        actual_epoch = start_epoch + epoch_idx
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        # 计算指标
        val_metrics = calculate_metrics(val_targets, val_preds)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # 更新学习率（使用余弦退火 + 平台检测双调度器）
        scheduler.step(actual_epoch)
        plateau_scheduler.step(val_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 警告：学习率过低导致训练停滞
        if current_lr < 1e-6 and actual_epoch > 20:
            if verbose and actual_epoch % 20 == 0:
                print(f"⚠️  警告：学习率过低 ({current_lr:.2e})，可能导致训练停滞，建议增加学习率")
        
        # 更新进度条
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_rmse': f'{val_metrics["RMSE"]:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # 检查是否为最佳模型（现在由EarlyStopping类内部处理）
        # 早停检查 - 传入完整指标，使用R²作为选择标准
        early_stopping(val_metrics, model)

        # 更新历史记录中的最佳信息
        if early_stopping.best_score is not None:
            history['best_val_loss'] = val_loss  # 仍然记录loss用于显示
            history['best_epoch'] = actual_epoch + 1
            history['best_r2'] = early_stopping.best_score  # 新增：记录最佳R²
        if early_stopping.early_stop:
            if verbose:
                print(f"\n早停触发于 epoch {actual_epoch + 1}")
            break
    
    # 加载本次训练的最佳模型（用于后续对比）
    early_stopping.load_best_model(model)
    current_best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    history['training_time'] = time.time() - start_time
    
    # ==================== 关键修复：对比新旧最佳模型，保留历史最佳 ====================
    model_filename = f"{model_name}_{task_name}.pth"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # 合并完整训练历史（无论是否改进都要合并）
    merged_history = history.copy()
    if previous_history is not None:
        # 合并历史记录
        merged_history['train_loss'] = previous_history.get('train_loss', []) + history['train_loss']
        merged_history['val_loss'] = previous_history.get('val_loss', []) + history['val_loss']
        merged_history['val_metrics'] = previous_history.get('val_metrics', []) + history['val_metrics']
        merged_history['training_time'] = previous_history.get('training_time', 0) + history['training_time']
        
        # 对比历史最佳和本次最佳，选择最优的（基于R²）
        prev_best_r2 = previous_history.get('best_r2', float('-inf'))
        current_best_r2 = history.get('best_r2', early_stopping.best_score)

        if current_best_r2 > prev_best_r2:
            # 本次训练产生了更好的模型
            best_val_loss = history['best_val_loss']
            best_epoch = start_epoch + history['best_epoch']  # 调整epoch编号
            best_model_state = current_best_model_state
            history_improved = True
        else:
            # 历史模型更好，保留历史最佳
            best_val_loss = previous_history.get('best_val_loss', history['best_val_loss'])
            best_epoch = previous_history.get('best_epoch', 0)
            # 需要从旧检查点加载历史最佳模型权重
            if os.path.exists(model_path):
                old_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                best_model_state = old_checkpoint['model_state_dict']
            else:
                best_model_state = current_best_model_state
            history_improved = False
    else:
        # 首次训练
        best_val_loss = history['best_val_loss']
        best_epoch = history['best_epoch']
        best_model_state = current_best_model_state
        history_improved = True
    
    # 更新合并后的历史记录中的最佳信息
    merged_history['best_val_loss'] = best_val_loss
    merged_history['best_epoch'] = best_epoch
    
    # 打印对比信息
    if verbose:
        print(f"\n{'='*60}")
        print(f"训练历史合并分析：")
        if previous_history is not None:
            prev_best_r2 = previous_history.get('best_r2', float('-inf'))
            current_best_r2 = history.get('best_r2', early_stopping.best_score)
            prev_best_loss = previous_history.get('best_val_loss', float('inf'))
            current_best_loss = history['best_val_loss']

            print(f"  历史最佳R²: {prev_best_r2:.4f} (损失: {prev_best_loss:.4f}, epoch {previous_history.get('best_epoch', '?')})")
            print(f"  本次最佳R²: {current_best_r2:.4f} (损失: {current_best_loss:.4f}, epoch {start_epoch + history['best_epoch']})")

            improvement = current_best_r2 - prev_best_r2
            improvement_pct = abs(improvement / prev_best_r2) * 100 if prev_best_r2 != 0 else 0

            if history_improved:
                print(f"  ✅ 本次训练改进: +{improvement:.4f} ({improvement_pct:.2f}%)")
            else:
                print(f"  ⚠️  本次训练未改进，保留历史最佳模型")
        print(f"  最终保留最佳R²: {early_stopping.best_score:.4f} (损失: {best_val_loss:.4f}, epoch {best_epoch})")
        print(f"  累计训练轮数: {len(merged_history['train_loss'])}")
        print(f"{'='*60}\n")
    
    # 保存模型（始终保存历史最佳权重 + 完整训练历史）
    if save_best:
        torch.save({
            'model_state_dict': best_model_state,  # 历史最佳权重
            'model_name': model_name,
            'task_name': task_name,
            'history': merged_history,  # 完整训练历史（包含所有微调过程）
            'total_epochs': len(merged_history['train_loss']),
        }, model_path)
        if verbose:
            if history_improved:
                print(f"✅ 已保存改进后的模型至: {model_path}")
            else:
                print(f"✅ 已更新训练历史，保留历史最佳模型: {model_path}")
    
    # 返回合并后的完整历史
    return merged_history
    
    if verbose:
        total_epochs = len(merged_history['train_loss'])
        print(f"\n训练完成!")
        print(f"总训练轮数: {total_epochs}")
        print(f"最佳R²: {early_stopping.best_score:.4f} (验证损失: {merged_history.get('best_val_loss', history['best_val_loss']):.4f})")
        print(f"最佳模型所在epoch: {merged_history.get('best_epoch', history['best_epoch'])}")
        print(f"本次训练时间: {time.time() - start_time:.2f}秒")
        if previous_history:
            print(f"累计训练时间: {merged_history['training_time']:.2f}秒")
    
    return history


def test_model(model, test_loader, scaler_targets, device=DEVICE):
    """
    测试模型并返回详细结果
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        scaler_targets: 目标标准化器（用于反标准化）
        device: 设备
        
    Returns:
        测试指标字典, 预测值, 真实值
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 反标准化
    original_shape = all_predictions.shape
    all_predictions_flat = all_predictions.reshape(-1, original_shape[-1])
    all_targets_flat = all_targets.reshape(-1, original_shape[-1])
    
    all_predictions_inv = scaler_targets.inverse_transform(all_predictions_flat)
    all_targets_inv = scaler_targets.inverse_transform(all_targets_flat)
    
    all_predictions_inv = all_predictions_inv.reshape(original_shape)
    all_targets_inv = all_targets_inv.reshape(original_shape)
    
    # 计算指标（使用原始尺度）
    metrics = calculate_metrics(all_targets_inv, all_predictions_inv)
    
    # 计算每个目标的指标
    metrics_per_target = []
    for i in range(original_shape[-1]):
        target_metrics = calculate_metrics(
            all_targets_inv[:, :, i],
            all_predictions_inv[:, :, i]
        )
        metrics_per_target.append(target_metrics)
    
    return metrics, metrics_per_target, all_predictions_inv, all_targets_inv


def load_model(model, model_path):
    """
    加载保存的模型
    
    Args:
        model: 模型实例
        model_path: 模型文件路径
        
    Returns:
        加载了权重的模型, 训练历史
    """
    # PyTorch 2.6+ 需要设置 weights_only=False
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    history = checkpoint.get('history', {})
    return model, history


def print_test_results(model_name, task_name, metrics, metrics_per_target, target_names):
    """
    打印测试结果
    
    Args:
        model_name: 模型名称
        task_name: 任务名称
        metrics: 总体指标
        metrics_per_target: 每个目标的指标
        target_names: 目标名称列表
    """
    print(f"\n{'='*60}")
    print(f"测试结果: {model_name} - {task_name}")
    print(f"{'='*60}")
    
    print(f"\n总体指标:")
    print(f"  MSE:  {metrics['MSE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    print(f"\n各目标指标:")
    for i, (name, m) in enumerate(zip(target_names, metrics_per_target)):
        print(f"\n  {name}:")
        print(f"    MSE:  {m['MSE']:.4f}")
        print(f"    RMSE: {m['RMSE']:.4f}")
        print(f"    MAE:  {m['MAE']:.4f}")
        print(f"    R²:   {m['R2']:.4f}")


def compare_models(results_dict):
    """
    对比多个模型的结果
    
    Args:
        results_dict: {model_name: {task_name: metrics}}
        
    Returns:
        对比结果DataFrame
    """
    import pandas as pd
    
    rows = []
    for model_name, tasks in results_dict.items():
        for task_name, metrics in tasks.items():
            row = {
                'Model': model_name,
                'Task': task_name,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # 测试训练模块
    print("测试训练模块...")
    
    # 创建模拟数据
    from torch.utils.data import TensorDataset, DataLoader
    
    batch_size = 32
    input_len = 8
    output_len = 1
    num_features = 20
    num_targets = 3
    num_samples = 1000
    
    # 模拟数据
    X = torch.randn(num_samples, input_len, num_features)
    Y = torch.randn(num_samples, output_len, num_targets)
    
    dataset = TensorDataset(X, Y)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 测试训练
    from models import LinearModel
    model = LinearModel(input_len, output_len, num_features, num_targets)
    
    history = train_model(
        model, train_loader, val_loader,
        model_name='Linear_test',
        task_name='test',
        num_epochs=5,
        save_best=False,
        verbose=True
    )
    
    print(f"\n训练历史:")
    print(f"  最佳epoch: {history['best_epoch']}")
    print(f"  最佳验证损失: {history['best_val_loss']:.4f}")
    
    print("\n训练模块测试完成！")
