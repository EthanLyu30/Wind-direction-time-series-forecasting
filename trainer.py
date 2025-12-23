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
    """早停机制"""
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0, mode='min'):
        """
        初始化早停
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def load_best_model(self, model):
        """加载最佳模型状态"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


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
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE,
                save_best=True, verbose=True):
    """
    完整的模型训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model_name: 模型名称
        task_name: 任务名称 ('singlestep', 'multistep_1h', 'multistep_16h')
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        save_best: 是否保存最佳模型
        verbose: 是否打印详细信息
        
    Returns:
        训练历史字典
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # PyTorch 2.x 中 ReduceLROnPlateau 移除了 verbose 参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'training_time': 0
    }
    
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"训练 {model_name} - {task_name}")
        print(f"{'='*60}")
        print(f"设备: {device}")
        print(f"学习率: {lr}")
        print(f"训练轮数: {num_epochs}")
    
    progress_bar = tqdm(range(num_epochs), desc=f"Training {model_name}")
    
    for epoch in progress_bar:
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
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 更新进度条
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_rmse': f'{val_metrics["RMSE"]:.4f}'
        })
        
        # 检查是否为最佳模型
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            if verbose:
                print(f"\n早停触发于 epoch {epoch + 1}")
            break
    
    # 加载最佳模型
    early_stopping.load_best_model(model)
    
    history['training_time'] = time.time() - start_time
    
    # 保存模型
    if save_best:
        model_filename = f"{model_name}_{task_name}.pth"
        model_path = os.path.join(MODELS_DIR, model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'task_name': task_name,
            'history': history,
        }, model_path)
        if verbose:
            print(f"\n模型已保存至: {model_path}")
    
    if verbose:
        print(f"\n训练完成!")
        print(f"最佳验证损失: {history['best_val_loss']:.4f} (epoch {history['best_epoch']})")
        print(f"训练时间: {history['training_time']:.2f}秒")
    
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
