"""
生成各模型的预测结果可视化图
功能：
1. 对比真实值与预测值的时序图
2. 散点图（预测值 vs 真实值）
3. 多步预测曲线图
4. 误差分布图
"""
import os
import sys
import torch
import numpy as np
import pandas as pd

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 导入项目模块
from config import MODELS_DIR, RESULTS_DIR, DEVICE, SINGLE_STEP_INPUT_LEN, MULTI_STEP_INPUT_LEN
from data_loader import load_all_data, preprocess_data, create_dataloaders
from models import get_model
from models_innovative import get_innovative_model
from models_advanced import get_advanced_model
from models_simple import get_simple_model

# 创建预测结果保存目录
PREDICTION_PLOTS_DIR = os.path.join(RESULTS_DIR, 'prediction_plots')
os.makedirs(PREDICTION_PLOTS_DIR, exist_ok=True)

# 模型列表及其对应的获取函数
MODEL_GETTERS = {
    # 基础模型
    'Linear': ('basic', get_model),
    'LSTM': ('basic', get_model),
    'Transformer': ('basic', get_model),
    # 创新模型
    'DLinear': ('advanced', get_advanced_model),
    'TCN': ('innovative', get_innovative_model),
    'WaveNet': ('innovative', get_innovative_model),
    'LSTNet': ('innovative', get_innovative_model),
    'CNN_LSTM': ('innovative', get_innovative_model),
    'HeightAttention': ('advanced', get_advanced_model),
    # 简单模型
    'TrendLinear': ('simple', get_simple_model),
    'WindShear': ('simple', get_simple_model),
    'Persistence': ('simple', get_simple_model),
}

MODELS = list(MODEL_GETTERS.keys())

TARGET_NAMES = ['10m Wind Speed', '50m Wind Speed', '100m Wind Speed']

# 配置参数
NUM_FEATURES = 21
NUM_TARGETS = 3


def load_model(model_name, task_type, input_len, output_len):
    """加载训练好的模型"""
    if task_type == 'singlestep':
        model_path = os.path.join(MODELS_DIR, f'{model_name}_singlestep.pth')
        actual_output_len = 1
    else:
        model_path = os.path.join(MODELS_DIR, f'{model_name}_multistep_16h.pth')
        actual_output_len = 16
    
    if not os.path.exists(model_path):
        print(f"  ⚠️ 模型文件不存在: {model_path}")
        return None
    
    try:
        # 获取模型类别和获取函数
        model_type, model_getter = MODEL_GETTERS[model_name]
        
        # 创建模型实例
        model = model_getter(model_name, input_len, actual_output_len, NUM_FEATURES, NUM_TARGETS)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"  ❌ 加载模型失败 ({model_name}): {e}")
        return None


def get_predictions(model, test_loader):
    """获取模型在测试集上的预测结果"""
    y_true_list = []
    y_pred_list = []
    
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            pred = model(X)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    return y_true, y_pred


def plot_predictions_comparison(y_true, y_pred, model_name, task_name, num_samples=200, save_path=None):
    """
    绘制预测结果对比图（真实值 vs 预测值）
    """
    num_targets = y_true.shape[-1]
    fig, axes = plt.subplots(num_targets, 1, figsize=(14, 4*num_targets))
    
    if num_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        # 取第一个输出步长的预测
        true_vals = y_true[:num_samples, 0, i]
        pred_vals = y_pred[:num_samples, 0, i]
        
        x = range(len(true_vals))
        
        ax.plot(x, true_vals, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
        ax.plot(x, pred_vals, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
        ax.fill_between(x, true_vals, pred_vals, alpha=0.2, color='gray')
        
        # 计算该目标的指标
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Wind Speed (m/s)', fontsize=11)
        ax.set_title(f'{name} (R²={r2:.4f}, RMSE={rmse:.4f} m/s)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - {task_name} Prediction Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ 保存: {os.path.basename(save_path)}")
    
    plt.close()


def plot_scatter(y_true, y_pred, model_name, task_name, save_path=None):
    """
    绘制散点图（预测值 vs 真实值）
    """
    num_targets = y_true.shape[-1]
    fig, axes = plt.subplots(1, num_targets, figsize=(5*num_targets, 5))
    
    if num_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        true_vals = y_true[:, :, i].flatten()
        pred_vals = y_pred[:, :, i].flatten()
        
        # 散点图
        ax.scatter(true_vals, pred_vals, alpha=0.3, s=10, c='steelblue')
        
        # 理想线
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        
        # 计算R²
        r2 = r2_score(true_vals, pred_vals)
        
        ax.set_xlabel('Actual (m/s)', fontsize=11)
        ax.set_ylabel('Predicted (m/s)', fontsize=11)
        ax.set_title(f'{name}\nR² = {r2:.4f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'{model_name} - {task_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ 保存: {os.path.basename(save_path)}")
    
    plt.close()


def plot_multistep_curves(y_true, y_pred, model_name, target_idx=0, sample_indices=None, save_path=None):
    """
    绘制多步预测曲线图（展示16个预测步长）
    """
    if sample_indices is None:
        # 默认选择4个代表性样本
        n = len(y_true)
        sample_indices = [0, n//4, n//2, 3*n//4]
    
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    output_len = y_true.shape[1]
    x = range(1, output_len + 1)
    
    for ax, idx in zip(axes, sample_indices):
        true_vals = y_true[idx, :, target_idx]
        pred_vals = y_pred[idx, :, target_idx]
        
        ax.plot(x, true_vals, 'b-o', label='Actual', linewidth=2, markersize=5)
        ax.plot(x, pred_vals, 'r--s', label='Predicted', linewidth=2, markersize=5)
        
        # 计算该样本的误差
        mae = np.mean(np.abs(true_vals - pred_vals))
        
        ax.set_xlabel('Prediction Step (hours)', fontsize=11)
        ax.set_ylabel('Wind Speed (m/s)', fontsize=11)
        ax.set_title(f'Sample {idx} (MAE={mae:.3f} m/s)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
    
    plt.suptitle(f'{model_name} - Multi-step Prediction (16h) - {TARGET_NAMES[target_idx]}', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ 保存: {os.path.basename(save_path)}")
    
    plt.close()


def plot_error_distribution(y_true, y_pred, model_name, task_name, save_path=None):
    """
    绘制预测误差分布图
    """
    num_targets = y_true.shape[-1]
    fig, axes = plt.subplots(1, num_targets, figsize=(5*num_targets, 4))
    
    if num_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        errors = (y_pred[:, :, i] - y_true[:, :, i]).flatten()
        
        ax.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # 统计信息
        mu, std = errors.mean(), errors.std()
        
        # 添加正态分布拟合曲线
        from scipy import stats
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, 
               label=f'Normal\nμ={mu:.3f}, σ={std:.3f}')
        
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Prediction Error (m/s)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - {task_name} Error Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ 保存: {os.path.basename(save_path)}")
    
    plt.close()


def generate_all_plots_for_model(model_name, task_type, test_loader, input_len, output_len):
    """为单个模型生成所有可视化图"""
    task_name = 'Single-step (1h)' if task_type == 'singlestep' else 'Multi-step (16h)'
    task_suffix = 'singlestep' if task_type == 'singlestep' else 'multistep'
    
    print(f"\n  📊 {model_name} - {task_name}")
    
    # 加载模型
    model = load_model(model_name, task_type, input_len, output_len)
    if model is None:
        return
    
    # 获取预测结果
    try:
        y_true, y_pred = get_predictions(model, test_loader)
    except Exception as e:
        print(f"    ❌ 预测失败: {e}")
        return
    
    # 创建模型专属目录
    model_dir = os.path.join(PREDICTION_PLOTS_DIR, f'{model_name}_{task_suffix}')
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. 时序对比图
    plot_predictions_comparison(
        y_true, y_pred, model_name, task_name,
        save_path=os.path.join(model_dir, f'{model_name}_{task_suffix}_timeseries.png')
    )
    
    # 2. 散点图
    plot_scatter(
        y_true, y_pred, model_name, task_name,
        save_path=os.path.join(model_dir, f'{model_name}_{task_suffix}_scatter.png')
    )
    
    # 3. 误差分布图
    plot_error_distribution(
        y_true, y_pred, model_name, task_name,
        save_path=os.path.join(model_dir, f'{model_name}_{task_suffix}_error_dist.png')
    )
    
    # 4. 多步预测曲线（仅对多步预测任务）
    if task_type == 'multistep' and y_true.shape[1] > 1:
        plot_multistep_curves(
            y_true, y_pred, model_name, target_idx=0,
            save_path=os.path.join(model_dir, f'{model_name}_multistep_curves_10m.png')
        )
        plot_multistep_curves(
            y_true, y_pred, model_name, target_idx=2,  # 100m
            save_path=os.path.join(model_dir, f'{model_name}_multistep_curves_100m.png')
        )


def generate_comparison_plot(all_results, task_type):
    """生成所有模型的对比图"""
    task_name = 'Single-step' if task_type == 'singlestep' else 'Multi-step'
    
    if not all_results:
        return
    
    # 准备数据
    models = list(all_results.keys())
    r2_scores = [all_results[m]['r2'] for m in models]
    rmse_scores = [all_results[m]['rmse'] for m in models]
    
    # 按R²排序
    sorted_idx = np.argsort(r2_scores)[::-1]
    models = [models[i] for i in sorted_idx]
    r2_scores = [r2_scores[i] for i in sorted_idx]
    rmse_scores = [rmse_scores[i] for i in sorted_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² 对比
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))
    bars1 = axes[0].barh(models, r2_scores, color=colors)
    axes[0].set_xlabel('R²', fontsize=12)
    axes[0].set_title(f'{task_name} - R² Comparison', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                    va='center', fontsize=9)
    
    # RMSE 对比
    colors_rmse = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars2 = axes[1].barh(models, rmse_scores, color=colors_rmse)
    axes[1].set_xlabel('RMSE (m/s)', fontsize=12)
    axes[1].set_title(f'{task_name} - RMSE Comparison', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars2, rmse_scores):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                    va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(PREDICTION_PLOTS_DIR, f'all_models_comparison_{task_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 模型对比图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("生成模型预测结果可视化")
    print("=" * 60)
    
    # 加载数据
    print("\n📂 加载数据...")
    raw_df = load_all_data()
    processed_df = preprocess_data(raw_df)
    
    # 处理单步预测
    print("\n" + "=" * 60)
    print("单步预测 (8h → 1h)")
    print("=" * 60)
    
    _, _, test_loader_single, _, _, _, _ = create_dataloaders(
        processed_df, input_len=SINGLE_STEP_INPUT_LEN, output_len=1, batch_size=64
    )
    
    single_results = {}
    for model_name in MODELS:
        model = load_model(model_name, 'singlestep', SINGLE_STEP_INPUT_LEN, 1)
        if model is not None:
            try:
                y_true, y_pred = get_predictions(model, test_loader_single)
                r2 = r2_score(y_true.flatten(), y_pred.flatten())
                rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
                single_results[model_name] = {'r2': r2, 'rmse': rmse}
                generate_all_plots_for_model(model_name, 'singlestep', test_loader_single, 
                                            SINGLE_STEP_INPUT_LEN, 1)
            except Exception as e:
                print(f"  ❌ {model_name} 处理失败: {e}")
    
    generate_comparison_plot(single_results, 'singlestep')
    
    # 处理多步预测
    print("\n" + "=" * 60)
    print("多步预测 (8h → 16h)")
    print("=" * 60)
    
    _, _, test_loader_multi, _, _, _, _ = create_dataloaders(
        processed_df, input_len=MULTI_STEP_INPUT_LEN, output_len=16, batch_size=64
    )
    
    multi_results = {}
    for model_name in MODELS:
        model = load_model(model_name, 'multistep', MULTI_STEP_INPUT_LEN, 16)
        if model is not None:
            try:
                y_true, y_pred = get_predictions(model, test_loader_multi)
                r2 = r2_score(y_true.flatten(), y_pred.flatten())
                rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
                multi_results[model_name] = {'r2': r2, 'rmse': rmse}
                generate_all_plots_for_model(model_name, 'multistep', test_loader_multi,
                                            MULTI_STEP_INPUT_LEN, 16)
            except Exception as e:
                print(f"  ❌ {model_name} 处理失败: {e}")
    
    generate_comparison_plot(multi_results, 'multistep')
    
    print("\n" + "=" * 60)
    print(f"✅ 所有可视化已保存到: {PREDICTION_PLOTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
