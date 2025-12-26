"""
可视化模块
功能：
1. 数据集可视化
2. 训练过程可视化
3. 预测结果可视化
4. 模型对比可视化
"""
import os
import sys
import numpy as np
import pandas as pd

# ==================== 解决服务器无图形界面问题 ====================
# 必须在导入matplotlib.pyplot之前设置后端
import matplotlib

def _is_headless():
    """检测是否为无头服务器环境"""
    # Windows通常有图形界面
    if sys.platform == 'win32':
        return False
    # Linux/Mac 检查DISPLAY环境变量
    if not os.environ.get('DISPLAY'):
        return True
    # 检查是否在SSH会话中（无X11转发）
    if os.environ.get('SSH_CONNECTION') and not os.environ.get('DISPLAY'):
        return True
    # 检查QT_QPA_PLATFORM是否设置为offscreen
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        return True
    return False

# 如果是无头服务器，强制使用Agg后端
if _is_headless():
    matplotlib.use('Agg')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
else:
    # 尝试使用Agg后端以避免Qt问题（更安全）
    try:
        matplotlib.use('Agg')
    except:
        pass

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties, fontManager
import warnings
warnings.filterwarnings('ignore')

from config import RESULTS_DIR, TARGET_COL

# ==================== 字体配置 ====================
def setup_chinese_font():
    """设置中文字体，确保在Windows/Linux上都能正确显示"""
    chinese_fonts = []
    
    if sys.platform == 'win32':
        # Windows系统常见中文字体
        font_candidates = [
            'Microsoft YaHei',   # 微软雅黑
            'SimHei',            # 黑体
            'SimSun',            # 宋体
            'KaiTi',             # 楷体
            'FangSong',          # 仿宋
        ]
    else:
        # Linux系统
        font_candidates = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei', 
            'Noto Sans CJK SC',
            'Droid Sans Fallback',
            'DejaVu Sans',
        ]
    
    # 获取系统已安装的字体
    available_fonts = set([f.name for f in fontManager.ttflist])
    
    # 找到可用的中文字体
    for font in font_candidates:
        if font in available_fonts:
            chinese_fonts.append(font)
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial']
        print(f"📝 使用字体: {chinese_fonts[0]}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        print("⚠️ 未找到中文字体，使用默认字体（中文可能显示为方框）")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_fonts[0] if chinese_fonts else 'DejaVu Sans'

# 初始化字体
MAIN_FONT = setup_chinese_font()

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

def plot_dataset_overview(df, save_path=None):
    """
    绘制数据集概览
    
    Args:
        df: 预处理后的DataFrame
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 时间序列图 - 三个高度的风速
    ax1 = axes[0, 0]
    for height in [10, 50, 100]:
        col_name = f'{TARGET_COL}_{height}m'
        if col_name in df.columns:
            ax1.plot(df.index[:500], df[col_name].values[:500], 
                    label=f'{height}m', alpha=0.8)
    ax1.set_xlabel('样本索引')
    ax1.set_ylabel('风速 (m/s)')
    ax1.set_title('风速时间序列（前500个样本）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 风速分布直方图
    ax2 = axes[0, 1]
    for height in [10, 50, 100]:
        col_name = f'{TARGET_COL}_{height}m'
        if col_name in df.columns:
            ax2.hist(df[col_name].dropna(), bins=50, alpha=0.5, 
                    label=f'{height}m')
    ax2.set_xlabel('风速 (m/s)')
    ax2.set_ylabel('频次')
    ax2.set_title('风速分布')
    ax2.legend()
    
    # 3. 特征相关性热图
    ax3 = axes[1, 0]
    feature_cols = [col for col in df.columns if any(x in col for x in ['Avg', 'sin', 'cos'])][:12]
    if len(feature_cols) > 0:
        corr_matrix = df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   ax=ax3, vmin=-1, vmax=1, center=0)
        ax3.set_title('特征相关性矩阵')
    
    # 4. 箱线图
    ax4 = axes[1, 1]
    speed_cols = [f'{TARGET_COL}_{h}m' for h in [10, 50, 100]]
    speed_cols = [c for c in speed_cols if c in df.columns]
    if speed_cols:
        df[speed_cols].boxplot(ax=ax4)
        ax4.set_ylabel('风速 (m/s)')
        ax4.set_title('各高度风速箱线图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def plot_training_history(history, model_name, task_name, save_path=None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        model_name: 模型名称
        task_name: 任务名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. 损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.axvline(x=history['best_epoch'], color='g', linestyle='--', 
                label=f'最佳模型 (epoch {history["best_epoch"]})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失 (MSE)')
    ax1.set_title(f'{model_name} - {task_name} 训练过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 验证指标
    ax2 = axes[1]
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        rmse_values = [m['RMSE'] for m in history['val_metrics']]
        r2_values = [m['R2'] for m in history['val_metrics']]
        
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(epochs, rmse_values, 'b-', label='RMSE', linewidth=2)
        line2 = ax2_twin.plot(epochs, r2_values, 'g-', label='R²', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE', color='b')
        ax2_twin.set_ylabel('R²', color='g')
        ax2.set_title('验证集指标')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def plot_predictions(y_true, y_pred, model_name, task_name, target_names, 
                    num_samples=200, save_path=None):
    """
    绘制预测结果对比
    
    Args:
        y_true: 真实值 (samples, output_len, num_targets)
        y_pred: 预测值
        model_name: 模型名称
        task_name: 任务名称
        target_names: 目标名称列表
        num_samples: 显示样本数
        save_path: 保存路径
    """
    num_targets = len(target_names)
    fig, axes = plt.subplots(num_targets, 1, figsize=(14, 4*num_targets))
    
    if num_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        # 取第一个输出步长的预测
        true_vals = y_true[:num_samples, 0, i]
        pred_vals = y_pred[:num_samples, 0, i]
        
        x = range(len(true_vals))
        
        ax.plot(x, true_vals, 'b-', label='真实值', linewidth=1.5, alpha=0.8)
        ax.plot(x, pred_vals, 'r--', label='预测值', linewidth=1.5, alpha=0.8)
        ax.fill_between(x, true_vals, pred_vals, alpha=0.2, color='gray')
        
        ax.set_xlabel('样本索引')
        ax.set_ylabel('风速 (m/s)')
        ax.set_title(f'{name} - {model_name} ({task_name})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def plot_prediction_scatter(y_true, y_pred, model_name, task_name, target_names, save_path=None):
    """
    绘制预测值vs真实值散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        task_name: 任务名称
        target_names: 目标名称列表
        save_path: 保存路径
    """
    num_targets = len(target_names)
    fig, axes = plt.subplots(1, num_targets, figsize=(5*num_targets, 5))
    
    if num_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        true_vals = y_true[:, :, i].flatten()
        pred_vals = y_pred[:, :, i].flatten()
        
        # 散点图
        ax.scatter(true_vals, pred_vals, alpha=0.3, s=10)
        
        # 理想线
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
        
        # 计算R²
        from sklearn.metrics import r2_score
        r2 = r2_score(true_vals, pred_vals)
        
        ax.set_xlabel('真实值 (m/s)')
        ax.set_ylabel('预测值 (m/s)')
        ax.set_title(f'{name}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle(f'{model_name} - {task_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def plot_multistep_predictions(y_true, y_pred, model_name, task_name, target_idx=0,
                              sample_indices=None, save_path=None):
    """
    绘制多步预测结果
    
    Args:
        y_true: 真实值 (samples, output_len, num_targets)
        y_pred: 预测值
        model_name: 模型名称
        task_name: 任务名称
        target_idx: 目标索引
        sample_indices: 要显示的样本索引列表
        save_path: 保存路径
    """
    if sample_indices is None:
        sample_indices = [0, len(y_true)//4, len(y_true)//2, 3*len(y_true)//4]
    
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    output_len = y_true.shape[1]
    x = range(output_len)
    
    for ax, idx in zip(axes, sample_indices):
        true_vals = y_true[idx, :, target_idx]
        pred_vals = y_pred[idx, :, target_idx]
        
        ax.plot(x, true_vals, 'b-o', label='真实值', linewidth=2, markersize=4)
        ax.plot(x, pred_vals, 'r--s', label='预测值', linewidth=2, markersize=4)
        
        ax.set_xlabel('预测步长')
        ax.set_ylabel('风速 (m/s)')
        ax.set_title(f'样本 {idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
    
    plt.suptitle(f'{model_name} - {task_name} 多步预测', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def plot_model_comparison(results_df, metric='RMSE', save_path=None):
    """
    绘制模型对比图
    
    Args:
        results_df: 包含模型结果的DataFrame
        metric: 要比较的指标
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 按任务分组
    tasks = results_df['Task'].unique()
    models = results_df['Model'].unique()
    
    x = np.arange(len(tasks))
    width = 0.8 / len(models)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model]
        values = [model_data[model_data['Task'] == task][metric].values[0] 
                 for task in tasks if len(model_data[model_data['Task'] == task]) > 0]
        
        bars = ax.bar(x[:len(values)] + i * width, values, width, 
                     label=model, color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('任务')
    ax.set_ylabel(metric)
    ax.set_title(f'模型性能对比 - {metric}')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def plot_error_distribution(y_true, y_pred, model_name, task_name, target_names, save_path=None):
    """
    绘制预测误差分布
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        task_name: 任务名称
        target_names: 目标名称列表
        save_path: 保存路径
    """
    num_targets = len(target_names)
    fig, axes = plt.subplots(1, num_targets, figsize=(5*num_targets, 4))
    
    if num_targets == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        errors = (y_pred[:, :, i] - y_true[:, :, i]).flatten()
        
        ax.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # 添加正态分布拟合曲线
        mu, std = errors.mean(), errors.std()
        x = np.linspace(errors.min(), errors.max(), 100)
        from scipy import stats
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'正态分布\nμ={mu:.3f}, σ={std:.3f}')
        
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('预测误差 (m/s)')
        ax.set_ylabel('密度')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - {task_name} 误差分布', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    plt.close()


def create_results_summary_table(results_dict, save_path=None):
    """
    创建结果汇总表格
    
    Args:
        results_dict: {model_name: {task_name: metrics}}
        save_path: 保存路径
    """
    rows = []
    for model_name, tasks in results_dict.items():
        for task_name, metrics in tasks.items():
            rows.append({
                '模型': model_name,
                '任务': task_name,
                'MSE': f"{metrics['MSE']:.4f}",
                'RMSE': f"{metrics['RMSE']:.4f}",
                'MAE': f"{metrics['MAE']:.4f}",
                'R²': f"{metrics['R2']:.4f}"
            })
    
    df = pd.DataFrame(rows)
    
    # 绘制表格
    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表头颜色
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('gray')
    
    plt.title('模型性能汇总', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"表格已保存至: {save_path}")
    
    plt.show()
    plt.close()
    
    return df


if __name__ == "__main__":
    # 测试可视化模块
    print("测试可视化模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟预测结果
    num_samples = 100
    output_len = 16
    num_targets = 3
    
    y_true = np.random.randn(num_samples, output_len, num_targets) * 5 + 10
    y_pred = y_true + np.random.randn(num_samples, output_len, num_targets) * 0.5
    
    target_names = ['10m风速', '50m风速', '100m风速']
    
    # 测试预测散点图
    print("测试预测散点图...")
    plot_prediction_scatter(y_true, y_pred, 'Test_Model', 'test_task', target_names)
    
    # 测试误差分布图
    print("测试误差分布图...")
    plot_error_distribution(y_true, y_pred, 'Test_Model', 'test_task', target_names)
    
    print("\n可视化模块测试完成！")
