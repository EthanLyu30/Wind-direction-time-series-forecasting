"""
生成完整的可视化结果
包括：数据集概览、模型对比图等
"""
import os
import sys
import pandas as pd
import numpy as np

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

RESULTS_DIR = '/workspace/results'

def generate_dataset_overview():
    """生成数据集概览图"""
    print("生成数据集概览图...")
    
    # 加载数据
    sys.path.insert(0, '/workspace')
    from data_loader import load_all_data, preprocess_data
    from config import TARGET_COL
    
    raw_df = load_all_data()
    processed_df = preprocess_data(raw_df)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 时间序列图 - 三个高度的风速
    ax1 = axes[0, 0]
    for height in [10, 50, 100]:
        col_name = f'{TARGET_COL}_{height}m'
        if col_name in processed_df.columns:
            ax1.plot(processed_df.index[:500], processed_df[col_name].values[:500], 
                    label=f'{height}m', alpha=0.8)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Wind Speed (m/s)')
    ax1.set_title('Wind Speed Time Series (First 500 Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 风速分布直方图
    ax2 = axes[0, 1]
    for height in [10, 50, 100]:
        col_name = f'{TARGET_COL}_{height}m'
        if col_name in processed_df.columns:
            ax2.hist(processed_df[col_name].dropna(), bins=50, alpha=0.5, 
                    label=f'{height}m')
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Wind Speed Distribution')
    ax2.legend()
    
    # 3. 特征相关性热图
    ax3 = axes[1, 0]
    feature_cols = [col for col in processed_df.columns if any(x in col for x in ['Avg', 'sin', 'cos'])][:12]
    if len(feature_cols) > 0:
        corr_matrix = processed_df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   ax=ax3, vmin=-1, vmax=1, center=0, annot_kws={'fontsize': 8})
        ax3.set_title('Feature Correlation Matrix')
    
    # 4. 箱线图
    ax4 = axes[1, 1]
    speed_cols = [f'{TARGET_COL}_{h}m' for h in [10, 50, 100]]
    speed_cols = [c for c in speed_cols if c in processed_df.columns]
    if speed_cols:
        processed_df[speed_cols].boxplot(ax=ax4)
        ax4.set_ylabel('Wind Speed (m/s)')
        ax4.set_title('Wind Speed Boxplot by Height')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'dataset_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()
    
    return processed_df

def generate_comparison_charts(df):
    """生成模型对比图"""
    print("\n生成模型对比图...")
    
    # 加载结果
    results_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    results_df = pd.read_csv(results_path)
    
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 按任务分组
        tasks = results_df['Task'].unique()
        models = results_df['Model'].unique()
        
        x = np.arange(len(tasks))
        width = 0.8 / len(models)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_data = results_df[results_df['Model'] == model]
            values = []
            for task in tasks:
                task_data = model_data[model_data['Task'] == task]
                if len(task_data) > 0:
                    values.append(task_data[metric].values[0])
            
            if len(values) == len(tasks):
                bars = ax.bar(x + i * width, values, width, 
                             label=model, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Task')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Performance Comparison - {metric}')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(['Single-step (8h->1h)', 'Multi-step (8h->16h)'])
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'comparison_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存: {save_path}")
        plt.close()

def generate_summary_table():
    """生成结果汇总表格"""
    print("\n生成结果汇总表格...")
    
    results_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    results_df = pd.read_csv(results_path)
    
    fig, ax = plt.subplots(figsize=(14, len(results_df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # 格式化数据
    display_data = []
    for _, row in results_df.iterrows():
        display_data.append([
            row['Model'],
            row['Task'],
            f"{row['MSE']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['R2']:.4f}"
        ])
    
    table = ax.table(
        cellText=display_data,
        colLabels=['Model', 'Task', 'MSE', 'RMSE', 'MAE', 'R2'],
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * 6
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表头颜色
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('gray')
    
    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    save_path = os.path.join(RESULTS_DIR, 'results_summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("生成完整可视化结果")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 生成数据集概览
    df = generate_dataset_overview()
    
    # 生成模型对比图
    generate_comparison_charts(df)
    
    # 生成汇总表格
    generate_summary_table()
    
    print("\n" + "=" * 60)
    print("所有可视化图像已生成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
