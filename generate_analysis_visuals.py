"""
生成模型分析可视化图像
生成各种对比图、性能排名图等
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 定义颜色方案
COLORS = {
    'Linear': '#3498db',
    'LSTM': '#e74c3c', 
    'Transformer': '#9b59b6',
    'CNN_LSTM': '#2ecc71',
    'TCN': '#f39c12',
    'WaveNet': '#1abc9c',
    'LSTNet': '#e67e22',
    'DLinear': '#34495e',
    'HeightAttention': '#c0392b',
    'TrendLinear': '#27ae60',
    'WindShear': '#8e44ad',
    'Persistence': '#95a5a6',
}

# 模型类型分类
MODEL_TYPES = {
    'Linear': 'Basic',
    'LSTM': 'Basic',
    'Transformer': 'Basic',
    'CNN_LSTM': 'Innovative',
    'TCN': 'Innovative',
    'WaveNet': 'Innovative',
    'LSTNet': 'Innovative',
    'DLinear': 'Innovative',
    'HeightAttention': 'Innovative',
    'TrendLinear': 'Innovative',
    'WindShear': 'Innovative',
    'Persistence': 'Baseline',
}

def load_results():
    """加载模型比较结果"""
    results_path = '/workspace/results/model_comparison.csv'
    df = pd.read_csv(results_path)
    return df

def plot_performance_bars(df, save_dir):
    """绘制各模型性能条形图"""
    # 单步预测
    single_df = df[df['Task'] == 'singlestep'].sort_values('RMSE')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 单步预测 - RMSE
    ax1 = axes[0, 0]
    colors = [COLORS.get(m, '#666666') for m in single_df['Model']]
    bars = ax1.barh(range(len(single_df)), single_df['RMSE'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(single_df)))
    ax1.set_yticklabels(single_df['Model'])
    ax1.set_xlabel('RMSE (m/s)')
    ax1.set_title('Single-step Prediction (8h→1h) - RMSE Ranking', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, single_df['RMSE'])):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9)
    
    # 单步预测 - R²
    ax2 = axes[0, 1]
    single_df_r2 = single_df.sort_values('R2', ascending=False)
    colors = [COLORS.get(m, '#666666') for m in single_df_r2['Model']]
    bars = ax2.barh(range(len(single_df_r2)), single_df_r2['R2'], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(single_df_r2)))
    ax2.set_yticklabels(single_df_r2['Model'])
    ax2.set_xlabel('R² Score')
    ax2.set_title('Single-step Prediction (8h→1h) - R² Ranking', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, single_df_r2['R2'])):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9)
    
    # 多步预测
    multi_df = df[df['Task'] == 'multistep_16h'].sort_values('RMSE')
    
    # 多步预测 - RMSE
    ax3 = axes[1, 0]
    colors = [COLORS.get(m, '#666666') for m in multi_df['Model']]
    bars = ax3.barh(range(len(multi_df)), multi_df['RMSE'], color=colors, alpha=0.8)
    ax3.set_yticks(range(len(multi_df)))
    ax3.set_yticklabels(multi_df['Model'])
    ax3.set_xlabel('RMSE (m/s)')
    ax3.set_title('Multi-step Prediction (8h→16h) - RMSE Ranking', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, multi_df['RMSE'])):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9)
    
    # 多步预测 - R²
    ax4 = axes[1, 1]
    multi_df_r2 = multi_df.sort_values('R2', ascending=False)
    colors = [COLORS.get(m, '#666666') for m in multi_df_r2['Model']]
    bars = ax4.barh(range(len(multi_df_r2)), multi_df_r2['R2'], color=colors, alpha=0.8)
    ax4.set_yticks(range(len(multi_df_r2)))
    ax4.set_yticklabels(multi_df_r2['Model'])
    ax4.set_xlabel('R² Score')
    ax4.set_title('Multi-step Prediction (8h→16h) - R² Ranking', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, multi_df_r2['R2'])):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_performance_ranking.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def plot_comparison_grouped(df, save_dir):
    """按模型类型分组对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, task in enumerate(['singlestep', 'multistep_16h']):
        ax = axes[idx]
        task_df = df[df['Task'] == task].copy()
        task_df['Type'] = task_df['Model'].map(MODEL_TYPES)
        
        # 计算每个类型的平均性能
        type_order = ['Basic', 'Innovative', 'Baseline']
        x = np.arange(len(task_df))
        width = 0.35
        
        colors_rmse = ['#3498db' if t == 'Basic' else '#2ecc71' if t == 'Innovative' else '#95a5a6' 
                       for t in task_df['Type']]
        
        bars = ax.bar(x, task_df['RMSE'], width, label='RMSE', color=colors_rmse, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('RMSE (m/s)', fontsize=11)
        task_name = 'Single-step (8h→1h)' if task == 'singlestep' else 'Multi-step (8h→16h)'
        ax.set_title(f'{task_name} - RMSE by Model Type', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.8, label='Basic Models'),
            Patch(facecolor='#2ecc71', alpha=0.8, label='Innovative Models'),
            Patch(facecolor='#95a5a6', alpha=0.8, label='Baseline'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison_by_type.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def plot_radar_chart(df, save_dir):
    """绘制雷达图对比基础模型"""
    basic_models = ['Linear', 'LSTM', 'Transformer']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    
    for idx, task in enumerate(['singlestep', 'multistep_16h']):
        ax = axes[idx]
        task_df = df[df['Task'] == task]
        
        # 选择基础模型
        basic_df = task_df[task_df['Model'].isin(basic_models)]
        
        metrics = ['MSE', 'RMSE', 'MAE']
        num_vars = len(metrics)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        for i, model in enumerate(basic_models):
            model_data = basic_df[basic_df['Model'] == model].iloc[0]
            values = [model_data[m] for m in metrics]
            # 归一化到0-1范围
            max_vals = [task_df[m].max() for m in metrics]
            values_norm = [v/max_v for v, max_v in zip(values, max_vals)]
            values_norm += values_norm[:1]
            
            ax.plot(angles, values_norm, 'o-', linewidth=2, 
                   label=model, color=COLORS[model])
            ax.fill(angles, values_norm, alpha=0.15, color=COLORS[model])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        task_name = 'Single-step' if task == 'singlestep' else 'Multi-step'
        ax.set_title(f'{task_name} - Basic Models Comparison', fontsize=11, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'basic_models_radar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def plot_task_comparison(df, save_dir):
    """绘制单步vs多步预测对比"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    single_df = df[df['Task'] == 'singlestep'].set_index('Model')
    multi_df = df[df['Task'] == 'multistep_16h'].set_index('Model')
    
    single_rmse = [single_df.loc[m, 'RMSE'] if m in single_df.index else 0 for m in models]
    multi_rmse = [multi_df.loc[m, 'RMSE'] if m in multi_df.index else 0 for m in models]
    
    bars1 = ax.bar(x - width/2, single_rmse, width, label='Single-step (8h→1h)', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, multi_rmse, width, label='Multi-step (8h→16h)', 
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('RMSE (m/s)', fontsize=11)
    ax.set_title('Single-step vs Multi-step Prediction RMSE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'single_vs_multi_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def plot_all_metrics(df, save_dir):
    """绘制所有指标热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, task in enumerate(['singlestep', 'multistep_16h']):
        ax = axes[idx]
        task_df = df[df['Task'] == task].set_index('Model')
        
        # 选择指标列
        metrics_df = task_df[['MSE', 'RMSE', 'MAE', 'R2']]
        
        # 创建热力图数据（对R2需要反向，因为越大越好）
        plot_df = metrics_df.copy()
        
        sns.heatmap(plot_df, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': 'Value'})
        
        task_name = 'Single-step (8h→1h)' if task == 'singlestep' else 'Multi-step (8h→16h)'
        ax.set_title(f'{task_name} - All Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_metrics_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def plot_best_models_summary(df, save_dir):
    """绘制最佳模型总结图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 获取各任务最佳模型
    single_best = df[df['Task'] == 'singlestep'].nsmallest(3, 'RMSE')
    multi_best = df[df['Task'] == 'multistep_16h'].nsmallest(3, 'RMSE')
    
    # 创建表格数据
    table_data = []
    
    # 单步预测TOP3
    for i, (_, row) in enumerate(single_best.iterrows()):
        table_data.append([
            f"🥇" if i == 0 else f"🥈" if i == 1 else "🥉",
            'Single-step',
            row['Model'],
            f"{row['RMSE']:.4f}",
            f"{row['R2']:.4f}",
            MODEL_TYPES.get(row['Model'], 'Unknown')
        ])
    
    # 多步预测TOP3
    for i, (_, row) in enumerate(multi_best.iterrows()):
        table_data.append([
            f"🥇" if i == 0 else f"🥈" if i == 1 else "🥉",
            'Multi-step',
            row['Model'],
            f"{row['RMSE']:.4f}",
            f"{row['R2']:.4f}",
            MODEL_TYPES.get(row['Model'], 'Unknown')
        ])
    
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Rank', 'Task', 'Model', 'RMSE', 'R²', 'Type'],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.2, 0.2, 0.15, 0.15, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # 设置表头样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if j == 5:  # Type列
                if 'Basic' in cell.get_text().get_text():
                    cell.set_facecolor('#d5f4e6')
                elif 'Innovative' in cell.get_text().get_text():
                    cell.set_facecolor('#ffeaa7')
                else:
                    cell.set_facecolor('#f8f9fa')
    
    ax.set_title('🏆 Best Models Summary (Top 3 by RMSE)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'best_models_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def plot_improvement_analysis(df, save_dir):
    """分析创新模型相对基准的改进"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, task in enumerate(['singlestep', 'multistep_16h']):
        ax = axes[idx]
        task_df = df[df['Task'] == task]
        
        # 使用Persistence作为基准
        baseline = task_df[task_df['Model'] == 'Persistence']['RMSE'].values
        if len(baseline) == 0:
            baseline = task_df['RMSE'].max()
        else:
            baseline = baseline[0]
        
        # 计算相对改进
        improvements = []
        models = []
        for _, row in task_df.iterrows():
            if row['Model'] != 'Persistence':
                imp = (baseline - row['RMSE']) / baseline * 100
                improvements.append(imp)
                models.append(row['Model'])
        
        # 排序
        sorted_idx = np.argsort(improvements)[::-1]
        improvements = [improvements[i] for i in sorted_idx]
        models = [models[i] for i in sorted_idx]
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax.barh(range(len(models)), improvements, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel('Improvement vs Baseline (%)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        task_name = 'Single-step' if task == 'singlestep' else 'Multi-step'
        ax.set_title(f'{task_name} - RMSE Improvement vs Persistence Baseline', 
                    fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值
        for bar, val in zip(bars, improvements):
            x_pos = val + 0.3 if val >= 0 else val - 0.8
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                   va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'improvement_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("生成模型分析可视化图像")
    print("=" * 60)
    
    save_dir = '/workspace/results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    df = load_results()
    print(f"\n加载了 {len(df)} 条模型结果")
    print(f"模型: {df['Model'].unique().tolist()}")
    print(f"任务: {df['Task'].unique().tolist()}")
    
    # 生成各种可视化
    print("\n生成可视化图像...")
    
    print("\n1. 模型性能排名图")
    plot_performance_bars(df, save_dir)
    
    print("\n2. 按类型分组对比图")
    plot_comparison_grouped(df, save_dir)
    
    print("\n3. 基础模型雷达图")
    plot_radar_chart(df, save_dir)
    
    print("\n4. 单步vs多步对比图")
    plot_task_comparison(df, save_dir)
    
    print("\n5. 所有指标热力图")
    plot_all_metrics(df, save_dir)
    
    print("\n6. 最佳模型总结图")
    plot_best_models_summary(df, save_dir)
    
    print("\n7. 改进分析图")
    plot_improvement_analysis(df, save_dir)
    
    print("\n" + "=" * 60)
    print("所有可视化图像已生成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
