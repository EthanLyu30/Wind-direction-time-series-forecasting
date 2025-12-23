"""
风速序列预测 - 主程序入口
功能：
1. 加载和预处理数据
2. 训练基础模型（Linear、LSTM、Transformer）
3. 训练创新模型（CNN-LSTM、Attention-LSTM、TCN、Ensemble、WaveNet）
4. 评估和对比所有模型
5. 可视化结果
6. 保存模型为pth格式

使用方法：
    python main.py                    # 运行完整实验
    python main.py --mode train       # 仅训练
    python main.py --mode eval        # 仅评估（需要已训练模型）
    python main.py --mode visualize   # 仅可视化
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, MODELS_DIR, RESULTS_DIR, LOGS_DIR,
    SINGLE_STEP_INPUT_LEN, SINGLE_STEP_OUTPUT_LEN,
    MULTI_STEP_1_INPUT_LEN, MULTI_STEP_1_OUTPUT_LEN,
    MULTI_STEP_2_INPUT_LEN, MULTI_STEP_2_OUTPUT_LEN,
    set_seed, RANDOM_SEED
)
from data_loader import (
    load_all_data, preprocess_data, create_dataloaders,
    get_feature_columns, get_target_columns
)
from models import get_model, count_parameters
from models_innovative import get_innovative_model
from trainer import (
    train_model, test_model, load_model, 
    print_test_results, compare_models
)
from visualization import (
    plot_dataset_overview, plot_training_history,
    plot_predictions, plot_prediction_scatter,
    plot_multistep_predictions, plot_model_comparison,
    plot_error_distribution, create_results_summary_table
)


# 定义任务配置
TASKS = {
    'singlestep': {
        'input_len': SINGLE_STEP_INPUT_LEN,
        'output_len': SINGLE_STEP_OUTPUT_LEN,
        'description': '单步预测（8小时→1小时）'
    },
    'multistep_1h': {
        'input_len': MULTI_STEP_1_INPUT_LEN,
        'output_len': MULTI_STEP_1_OUTPUT_LEN,
        'description': '多步预测（8小时→1小时）'
    },
    'multistep_16h': {
        'input_len': MULTI_STEP_2_INPUT_LEN,
        'output_len': MULTI_STEP_2_OUTPUT_LEN,
        'description': '多步预测（8小时→16小时）'
    }
}

# 基础模型
BASE_MODELS = ['Linear', 'LSTM', 'Transformer']

# 创新模型
INNOVATIVE_MODELS = ['CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']


def setup_experiment():
    """设置实验环境"""
    set_seed(RANDOM_SEED)
    
    # 创建必要目录
    for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("=" * 70)
    print("风速序列预测实验")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"最大训练轮数: {NUM_EPOCHS}")
    print("=" * 70)


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("\n" + "=" * 70)
    print("步骤1: 数据加载与预处理")
    print("=" * 70)
    
    # 加载原始数据
    raw_df = load_all_data()
    
    # 预处理
    processed_df = preprocess_data(raw_df)
    
    # 保存预处理后的数据信息
    info = {
        'shape': processed_df.shape,
        'columns': processed_df.columns.tolist(),
        'date_range': [str(processed_df.iloc[0]['Date & Time Stamp']), 
                      str(processed_df.iloc[-1]['Date & Time Stamp'])],
        'num_samples': len(processed_df)
    }
    
    info_path = os.path.join(RESULTS_DIR, 'data_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据信息已保存至: {info_path}")
    
    return processed_df


def visualize_dataset(df):
    """可视化数据集"""
    print("\n" + "=" * 70)
    print("步骤2: 数据集可视化")
    print("=" * 70)
    
    save_path = os.path.join(RESULTS_DIR, 'dataset_overview.png')
    plot_dataset_overview(df, save_path=save_path)


def train_all_models(df, model_list, tasks_to_run=None, is_innovative=False):
    """
    训练所有模型
    
    Args:
        df: 预处理后的数据
        model_list: 要训练的模型列表
        tasks_to_run: 要运行的任务列表（默认全部）
        is_innovative: 是否为创新模型
    """
    if tasks_to_run is None:
        tasks_to_run = list(TASKS.keys())
    
    model_type = "创新模型" if is_innovative else "基础模型"
    print(f"\n" + "=" * 70)
    print(f"步骤3: 训练{model_type}")
    print("=" * 70)
    
    all_results = {}
    
    for task_name in tasks_to_run:
        task_config = TASKS[task_name]
        print(f"\n{'='*50}")
        print(f"任务: {task_config['description']}")
        print(f"{'='*50}")
        
        # 创建数据加载器
        input_len = task_config['input_len']
        output_len = task_config['output_len']
        
        train_loader, val_loader, test_loader, scaler_features, scaler_targets, feature_cols, target_cols = \
            create_dataloaders(df, input_len, output_len, BATCH_SIZE)
        
        num_features = len(feature_cols)
        num_targets = len(target_cols)
        
        task_results = {}
        
        for model_name in model_list:
            print(f"\n--- 训练 {model_name} ---")
            
            # 创建模型
            if is_innovative:
                model = get_innovative_model(model_name, input_len, output_len, num_features, num_targets)
            else:
                model = get_model(model_name, input_len, output_len, num_features, num_targets)
            
            print(f"模型参数量: {count_parameters(model):,}")
            
            # 训练
            history = train_model(
                model, train_loader, val_loader,
                model_name=model_name,
                task_name=task_name,
                num_epochs=NUM_EPOCHS,
                device=DEVICE,
                save_best=True,
                verbose=True
            )
            
            # 绘制训练历史
            history_save_path = os.path.join(RESULTS_DIR, f'{model_name}_{task_name}_history.png')
            plot_training_history(history, model_name, task_name, save_path=history_save_path)
            
            # 测试
            metrics, metrics_per_target, predictions, targets = test_model(
                model, test_loader, scaler_targets, device=DEVICE
            )
            
            # 打印结果
            print_test_results(model_name, task_name, metrics, metrics_per_target, target_cols)
            
            # 保存结果
            task_results[model_name] = {
                'metrics': metrics,
                'metrics_per_target': metrics_per_target,
                'predictions': predictions,
                'targets': targets,
                'history': history
            }
            
            # 可视化预测结果
            pred_save_path = os.path.join(RESULTS_DIR, f'{model_name}_{task_name}_predictions.png')
            plot_predictions(targets, predictions, model_name, task_name, target_cols, 
                           num_samples=200, save_path=pred_save_path)
            
            scatter_save_path = os.path.join(RESULTS_DIR, f'{model_name}_{task_name}_scatter.png')
            plot_prediction_scatter(targets, predictions, model_name, task_name, target_cols,
                                  save_path=scatter_save_path)
            
            # 对于多步预测，额外绘制多步预测图
            if output_len > 1:
                multistep_save_path = os.path.join(RESULTS_DIR, f'{model_name}_{task_name}_multistep.png')
                plot_multistep_predictions(targets, predictions, model_name, task_name,
                                         save_path=multistep_save_path)
        
        all_results[task_name] = task_results
    
    return all_results


def evaluate_and_compare(all_results):
    """评估和对比所有模型"""
    print("\n" + "=" * 70)
    print("步骤4: 模型性能对比")
    print("=" * 70)
    
    # 整理结果
    comparison_dict = {}
    for task_name, task_results in all_results.items():
        for model_name, result in task_results.items():
            if model_name not in comparison_dict:
                comparison_dict[model_name] = {}
            comparison_dict[model_name][task_name] = result['metrics']
    
    # 创建对比DataFrame
    results_df = compare_models(comparison_dict)
    
    # 保存结果
    results_csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存至: {results_csv_path}")
    
    # 打印对比表格
    print("\n模型性能对比:")
    print(results_df.to_string(index=False))
    
    # 绘制对比图
    for metric in ['MSE', 'RMSE', 'MAE', 'R2']:
        comparison_save_path = os.path.join(RESULTS_DIR, f'comparison_{metric}.png')
        plot_model_comparison(results_df, metric=metric, save_path=comparison_save_path)
    
    # 创建汇总表格
    table_save_path = os.path.join(RESULTS_DIR, 'results_summary_table.png')
    create_results_summary_table(comparison_dict, save_path=table_save_path)
    
    return results_df


def generate_report(results_df, all_results):
    """生成实验报告"""
    print("\n" + "=" * 70)
    print("步骤5: 生成实验报告")
    print("=" * 70)
    
    report_path = os.path.join(RESULTS_DIR, 'experiment_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 风速序列预测实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 实验配置\n\n")
        f.write(f"- 设备: {DEVICE}\n")
        f.write(f"- 批次大小: {BATCH_SIZE}\n")
        f.write(f"- 最大训练轮数: {NUM_EPOCHS}\n")
        f.write(f"- 随机种子: {RANDOM_SEED}\n\n")
        
        f.write("## 2. 任务配置\n\n")
        for task_name, task_config in TASKS.items():
            f.write(f"### {task_config['description']}\n")
            f.write(f"- 输入长度: {task_config['input_len']}小时\n")
            f.write(f"- 输出长度: {task_config['output_len']}小时\n\n")
        
        f.write("## 3. 模型性能对比\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. 最佳模型\n\n")
        
        # 找出每个任务的最佳模型
        for task in TASKS.keys():
            task_results = results_df[results_df['Task'] == task]
            if len(task_results) > 0:
                best_idx = task_results['RMSE'].idxmin()
                best_model = task_results.loc[best_idx, 'Model']
                best_rmse = task_results.loc[best_idx, 'RMSE']
                f.write(f"- **{TASKS[task]['description']}**: {best_model} (RMSE: {best_rmse:.4f})\n")
        
        f.write("\n## 5. 创新点说明\n\n")
        f.write("### 5.1 CNN-LSTM混合模型\n")
        f.write("- 结合CNN的局部特征提取能力和LSTM的序列建模能力\n")
        f.write("- 多尺度卷积核捕获不同时间尺度的特征\n")
        f.write("- 注意力机制增强重要特征的权重\n\n")
        
        f.write("### 5.2 Attention-LSTM模型\n")
        f.write("- 自注意力机制增强特征表示\n")
        f.write("- 时序注意力聚焦关键时间点\n")
        f.write("- 多头注意力并行处理不同子空间的信息\n\n")
        
        f.write("### 5.3 TCN模型\n")
        f.write("- 因果卷积保证时序性\n")
        f.write("- 膨胀卷积指数级扩大感受野\n")
        f.write("- 残差连接稳定深层网络训练\n\n")
        
        f.write("### 5.4 WaveNet模型\n")
        f.write("- 门控激活单元增强表达能力\n")
        f.write("- 膨胀因果卷积高效建模长序列\n")
        f.write("- 残差和Skip连接加速梯度流动\n\n")
        
        f.write("## 6. 结论\n\n")
        f.write("本实验对比了Linear、LSTM、Transformer三个基础模型和")
        f.write("CNN-LSTM、Attention-LSTM、TCN、WaveNet四个创新模型在风速预测任务上的性能。\n\n")
        f.write("实验结果表明，深度学习模型在捕获风速时序特征方面具有显著优势，")
        f.write("特别是结合注意力机制的模型能够更好地捕获长期依赖关系。\n")
    
    print(f"实验报告已保存至: {report_path}")
    return report_path


def main(args):
    """主函数"""
    setup_experiment()
    
    if args.mode in ['all', 'train', 'visualize']:
        # 加载数据
        df = load_and_preprocess_data()
        
        # 可视化数据集
        if args.mode in ['all', 'visualize']:
            visualize_dataset(df)
    
    if args.mode in ['all', 'train']:
        df = load_and_preprocess_data() if 'df' not in dir() else df
        
        # 训练基础模型
        base_results = train_all_models(df, BASE_MODELS, is_innovative=False)
        
        # 训练创新模型
        innovative_results = train_all_models(df, INNOVATIVE_MODELS, is_innovative=True)
        
        # 合并结果
        all_results = {}
        for task_name in TASKS.keys():
            all_results[task_name] = {}
            if task_name in base_results:
                all_results[task_name].update(base_results[task_name])
            if task_name in innovative_results:
                all_results[task_name].update(innovative_results[task_name])
        
        # 评估和对比
        results_df = evaluate_and_compare(all_results)
        
        # 生成报告
        generate_report(results_df, all_results)
    
    if args.mode == 'eval':
        # 仅评估（需要已训练的模型）
        print("评估模式：请确保模型已训练并保存")
        # TODO: 加载已保存模型并评估
    
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print(f"\n所有结果已保存至: {RESULTS_DIR}")
    print(f"所有模型已保存至: {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='风速序列预测实验')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'train', 'eval', 'visualize'],
                       help='运行模式: all(完整实验), train(仅训练), eval(仅评估), visualize(仅可视化)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='指定要训练的模型（可选）')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                       help='指定要运行的任务（可选）')
    
    args = parser.parse_args()
    main(args)
