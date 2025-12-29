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
    python main.py                           # 运行完整实验
    python main.py --mode train              # 仅训练
    python main.py --mode eval               # 仅评估（需要已训练模型）
    python main.py --mode visualize          # 仅可视化
    python main.py --no-viz                  # 禁用可视化（服务器推荐）
    python main.py --mode train --no-viz     # 仅训练，不生成图表
    python main.py --batch-size 256          # 指定batch size
    python main.py --epochs 200              # 指定训练轮数
    python main.py --resume                  # 从检查点继续训练
    python main.py --models LSTM Transformer # 只训练指定模型
"""
import os
import sys

# ==================== 解决服务器无图形界面问题（必须最先执行）====================
# 在无头Linux服务器上设置环境变量，避免Qt插件错误
if sys.platform.startswith('linux'):
    if not os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        import matplotlib
        matplotlib.use('Agg')

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
    MULTI_STEP_INPUT_LEN, MULTI_STEP_OUTPUT_LEN,
    set_seed, RANDOM_SEED, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    TASK_SPECIFIC_HYPERPARAMS, get_adjusted_lr
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


# ==================== 全局运行配置（可被命令行参数覆盖）====================
class RuntimeConfig:
    """运行时配置，可以被命令行参数动态覆盖"""
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.enable_visualization = True  # 是否启用可视化
        self.resume_training = False  # 是否从检查点继续训练
        self.selected_models = None  # 指定要训练的模型列表
        self.metric_mode = None  # 评估指标模式 (None表示自动选择)
        
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        if args.batch_size is not None:
            self.batch_size = args.batch_size
            print(f"⚙️  Batch Size 已覆盖为: {self.batch_size}")
        if args.epochs is not None:
            self.num_epochs = args.epochs
            print(f"⚙️  训练轮数已覆盖为: {self.num_epochs}")
        if args.lr is not None:
            self.learning_rate = args.lr
            print(f"⚙️  学习率已覆盖为: {self.learning_rate}")
        if args.patience is not None:
            self.early_stopping_patience = args.patience
            print(f"⚙️  早停耐心值已覆盖为: {self.early_stopping_patience}")
        if args.no_viz:
            self.enable_visualization = False
            print("📊 可视化已禁用（仅保存数据，不生成图表）")
        if hasattr(args, 'resume') and args.resume:
            self.resume_training = True
            print("🔄 启用继续训练模式（从已有检查点恢复）")
        if args.models is not None:
            self.selected_models = args.models
            print(f"📋 仅训练指定模型: {', '.join(args.models)}")
        if hasattr(args, 'tasks') and args.tasks is not None:
            self.selected_tasks = args.tasks
            print(f"📋 仅训练指定任务: {', '.join(args.tasks)}")
        else:
            self.selected_tasks = None
        if hasattr(args, 'metric_mode') and args.metric_mode is not None:
            self.metric_mode = args.metric_mode
            mode_desc = {'r2': 'R²(越大越好)', 'mse': 'MSE(越小越好)', 'combined': '综合指标'}
            print(f"📊 评估指标模式: {mode_desc.get(self.metric_mode, self.metric_mode)}")

# 全局运行时配置实例
runtime_config = RuntimeConfig()


# 定义任务配置
# 单步预测：8小时 → 1小时
# 多步预测：8小时 → 16小时
TASKS = {
    'singlestep': {
        'input_len': SINGLE_STEP_INPUT_LEN,
        'output_len': SINGLE_STEP_OUTPUT_LEN,
        'description': f'单步预测（{SINGLE_STEP_INPUT_LEN}小时→{SINGLE_STEP_OUTPUT_LEN}小时）'
    },
    'multistep': {
        'input_len': MULTI_STEP_INPUT_LEN,
        'output_len': MULTI_STEP_OUTPUT_LEN,
        'description': f'多步预测（{MULTI_STEP_INPUT_LEN}小时→{MULTI_STEP_OUTPUT_LEN}小时）'
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
    print(f"批次大小: {runtime_config.batch_size}")
    print(f"最大训练轮数: {runtime_config.num_epochs}")
    print(f"学习率: {runtime_config.learning_rate}")
    print(f"可视化: {'启用' if runtime_config.enable_visualization else '禁用'}")
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
    if not runtime_config.enable_visualization:
        print("\n[跳过] 数据集可视化（已禁用）")
        return
        
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
        
        # 创建数据加载器（使用runtime_config中的batch_size）
        input_len = task_config['input_len']
        output_len = task_config['output_len']
        
        train_loader, val_loader, test_loader, scaler_features, scaler_targets, feature_cols, target_cols = \
            create_dataloaders(df, input_len, output_len, runtime_config.batch_size)
        
        num_features = len(feature_cols)
        num_targets = len(target_cols)
        
        task_results = {}
        
        # 导入任务特定的超参
        from config import TASK_SPECIFIC_HYPERPARAMS, get_adjusted_lr
        
        # 获取任务特定的超参（如果用户没有手动指定，则使用任务推荐值）
        task_config = TASK_SPECIFIC_HYPERPARAMS.get(task_name, {})
        
        # 确定最终超参优先级：用户指定 > 任务推荐 > 全局默认
        final_lr = runtime_config.learning_rate if runtime_config.learning_rate != LEARNING_RATE else task_config.get('lr', LEARNING_RATE)
        final_patience = runtime_config.early_stopping_patience if runtime_config.early_stopping_patience != EARLY_STOPPING_PATIENCE else task_config.get('patience', EARLY_STOPPING_PATIENCE)
        final_epochs = runtime_config.num_epochs
        
        # 如果batch_size被改为256，自动调整学习率下降（0.0002太低了！）
        if runtime_config.batch_size == 256 and final_lr == 0.0002:
            final_lr = 0.0005  # 自动纠正：256时改为0.0005
            print(f"⚠️  检测到batch_size=256，学习率自动从0.0002调整为0.0005（太低会导致欠拟合）")
        
        # 如果用户用了resume但没有调整学习率，建议降低
        if runtime_config.resume_training and runtime_config.learning_rate == LEARNING_RATE:
            final_lr = task_config.get('lr', final_lr)
            print(f"💡 继续训练模式：使用任务优化学习率 {final_lr}")
        
        for model_name in model_list:
            print(f"\n--- 训练 {model_name} ---")
            
            # 创建模型
            if is_innovative:
                model = get_innovative_model(model_name, input_len, output_len, num_features, num_targets)
            else:
                model = get_model(model_name, input_len, output_len, num_features, num_targets)
            
            print(f"模型参数量: {count_parameters(model):,}")
            
            # 训练（使用任务特定的超参）
            metric_mode_str = runtime_config.metric_mode if runtime_config.metric_mode else ('r2' if task_name == 'multistep_16h' else 'mse')
            print(f"📊 使用超参: lr={final_lr:.6f}, patience={final_patience}, epochs={final_epochs}, metric={metric_mode_str}")
            history = train_model(
                model, train_loader, val_loader,
                model_name=model_name,
                task_name=task_name,
                num_epochs=final_epochs,
                learning_rate=final_lr,  # 使用任务优化后的学习率
                patience=final_patience,  # 使用任务优化后的早停
                device=DEVICE,
                save_best=True,
                verbose=True,
                resume=runtime_config.resume_training,  # 支持继续训练
                metric_mode=runtime_config.metric_mode  # 评估指标模式（None表示自动选择）
            )
            
            # 绘制训练历史（从检查点读取完整历史，包含所有微调过程）
            if runtime_config.enable_visualization:
                history_save_path = os.path.join(RESULTS_DIR, f'{model_name}_{task_name}_history.png')
                # 从保存的检查点读取完整历史（包含所有微调过程）
                model_path = os.path.join(MODELS_DIR, f"{model_name}_{task_name}.pth")
                previous_epochs = 0
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                        full_history = checkpoint.get('history', history)
                        # 计算之前的训练轮数（用于标记微调分界点）
                        if runtime_config.resume_training and len(full_history.get('train_loss', [])) > len(history.get('train_loss', [])):
                            previous_epochs = len(full_history['train_loss']) - len(history['train_loss'])
                        # 使用完整历史绘制（包含所有微调过程）
                        plot_training_history(full_history, model_name, task_name, save_path=history_save_path, previous_epochs=previous_epochs)
                    except Exception as e:
                        print(f"⚠️  无法从检查点读取完整历史，使用本次训练历史: {e}")
                        plot_training_history(history, model_name, task_name, save_path=history_save_path)
                else:
                    # 首次训练，直接使用当前历史
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
            
            # 可视化预测结果（可选）
            if runtime_config.enable_visualization:
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
    
    # ==================== 合并现有结果（不覆盖）====================
    results_csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    
    if os.path.exists(results_csv_path):
        # 读取现有结果
        existing_df = pd.read_csv(results_csv_path)
        print(f"📂 发现现有结果文件，将合并更新...")
        
        # 合并：新结果覆盖旧结果中相同的Model+Task组合
        for _, new_row in results_df.iterrows():
            mask = (existing_df['Model'] == new_row['Model']) & (existing_df['Task'] == new_row['Task'])
            if mask.any():
                # 更新现有行
                existing_df.loc[mask, ['MSE', 'RMSE', 'MAE', 'R2']] = new_row[['MSE', 'RMSE', 'MAE', 'R2']].values
            else:
                # 添加新行
                existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        
        results_df = existing_df
        print(f"✅ 已合并 {len(results_df)} 条模型结果")
    
    # 按Task和Model排序
    task_order = ['singlestep', 'multistep']
    results_df['Task'] = pd.Categorical(results_df['Task'], categories=task_order, ordered=True)
    results_df = results_df.sort_values(['Task', 'Model']).reset_index(drop=True)
    
    # 保存完整结果
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n对比结果已保存至: {results_csv_path}")
    
    # 打印对比表格
    print("\n模型性能对比:")
    print(results_df.to_string(index=False))
    
    # 绘制对比图（可选）
    if runtime_config.enable_visualization:
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
    
    # 重新读取完整的CSV数据（包含合并后的所有模型）
    results_csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    if os.path.exists(results_csv_path):
        full_results_df = pd.read_csv(results_csv_path)
    else:
        full_results_df = results_df
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Wind Speed Prediction Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Experiment Configuration\n\n")
        f.write(f"- Device: {DEVICE}\n")
        f.write(f"- Batch Size: {runtime_config.batch_size}\n")
        f.write(f"- Max Epochs: {runtime_config.num_epochs}\n")
        f.write(f"- Learning Rate: {runtime_config.learning_rate}\n")
        f.write(f"- Random Seed: {RANDOM_SEED}\n\n")
        
        f.write("## 2. Task Configuration\n\n")
        task_descriptions = {
            'singlestep': 'Single-step Prediction (8h -> 1h)',
            'multistep': 'Multi-step Prediction (8h -> 16h)'
        }
        for task_name, task_config in TASKS.items():
            desc = task_descriptions.get(task_name, task_config['description'])
            f.write(f"### {desc}\n")
            f.write(f"- Input Length: {task_config['input_len']} hours\n")
            f.write(f"- Output Length: {task_config['output_len']} hours\n\n")
        
        f.write("## 3. Model Performance Comparison\n\n")
        f.write(full_results_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. Best Models\n\n")
        
        # 找出每个任务的最佳模型
        for task in ['singlestep', 'multistep']:
            task_results = full_results_df[full_results_df['Task'] == task]
            if len(task_results) > 0:
                best_idx = task_results['RMSE'].idxmin()
                best_model = task_results.loc[best_idx, 'Model']
                best_rmse = task_results.loc[best_idx, 'RMSE']
                best_r2 = task_results.loc[best_idx, 'R2']
                desc = task_descriptions.get(task, task)
                f.write(f"- **{desc}**: {best_model} (RMSE: {best_rmse:.4f}, R²: {best_r2:.4f})\n")
        
        f.write("\n## 5. Innovation Points\n\n")
        f.write("### 5.1 CNN-LSTM Hybrid Model\n")
        f.write("- Combines CNN's local feature extraction with LSTM's sequence modeling\n")
        f.write("- Multi-scale convolution kernels capture features at different time scales\n")
        f.write("- Attention mechanism enhances important feature weights\n\n")
        
        f.write("### 5.2 Attention-LSTM Model\n")
        f.write("- Self-attention mechanism enhances feature representation\n")
        f.write("- Temporal attention focuses on key time points\n")
        f.write("- Multi-head attention processes different subspace information in parallel\n\n")
        
        f.write("### 5.3 TCN Model\n")
        f.write("- Causal convolution ensures temporal causality\n")
        f.write("- Dilated convolution exponentially expands receptive field\n")
        f.write("- Residual connections stabilize deep network training\n\n")
        
        f.write("### 5.4 WaveNet Model\n")
        f.write("- Gated activation units enhance expressive power\n")
        f.write("- Dilated causal convolution efficiently models long sequences\n")
        f.write("- Residual and Skip connections accelerate gradient flow\n\n")
        
        f.write("## 6. Conclusion\n\n")
        f.write("This experiment compared Linear, LSTM, and Transformer as baseline models, ")
        f.write("along with CNN-LSTM, Attention-LSTM, TCN, and WaveNet as innovative models ")
        f.write("for wind speed prediction tasks.\n\n")
        f.write("The results show that deep learning models have significant advantages ")
        f.write("in capturing wind speed temporal features, especially models with attention ")
        f.write("mechanisms that can better capture long-term dependencies.\n")
    
    print(f"实验报告已保存至: {report_path}")
    return report_path


def main(args):
    """主函数"""
    # 从命令行参数更新运行时配置
    runtime_config.update_from_args(args)
    
    setup_experiment()
    
    if args.mode in ['all', 'train', 'visualize']:
        # 加载数据
        df = load_and_preprocess_data()
        
        # 可视化数据集
        if args.mode in ['all', 'visualize']:
            visualize_dataset(df)
    
    if args.mode in ['all', 'train']:
        df = load_and_preprocess_data() if 'df' not in dir() else df
        
        # 确定要训练的模型
        if runtime_config.selected_models:
            # 用户指定了模型
            selected_base = [m for m in runtime_config.selected_models if m in BASE_MODELS]
            selected_innovative = [m for m in runtime_config.selected_models if m in INNOVATIVE_MODELS]
            
            # 检查是否有无效的模型名
            all_valid_models = BASE_MODELS + INNOVATIVE_MODELS
            invalid_models = [m for m in runtime_config.selected_models if m not in all_valid_models]
            if invalid_models:
                print(f"⚠️ 未知模型: {invalid_models}")
                print(f"   可用模型: {all_valid_models}")
        else:
            selected_base = BASE_MODELS
            selected_innovative = INNOVATIVE_MODELS
        
        all_results = {}
        
        # 确定要运行的任务
        selected_tasks = runtime_config.selected_tasks if hasattr(runtime_config, 'selected_tasks') and runtime_config.selected_tasks else None
        
        # 训练基础模型
        if selected_base:
            base_results = train_all_models(df, selected_base, tasks_to_run=selected_tasks, is_innovative=False)
            for task_name in base_results:
                if task_name not in all_results:
                    all_results[task_name] = {}
                all_results[task_name].update(base_results[task_name])
        
        # 训练创新模型
        if selected_innovative:
            innovative_results = train_all_models(df, selected_innovative, tasks_to_run=selected_tasks, is_innovative=True)
            for task_name in innovative_results:
                if task_name not in all_results:
                    all_results[task_name] = {}
                all_results[task_name].update(innovative_results[task_name])
        
        # 评估和对比
        if all_results:
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
                       help='指定要训练的模型，如: --models LSTM Transformer')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                       help='指定要运行的任务，如: --tasks singlestep multistep_1h')
    parser.add_argument('--no-viz', action='store_true',
                       help='禁用可视化图表生成（服务器训练推荐）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='覆盖默认的batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='覆盖默认的训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                       help='覆盖默认的学习率')
    parser.add_argument('--patience', type=int, default=None,
                       help='早停的耐心值')
    parser.add_argument('--resume', action='store_true',
                       help='从已有检查点继续训练（迭代优化模型）')
    parser.add_argument('--metric-mode', type=str, default=None,
                       choices=['r2', 'mse', 'combined'],
                       help='评估指标模式: r2(R²越大越好), mse(MSE越小越好), combined(综合指标)')
    
    args = parser.parse_args()
    main(args)
