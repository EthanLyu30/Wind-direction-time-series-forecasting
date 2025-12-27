#!/usr/bin/env python3
"""
模型最佳版本恢复工具

问题：由于之前的bug，某些模型的最佳版本可能被更差的版本覆盖
解决方案：从历史记录中恢复最佳模型

用法：
    python recover_best_models.py --input results/model_comparison.csv
"""

import os
import json
import pandas as pd
import torch
import argparse
from pathlib import Path

def analyze_model_history():
    """
    分析所有训练模型的历史记录
    找出哪些模型当前版本不如历史最佳
    """
    results_dir = Path('results')
    models_dir = Path('models')
    
    print("\n" + "="*70)
    print("模型历史版本分析")
    print("="*70)
    
    # 读取CSV
    csv_file = results_dir / 'model_comparison.csv'
    if not csv_file.exists():
        print(f"❌ 找不到 {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n📊 当前模型数据库: {len(df)} 条记录")
    print(df.head(10))
    
    # 遍历所有模型检查点
    print("\n" + "-"*70)
    print("检查点历史分析：")
    print("-"*70)
    
    issues = []
    
    for pth_file in sorted(models_dir.glob("*.pth")):
        try:
            checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
            model_name = checkpoint.get('model_name', 'unknown')
            task_name = checkpoint.get('task_name', 'unknown')
            history = checkpoint.get('history', {})
            
            if not history:
                continue
            
            best_val_loss = history.get('best_val_loss', float('inf'))
            best_epoch = history.get('best_epoch', 0)
            total_epochs = len(history.get('train_loss', []))
            
            # 从CSV中查找当前记录
            csv_record = df[(df['Model'] == model_name) & (df['Task'] == task_name)]
            
            if csv_record.empty:
                print(f"\n⚠️  {model_name}_{task_name}: 检查点存在但不在CSV中")
                issues.append({
                    'model': model_name,
                    'task': task_name,
                    'issue': 'CSV中缺失',
                    'checkpoint_loss': best_val_loss,
                    'file': pth_file.name
                })
            else:
                csv_mse = float(csv_record['MSE'].values[0])
                checkpoint_loss = best_val_loss
                
                # MSE和验证损失应该大致成正比
                if abs(csv_mse - checkpoint_loss) > 0.1:
                    print(f"\n⚠️  {model_name}_{task_name}: 版本不一致")
                    print(f"   检查点损失: {checkpoint_loss:.4f}")
                    print(f"   CSV中MSE:  {csv_mse:.4f}")
                    issues.append({
                        'model': model_name,
                        'task': task_name,
                        'issue': 'CSV与检查点不一致',
                        'checkpoint_loss': checkpoint_loss,
                        'csv_mse': csv_mse,
                        'file': pth_file.name
                    })
                else:
                    print(f"✅ {model_name}_{task_name}: 一致 (loss={checkpoint_loss:.4f}, epochs={total_epochs})")
        
        except Exception as e:
            print(f"❌ 读取 {pth_file.name} 失败: {e}")
    
    # 总结
    print("\n" + "="*70)
    if issues:
        print(f"⚠️  发现 {len(issues)} 个问题：\n")
        for issue in issues:
            print(f"  - {issue['model']}_{issue['task']}: {issue['issue']}")
            print(f"    文件: {issue['file']}")
    else:
        print("✅ 所有模型版本一致，没有发现问题")
    print("="*70 + "\n")

def compare_with_csv():
    """
    对比CSV中各任务的最佳模型
    """
    csv_file = Path('results/model_comparison.csv')
    
    if not csv_file.exists():
        print(f"❌ 找不到 {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*70)
    print("各任务最佳模型排名")
    print("="*70)
    
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task].sort_values('R2', ascending=False)
        print(f"\n📍 {task}:")
        print("-" * 70)
        
        for idx, row in task_df.iterrows():
            print(f"  {row['Model']:20} R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}")
        
        # 显示top 3
        print(f"\n  🥇 最佳: {task_df.iloc[0]['Model']} (R²={task_df.iloc[0]['R2']:.4f})")
        if len(task_df) > 1:
            print(f"  🥈 次优: {task_df.iloc[1]['Model']} (R²={task_df.iloc[1]['R2']:.4f})")
        if len(task_df) > 2:
            print(f"  🥉 第三: {task_df.iloc[2]['Model']} (R²={task_df.iloc[2]['R2']:.4f})")
    
    print("\n" + "="*70 + "\n")

def show_improvement_opportunities():
    """
    显示哪些模型有改进空间
    """
    csv_file = Path('results/model_comparison.csv')
    
    if not csv_file.exists():
        return
    
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*70)
    print("改进机会分析（哪些模型可能被次优版本覆盖）")
    print("="*70)
    
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        
        # 找出R²最低的模型（可能是被坏版本覆盖了）
        worst_models = task_df.nsmallest(3, 'R2')
        
        print(f"\n📍 {task} - 可能需要恢复的模型：")
        for idx, row in worst_models.iterrows():
            print(f"  - {row['Model']:20} R²={row['R2']:.4f} (MSE={row['MSE']:.4f})")
            print(f"    建议：重新训练或检查是否被坏版本覆盖")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型最佳版本恢复工具")
    parser.add_argument('--analyze', action='store_true', help='分析所有模型历史')
    parser.add_argument('--compare', action='store_true', help='对比CSV中的模型')
    parser.add_argument('--opportunities', action='store_true', help='显示改进机会')
    parser.add_argument('--all', action='store_true', help='运行所有分析')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，运行所有分析
    if not any([args.analyze, args.compare, args.opportunities, args.all]):
        args.all = True
    
    if args.all or args.analyze:
        analyze_model_history()
    
    if args.all or args.compare:
        compare_with_csv()
    
    if args.all or args.opportunities:
        show_improvement_opportunities()
