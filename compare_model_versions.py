#!/usr/bin/env python3
"""
模型版本对比脚本
展示新旧数据的对比，帮助识别被覆盖的最佳模型
"""

import pandas as pd
from pathlib import Path

# 新数据（当前的结果）
new_data = {
    'Model': [
        'Attention_LSTM', 'CNN_LSTM', 'LSTM', 'Linear', 'TCN', 'Transformer', 'WaveNet'
    ],
    'Task': ['multistep_16h'] * 7,
    'MSE_new': [4.089910, 4.777959, 5.262277, 3.925741, 5.716832, 6.543939, 6.172311],
    'R2_new': [0.455850, 0.364307, 0.299870, 0.477692, 0.239393, 0.129349, 0.178793]
}

# 旧数据（历史最好的结果）
old_data = {
    'MSE_old': [3.8546, 4.0583, 4.7978, 4.3853, 4.8017, 5.9359, 5.8641],
    'R2_old': [0.4871, 0.4601, 0.3617, 0.4165, 0.3611, 0.2102, 0.2198]
}

def analyze_model_degradation():
    """分析模型性能下降情况"""
    
    df = pd.DataFrame(new_data)
    df['MSE_old'] = old_data['MSE_old']
    df['R2_old'] = old_data['R2_old']
    
    # 计算变化
    df['MSE_change'] = df['MSE_new'] - df['MSE_old']
    df['R2_change'] = df['R2_new'] - df['R2_old']
    df['R2_change_pct'] = (df['R2_change'] / df['R2_old']) * 100
    
    print("\n" + "="*100)
    print("模型性能对比分析（multistep_16h）")
    print("="*100)
    print("\n{:<20} {:<10} {:<10} {:<12} {:<12} {:<10}".format(
        "模型", "旧R²", "新R²", "变化", "变化%", "状态"
    ))
    print("-"*100)
    
    # 按R²_old排序
    df_sorted = df.sort_values('R2_old', ascending=False)
    
    total_improved = 0
    total_degraded = 0
    
    for _, row in df_sorted.iterrows():
        model = row['Model']
        r2_old = row['R2_old']
        r2_new = row['R2_new']
        r2_change = row['R2_change']
        r2_pct = row['R2_change_pct']
        
        if r2_new > r2_old:
            status = "✅ 改进"
            total_improved += 1
        elif r2_new == r2_old:
            status = "➡️  无变"
        else:
            status = "❌ 下降"
            total_degraded += 1
        
        print("{:<20} {:<10.4f} {:<10.4f} {:<12.4f} {:<12.2f} {:<10}".format(
            model, r2_old, r2_new, r2_change, r2_pct, status
        ))
    
    print("="*100)
    print(f"\n📊 统计：改进 {total_improved} 个 | 下降 {total_degraded} 个")
    print("\n🔴 被覆盖的最佳模型（应该恢复）：")
    
    degraded_models = df_sorted[df_sorted['R2_new'] < df_sorted['R2_old']].sort_values('R2_change_pct')
    
    for idx, row in degraded_models.iterrows():
        loss_pct = abs(row['R2_change_pct'])
        print(f"  • {row['Model']:20} 损失了 {loss_pct:5.1f}% (R² {row['R2_old']:.4f} → {row['R2_new']:.4f})")
    
    print("\n✅ 改进的模型（保留新版本）：")
    improved_models = df_sorted[df_sorted['R2_new'] > df_sorted['R2_old']]
    
    if len(improved_models) > 0:
        for idx, row in improved_models.iterrows():
            gain_pct = row['R2_change_pct']
            print(f"  • {row['Model']:20} 获得了 {gain_pct:5.1f}% (R² {row['R2_old']:.4f} → {row['R2_new']:.4f})")
    else:
        print("  • 无")
    
    print("\n" + "="*100)
    
    # 建议
    print("\n💡 建议行动：\n")
    print("1. 代码修复已完成（trainer.py）")
    print("   - 新训练不会再覆盖历史最好模型")
    print("   - 会自动对比新旧并选择最优版本")
    print()
    print("2. 立即恢复被覆盖的模型：")
    print("   python main.py --models {} --tasks multistep_16h \\".format(
        ' '.join(degraded_models['Model'].values[:3])))
    print("     --epochs 100 --batch-size 128 --lr 0.0003 --patience 25 --resume")
    print()
    print("3. 验证修复结果：")
    print("   python recover_best_models.py --compare")
    print()
    print("="*100 + "\n")

if __name__ == "__main__":
    analyze_model_degradation()
