"""
快速测试脚本 - 验证项目结构和代码正确性
使用小数据量快速运行一遍完整流程
"""
import os
import sys

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    try:
        import config
        print("  ✓ config.py")
        
        import data_loader
        print("  ✓ data_loader.py")
        
        import models
        print("  ✓ models.py")
        
        import models_innovative
        print("  ✓ models_innovative.py")
        
        import trainer
        print("  ✓ trainer.py")
        
        import visualization
        print("  ✓ visualization.py")
        
        print("\n所有模块导入成功！")
        return True
    except Exception as e:
        print(f"\n导入失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n测试数据加载...")
    try:
        from data_loader import load_all_data, preprocess_data
        
        raw_df = load_all_data()
        print(f"  ✓ 原始数据加载成功，形状: {raw_df.shape}")
        
        processed_df = preprocess_data(raw_df)
        print(f"  ✓ 数据预处理成功，形状: {processed_df.shape}")
        
        return processed_df
    except Exception as e:
        print(f"\n数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_models():
    """测试模型创建"""
    print("\n测试模型创建...")
    import torch
    
    batch_size = 4
    input_len = 8
    output_len = 1
    num_features = 20
    num_targets = 3
    
    x = torch.randn(batch_size, input_len, num_features)
    
    try:
        from models import get_model
        
        for name in ['Linear', 'LSTM', 'Transformer']:
            model = get_model(name, input_len, output_len, num_features, num_targets)
            output = model(x)
            assert output.shape == (batch_size, output_len, num_targets)
            print(f"  ✓ {name} 模型测试通过")
        
        from models_innovative import get_innovative_model
        
        for name in ['CNN_LSTM', 'Attention_LSTM', 'TCN', 'WaveNet']:
            model = get_innovative_model(name, input_len, output_len, num_features, num_targets)
            output = model(x)
            assert output.shape == (batch_size, output_len, num_targets)
            print(f"  ✓ {name} 创新模型测试通过")
        
        print("\n所有模型测试通过！")
        return True
    except Exception as e:
        print(f"\n模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_quick(df):
    """快速训练测试"""
    print("\n快速训练测试...")
    try:
        from data_loader import create_dataloaders, get_feature_columns, get_target_columns
        from models import get_model
        from trainer import train_model
        from config import DEVICE
        
        # 创建数据加载器
        train_loader, val_loader, test_loader, sf, st, fc, tc = create_dataloaders(
            df, input_len=8, output_len=1, batch_size=32
        )
        
        # 创建模型
        model = get_model('Linear', 8, 1, len(fc), len(tc))
        
        # 快速训练
        history = train_model(
            model, train_loader, val_loader,
            model_name='Linear_test',
            task_name='quick_test',
            num_epochs=3,  # 只训练3轮
            device=DEVICE,
            save_best=False,
            verbose=True
        )
        
        print(f"\n  ✓ 快速训练完成")
        print(f"    最终训练损失: {history['train_loss'][-1]:.4f}")
        print(f"    最终验证损失: {history['val_loss'][-1]:.4f}")
        
        return True
    except Exception as e:
        print(f"\n训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("风速预测项目 - 快速测试")
    print("=" * 60)
    
    # 测试导入
    if not test_imports():
        return False
    
    # 测试模型
    if not test_models():
        return False
    
    # 测试数据加载
    df = test_data_loading()
    if df is None:
        return False
    
    # 快速训练测试
    if not test_training_quick(df):
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！项目可以正常运行")
    print("=" * 60)
    print("\n运行完整实验: python main.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
