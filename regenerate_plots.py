"""重新生成所有图表，修复字体问题"""
import pandas as pd
from visualization import plot_model_comparison

# 加载结果
df = pd.read_csv('results/model_comparison.csv')

# 重新生成对比图
plot_model_comparison(df, save_path='results/model_comparison.png')
print('图表已重新生成')
