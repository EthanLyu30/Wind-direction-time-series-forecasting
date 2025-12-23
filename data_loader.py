"""
数据加载与预处理模块
功能：
1. 加载三个高度的风速数据
2. 根据时间戳拼接数据
3. 特征工程：处理缺失值和异常值
4. 数据标准化
5. 创建序列数据集
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_PATHS, FEATURE_COLS, TARGET_COL, TIMESTAMP_COL, HEIGHT_COL,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    SINGLE_STEP_INPUT_LEN, SINGLE_STEP_OUTPUT_LEN,
    MULTI_STEP_1_INPUT_LEN, MULTI_STEP_1_OUTPUT_LEN,
    MULTI_STEP_2_INPUT_LEN, MULTI_STEP_2_OUTPUT_LEN,
    BATCH_SIZE
)


def load_parquet_data(data_path):
    """
    加载parquet格式的数据
    
    Args:
        data_path: 数据目录路径
        
    Returns:
        合并后的DataFrame
    """
    train_file = os.path.join(data_path, 'train-00000-of-00001.parquet')
    val_file = os.path.join(data_path, 'val-00000-of-00001.parquet')
    test_file = os.path.join(data_path, 'test-00000-of-00001.parquet')
    
    dfs = []
    for file in [train_file, val_file, test_file]:
        if os.path.exists(file):
            df = pd.read_parquet(file)
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"No parquet files found in {data_path}")


def load_all_data():
    """
    加载所有三个高度的数据并合并
    
    Returns:
        合并后的DataFrame
    """
    all_dfs = []
    
    for height, path in DATA_PATHS.items():
        print(f"加载 {height} 高度数据...")
        df = load_parquet_data(path)
        print(f"  - 数据形状: {df.shape}")
        all_dfs.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并后数据形状: {combined_df.shape}")
    
    return combined_df


def parse_timestamp(df):
    """
    解析时间戳列
    
    Args:
        df: DataFrame
        
    Returns:
        处理后的DataFrame
    """
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=[TIMESTAMP_COL, HEIGHT_COL])
    return df


def handle_missing_values(df, method='interpolate'):
    """
    处理缺失值
    
    Args:
        df: DataFrame
        method: 处理方法 ('interpolate', 'mean', 'median', 'drop')
        
    Returns:
        处理后的DataFrame
    """
    df = df.copy()
    
    # 统计缺失值
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("\n缺失值统计:")
        print(missing_counts[missing_counts > 0])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'interpolate':
        # 使用插值法填充
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    elif method == 'mean':
        # 使用均值填充
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'median':
        # 使用中位数填充
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == 'drop':
        # 删除包含缺失值的行
        df = df.dropna()
    
    # 对于仍然存在的缺失值，使用前向/后向填充
    df = df.ffill().bfill()
    
    return df


def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    处理异常值
    
    Args:
        df: DataFrame
        columns: 要处理的列，None表示所有数值列
        method: 处理方法 ('iqr', 'zscore', 'clip')
        threshold: 阈值
        
    Returns:
        处理后的DataFrame
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除height列
        if HEIGHT_COL in columns:
            columns.remove(HEIGHT_COL)
    
    outlier_counts = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_counts[col] = outliers.sum()
            
            # 使用边界值替换异常值
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            
            outliers = z_scores > threshold
            outlier_counts[col] = outliers.sum()
            
            # 使用均值替换异常值
            df.loc[outliers, col] = mean
            
        elif method == 'clip':
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            outliers = (df[col] < lower) | (df[col] > upper)
            outlier_counts[col] = outliers.sum()
            df[col] = df[col].clip(lower, upper)
    
    print("\n异常值统计:")
    for col, count in outlier_counts.items():
        if count > 0:
            print(f"  {col}: {count} 个异常值")
    
    return df


def add_time_features(df):
    """
    添加时间特征
    
    Args:
        df: DataFrame
        
    Returns:
        添加时间特征后的DataFrame
    """
    df = df.copy()
    
    # 提取时间特征
    df['hour'] = df[TIMESTAMP_COL].dt.hour
    df['day_of_week'] = df[TIMESTAMP_COL].dt.dayofweek
    df['month'] = df[TIMESTAMP_COL].dt.month
    df['day_of_year'] = df[TIMESTAMP_COL].dt.dayofyear
    
    # 周期性编码（使用sin和cos）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def pivot_by_height(df):
    """
    将数据按高度透视，使每个时间点包含所有高度的数据
    
    Args:
        df: DataFrame
        
    Returns:
        透视后的DataFrame
    """
    df = df.copy()
    
    # 选择需要的列
    base_cols = [TIMESTAMP_COL, HEIGHT_COL, TARGET_COL] + FEATURE_COLS
    df_subset = df[base_cols].copy()
    
    # 按时间戳和高度分组，取平均值（处理重复值）
    df_subset = df_subset.groupby([TIMESTAMP_COL, HEIGHT_COL]).mean().reset_index()
    
    # 透视表
    pivoted_dfs = []
    for col in [TARGET_COL] + FEATURE_COLS:
        pivot = df_subset.pivot(index=TIMESTAMP_COL, columns=HEIGHT_COL, values=col)
        pivot.columns = [f'{col}_{h}m' for h in pivot.columns]
        pivoted_dfs.append(pivot)
    
    result = pd.concat(pivoted_dfs, axis=1)
    result = result.reset_index()
    
    return result


def preprocess_data(df):
    """
    完整的数据预处理流程
    
    Args:
        df: 原始DataFrame
        
    Returns:
        预处理后的DataFrame
    """
    print("=" * 50)
    print("数据预处理开始")
    print("=" * 50)
    
    # 1. 解析时间戳
    print("\n1. 解析时间戳...")
    df = parse_timestamp(df)
    
    # 2. 处理缺失值
    print("\n2. 处理缺失值...")
    df = handle_missing_values(df, method='interpolate')
    
    # 3. 处理异常值
    print("\n3. 处理异常值...")
    df = handle_outliers(df, method='iqr', threshold=3.0)
    
    # 4. 按高度透视数据
    print("\n4. 按高度透视数据...")
    df = pivot_by_height(df)
    print(f"  透视后数据形状: {df.shape}")
    
    # 5. 添加时间特征
    print("\n5. 添加时间特征...")
    df = add_time_features(df)
    
    # 6. 最终处理
    print("\n6. 最终数据清洗...")
    df = df.dropna()
    df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)
    
    print(f"\n最终数据形状: {df.shape}")
    print("=" * 50)
    print("数据预处理完成")
    print("=" * 50)
    
    return df


class WindSpeedDataset(Dataset):
    """
    风速预测数据集类
    
    支持单步预测和多步预测
    """
    
    def __init__(self, data, feature_cols, target_cols, input_len, output_len, scaler_features=None, scaler_targets=None):
        """
        初始化数据集
        
        Args:
            data: DataFrame或numpy array
            feature_cols: 特征列名列表
            target_cols: 目标列名列表
            input_len: 输入序列长度
            output_len: 输出序列长度
            scaler_features: 特征标准化器（如果为None则创建新的）
            scaler_targets: 目标标准化器（如果为None则创建新的）
        """
        self.input_len = input_len
        self.output_len = output_len
        
        # 提取特征和目标
        if isinstance(data, pd.DataFrame):
            self.features = data[feature_cols].values.astype(np.float32)
            self.targets = data[target_cols].values.astype(np.float32)
        else:
            self.features = data[:, :len(feature_cols)].astype(np.float32)
            self.targets = data[:, len(feature_cols):].astype(np.float32)
        
        # 标准化
        if scaler_features is None:
            self.scaler_features = StandardScaler()
            self.features = self.scaler_features.fit_transform(self.features)
        else:
            self.scaler_features = scaler_features
            self.features = self.scaler_features.transform(self.features)
        
        if scaler_targets is None:
            self.scaler_targets = StandardScaler()
            self.targets = self.scaler_targets.fit_transform(self.targets)
        else:
            self.scaler_targets = scaler_targets
            self.targets = self.scaler_targets.transform(self.targets)
        
        # 转换为tensor
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.FloatTensor(self.targets)
        
        # 计算有效样本数
        self.num_samples = len(self.features) - input_len - output_len + 1
    
    def __len__(self):
        return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            x: 输入序列 (input_len, num_features)
            y: 目标序列 (output_len, num_targets)
        """
        x = self.features[idx:idx + self.input_len]
        y = self.targets[idx + self.input_len:idx + self.input_len + self.output_len]
        
        return x, y


def get_feature_columns(df):
    """
    获取所有特征列名
    
    Args:
        df: 预处理后的DataFrame
        
    Returns:
        特征列名列表
    """
    feature_cols = []
    
    # 原始特征（三个高度）
    for col in FEATURE_COLS:
        for height in [10, 50, 100]:
            col_name = f'{col}_{height}m'
            if col_name in df.columns:
                feature_cols.append(col_name)
    
    # 时间特征
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    for tf in time_features:
        if tf in df.columns:
            feature_cols.append(tf)
    
    # 添加其他高度的风速作为特征（帮助预测）
    for height in [10, 50, 100]:
        col_name = f'{TARGET_COL}_{height}m'
        if col_name in df.columns:
            feature_cols.append(col_name)
    
    return feature_cols


def get_target_columns(df):
    """
    获取目标列名（三个高度的风速）
    
    Args:
        df: 预处理后的DataFrame
        
    Returns:
        目标列名列表
    """
    target_cols = []
    for height in [10, 50, 100]:
        col_name = f'{TARGET_COL}_{height}m'
        if col_name in df.columns:
            target_cols.append(col_name)
    return target_cols


def create_dataloaders(df, input_len, output_len, batch_size=BATCH_SIZE):
    """
    创建训练、验证、测试数据加载器
    
    Args:
        df: 预处理后的DataFrame
        input_len: 输入序列长度
        output_len: 输出序列长度
        batch_size: 批次大小
        
    Returns:
        train_loader, val_loader, test_loader, scaler_features, scaler_targets, feature_cols, target_cols
    """
    feature_cols = get_feature_columns(df)
    target_cols = get_target_columns(df)
    
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"目标数量: {len(target_cols)}")
    
    # 按时间顺序划分数据集
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")
    
    # 创建数据集（使用训练集的scaler）
    train_dataset = WindSpeedDataset(
        train_df, feature_cols, target_cols, input_len, output_len
    )
    
    val_dataset = WindSpeedDataset(
        val_df, feature_cols, target_cols, input_len, output_len,
        scaler_features=train_dataset.scaler_features,
        scaler_targets=train_dataset.scaler_targets
    )
    
    test_dataset = WindSpeedDataset(
        test_df, feature_cols, target_cols, input_len, output_len,
        scaler_features=train_dataset.scaler_features,
        scaler_targets=train_dataset.scaler_targets
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return (train_loader, val_loader, test_loader, 
            train_dataset.scaler_features, train_dataset.scaler_targets,
            feature_cols, target_cols)


def get_data_statistics(df):
    """
    获取数据统计信息
    
    Args:
        df: DataFrame
        
    Returns:
        统计信息字典
    """
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'describe': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'date_range': (df[TIMESTAMP_COL].min(), df[TIMESTAMP_COL].max()) if TIMESTAMP_COL in df.columns else None,
    }
    return stats


if __name__ == "__main__":
    # 测试数据加载和预处理
    print("测试数据加载模块...")
    
    # 加载数据
    raw_df = load_all_data()
    print(f"\n原始数据列: {raw_df.columns.tolist()}")
    
    # 预处理
    processed_df = preprocess_data(raw_df)
    
    # 获取特征和目标列
    feature_cols = get_feature_columns(processed_df)
    target_cols = get_target_columns(processed_df)
    
    print(f"\n特征列: {feature_cols}")
    print(f"目标列: {target_cols}")
    
    # 创建数据加载器测试
    train_loader, val_loader, test_loader, _, _, _, _ = create_dataloaders(
        processed_df, SINGLE_STEP_INPUT_LEN, SINGLE_STEP_OUTPUT_LEN
    )
    
    # 测试获取一个batch
    for x, y in train_loader:
        print(f"\n输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")
        break
    
    print("\n数据加载模块测试完成！")
