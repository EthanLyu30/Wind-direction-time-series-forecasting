# =================================================================
# 快速部署到远程服务器的说明文档
# =================================================================

## 方式一：使用 SCP 上传到Linux服务器

### 1. 打包项目（在Windows上）
```powershell
# 在项目目录下，打包除了大文件外的所有内容
tar -czvf wind_speed_project.tar.gz --exclude='*.pth' --exclude='__pycache__' --exclude='.git' .
```

### 2. 上传到服务器
```bash
scp wind_speed_project.tar.gz username@your-server-ip:/home/username/
```

### 3. 在服务器上解压并运行
```bash
# 登录服务器
ssh username@your-server-ip

# 创建项目目录并解压
mkdir -p ~/wind_speed_prediction
cd ~/wind_speed_prediction
tar -xzvf ~/wind_speed_project.tar.gz

# 运行训练脚本
bash train_remote.sh
```

---

## 方式二：使用 Git（推荐）

### 1. 在服务器上克隆仓库
```bash
ssh username@your-server-ip
git clone https://github.com/你的用户名/wind-speed-prediction.git
cd wind-speed-prediction
```

### 2. 安装依赖并训练
```bash
# 安装GPU版PyTorch（如果有NVIDIA GPU）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip3 install pandas pyarrow numpy scikit-learn matplotlib seaborn tqdm

# 运行训练
python3 main.py
```

---

## 方式三：使用 rsync 同步（适合多次迭代）

```bash
# 首次同步
rsync -avz --exclude='.git' --exclude='*.pth' --exclude='__pycache__' \
    ./ username@server-ip:~/wind_speed_prediction/

# 后续更新只同步修改的文件
rsync -avz --exclude='.git' --exclude='*.pth' --exclude='__pycache__' \
    ./ username@server-ip:~/wind_speed_prediction/
```

---

## GPU服务器配置检查

在服务器上运行以下命令检查GPU状态：

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查PyTorch是否能使用GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## GPU服务器推荐配置

### A100/H100 40GB+（高端GPU，推荐用于大规模训练）

A100有40GB显存，支持混合精度训练(AMP)，训练速度极快：

```bash
# ==================== 首次完整训练 ====================
# 训练所有模型，所有任务
python3 main.py --batch-size 512 --epochs 300 --no-viz

# ==================== 继续训练/微调（推荐） ====================
# 从检查点继续训练所有模型
python3 main.py --resume --batch-size 512 --epochs 400 --no-viz

# 只训练多步预测任务（效果通常需要更多优化）
python3 main.py --resume --tasks multistep_16h --batch-size 512 --epochs 500 --no-viz

# 只训练特定模型
python3 main.py --resume --models LSTM WaveNet LSTNet --batch-size 512 --epochs 400 --no-viz

# ==================== 精细调参 ====================
# 手动指定学习率和早停耐心值
python3 main.py --resume --lr 0.00005 --patience 60 --epochs 500 --no-viz

# 使用R²作为评估指标（推荐用于多步预测）
python3 main.py --resume --tasks multistep_16h --metric-mode r2 --epochs 500 --no-viz
```

系统会自动检测A100并启用：
- **Batch Size**: 512（自动设置）
- **混合精度训练(AMP)**: 自动启用，加速2-3倍
- **多线程数据加载**: 8个workers

### RTX 3090/4090（24GB显存）

```bash
# 首次训练
python3 main.py --batch-size 256 --epochs 250 --no-viz

# 继续训练
python3 main.py --resume --batch-size 256 --epochs 350 --no-viz

# 只优化多步预测
python3 main.py --resume --tasks multistep_16h --batch-size 256 --epochs 400 --no-viz
```

### RTX 3060/3070（8-12GB显存）

```bash
# 首次训练
python3 main.py --batch-size 128 --epochs 200 --no-viz

# 继续训练
python3 main.py --resume --batch-size 128 --epochs 300 --no-viz

# 只训练效果较好的模型
python3 main.py --resume --models Linear LSTM WaveNet --batch-size 128 --epochs 300 --no-viz
```

### 训练参数完整说明

| 参数 | 说明 | A100推荐 | RTX 3090推荐 | RTX 3060推荐 |
|------|------|---------|-------------|-------------|
| `--batch-size` | 批次大小 | 512 | 256 | 128 |
| `--epochs` | 最大训练轮数 | 300-500 | 250-400 | 200-300 |
| `--lr` | 学习率（覆盖默认） | 不指定(自动) | 不指定(自动) | 不指定(自动) |
| `--patience` | 早停耐心值 | 40-60 | 35-50 | 30-40 |
| `--resume` | 从检查点继续 | ✅推荐 | ✅推荐 | ✅推荐 |
| `--no-viz` | 禁用可视化 | ✅服务器必须 | ✅服务器必须 | ✅服务器必须 |
| `--models` | 指定模型 | 可选 | 可选 | 可选 |
| `--tasks` | 指定任务 | 可选 | 可选 | 可选 |
| `--metric-mode` | 评估指标(r2/mse) | r2(多步) | r2(多步) | r2(多步) |

### 任务特定的默认超参数

训练时会自动应用以下任务特定配置：

| 任务 | 学习率 | 早停耐心值 | 最小训练轮数 | 评估指标 |
|------|--------|-----------|-------------|---------|
| singlestep | 0.001 | 20 | 50 | MSE |
| multistep_16h | 0.0001 | 40 | 100 | R² |

### 训练策略建议

1. **首次训练**: 使用默认参数完整训练一遍
   ```bash
   python3 main.py --epochs 200 --no-viz
   ```

2. **查看结果**: 检查 `results/model_comparison.csv`

3. **针对性优化**: 如果某个任务效果不好，单独继续训练
   ```bash
   # multistep_16h 效果差，单独优化
   python3 main.py --resume --tasks multistep_16h --epochs 400 --patience 50 --no-viz
   ```

4. **模型选择优化**: 只训练表现好的模型
   ```bash
   # 只训练LSTM、WaveNet、Linear（通常效果最好）
   python3 main.py --resume --models LSTM WaveNet Linear --epochs 300 --no-viz
   ```

---

## 训练完成后下载模型

```bash
# 从服务器下载训练好的模型
scp -r username@server-ip:~/wind_speed_prediction/models/ ./models_from_server/

# 下载结果
scp -r username@server-ip:~/wind_speed_prediction/results/ ./results_from_server/
```

---

## 训练时间估计

### 单次完整训练（200 epochs，8个模型×2个任务=16个训练任务）

| 设备 | 每个模型/任务 | 完整实验 | 说明 |
|------|-------------|---------|------|
| A100 40GB (AMP) | ~1-3分钟 | ~25-45分钟 | 混合精度自动启用 |
| RTX 3090/4090 | ~2-5分钟 | ~40-80分钟 | 建议启用AMP |
| RTX 3060 12GB | ~3-8分钟 | ~1-2小时 | |
| CPU (多核) | ~15-30分钟 | ~4-8小时 | 不推荐 |

### 继续训练/微调（从检查点恢复，额外200 epochs）

| 设备 | 额外时间 | 说明 |
|------|---------|------|
| A100 40GB | ~20-40分钟 | 如果模型已收敛，早停会更快触发 |
| RTX 3090/4090 | ~30-60分钟 | |
| RTX 3060 | ~1-1.5小时 | |

### 注意事项

- 早停机制会在模型不再改进时自动终止训练，实际时间可能更短
- singlestep任务通常收敛更快（~50-80 epochs）
- multistep_16h任务需要更多训练轮数（~100-200 epochs）
- 使用`--resume`继续训练时，如果模型已达到最优，会很快停止

> 💡 **提示**: 启用混合精度训练(AMP)可加速2-3倍，A100/H100会自动启用，其他GPU可在config.py中手动启用`USE_AMP=True`

---

## 常见问题

### 1. CUDA out of memory
减小batch_size：在 `config.py` 中将 `BATCH_SIZE` 改为 32 或 16

### 2. 服务器没有图形界面，matplotlib报错
在代码开头添加：
```python
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
```

### 3. 权限问题
```bash
chmod +x train_remote.sh
```
