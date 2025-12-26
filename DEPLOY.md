# =================================================================
# 快速部署到远程服务器的说明文档
# =================================================================

## 🚀 快速开始（服务器训练）

```bash
# 基础训练（禁用可视化，避免Qt问题）
python main.py --mode train --no-viz

# 指定训练轮数和早停耐心值
python main.py --mode train --no-viz --epochs 200 --patience 25

# 只训练特定模型
python main.py --mode train --no-viz --models LSTM Transformer

# 继续训练已有模型（迭代优化）
python main.py --mode train --no-viz --resume --epochs 300

# 调整学习率和batch size
python main.py --mode train --no-viz --lr 0.0005 --batch-size 256
```

## 📋 完整命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 运行模式 | `train`, `eval`, `visualize`, `all` |
| `--no-viz` | 禁用可视化（服务器必选） | `--no-viz` |
| `--models` | 指定训练的模型 | `--models LSTM Transformer WaveNet` |
| `--tasks` | 指定训练的任务 | `--tasks singlestep multistep_16h` |
| `--epochs` | 训练轮数 | `--epochs 200` |
| `--batch-size` | 批次大小 | `--batch-size 256` |
| `--lr` | 学习率 | `--lr 0.0005` |
| `--patience` | 早停耐心值 | `--patience 25` |
| `--resume` | 继续训练已有模型 | `--resume` |

## 🔄 可用模型列表

**基础模型:** `Linear`, `LSTM`, `Transformer`

**创新模型:** `CNN_LSTM`, `Attention_LSTM`, `TCN`, `WaveNet`

---

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

## RTX 3060 GPU服务器推荐配置

RTX 3060有12GB显存，可以适当增加batch_size提升训练速度：

在 `config.py` 中修改：
```python
BATCH_SIZE = 128  # 从64增加到128（GPU显存充足时）
NUM_EPOCHS = 100
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

| 设备 | 单个模型（单步预测） | 完整实验（21个模型） |
|------|---------------------|---------------------|
| CPU (Windows) | ~10-15分钟 | ~3-5小时 |
| RTX 3060 GPU | ~1-2分钟 | ~30-45分钟 |
| 云服务器 (CPU) | ~8-12分钟 | ~2-4小时 |

---

## 常见问题

### 1. CUDA out of memory
减小batch_size：
```bash
python main.py --mode train --no-viz --batch-size 32
```

### 2. 服务器没有图形界面，matplotlib报错
使用 `--no-viz` 参数禁用可视化：
```bash
python main.py --mode train --no-viz
```

### 3. 权限问题
```bash
chmod +x train_remote.sh
```

### 4. 想要迭代优化模型而不是从头训练
使用 `--resume` 参数继续训练：
```bash
# 第一次训练100轮
python main.py --mode train --no-viz --epochs 100

# 继续训练到200轮（自动加载已有模型）
python main.py --mode train --no-viz --resume --epochs 200

# 只继续训练特定模型
python main.py --mode train --no-viz --resume --models LSTM --epochs 300
```

### 5. 只想训练部分模型
```bash
# 只训练LSTM和Transformer
python main.py --mode train --no-viz --models LSTM Transformer

# 只训练创新模型
python main.py --mode train --no-viz --models CNN_LSTM Attention_LSTM TCN WaveNet
```
