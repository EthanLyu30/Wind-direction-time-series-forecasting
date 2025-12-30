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
