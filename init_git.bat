@echo off
REM Git仓库初始化脚本 (Windows版)
REM 在项目根目录运行此脚本

echo === 初始化Git仓库 ===

REM 初始化git仓库
git init

REM 添加所有文件
git add .

REM 创建初始提交
git commit -m "Initial commit: 风速序列预测项目"

echo.
echo === Git仓库初始化完成 ===
echo.
echo 接下来请执行以下步骤:
echo 1. 在GitHub上创建新仓库 (建议名称: wind-speed-prediction)
echo 2. 运行以下命令关联远程仓库:
echo    git remote add origin https://github.com/你的用户名/wind-speed-prediction.git
echo    git branch -M main
echo    git push -u origin main

pause
