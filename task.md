# 任务执行指南

本文档提供了在酵母细胞检测项目中执行各种任务的明确步骤和注意事项，以避免常见错误。请按照以下指南操作。

## 文件和目录清理操作

### 安全清理步骤

1. **始终先执行演示模式**
   ```bash
   python cleanup_data.py --dry-run
   ```
   确认将被删除的文件列表，验证没有需要保留的文件。

2. **直接删除模式（推荐）**
   ```bash
   python cleanup_data.py
   ```
   这会直接删除冗余文件。系统会要求确认操作，输入'y'进行确认。

3. **强制删除模式（高级）**
   ```bash
   python cleanup_data.py --force
   ```
   直接删除文件，不进行确认提示。

### 冗余文件处理原则

1. **不需要保留冗余文件**
   - 冗余文件会占用空间并可能导致导入错误
   - 在代码重构完成后，旧版模块应该被完全删除
   - 项目中不应保留 `.bak` 等备份文件

2. **在迁移测试后立即清理**
   ```bash
   # 运行测试
   python tools/test_migration.py

   # 测试成功后立即清理
   python cleanup_data.py
   ```

3. **确保代码依赖已更新**
   - 在清理前确保所有导入语句已更新到新模块
   - 使用 `python tools/check_migration.py --source_dir .` 检查

### 避免的错误操作

- ❌ **不要使用通配符直接删除文件**
  ```bash
  # 危险操作，请勿使用
  rm -rf *
  ```

- ❌ **不要在未确认内容前删除目录**
  ```bash
  # 危险操作，请勿使用
  Remove-Item -Recurse -Force [目录]
  ```

- ✅ **使用指定路径和错误处理**
  ```bash
  # 安全做法
  Remove-Item -Path [精确路径] -ErrorAction SilentlyContinue
  ```

## 数据处理操作

### 图像预处理

1. **指定完整的参数路径**
   ```bash
   python process_data.py --mode preprocess --raw_dir data/raw --output_dir data/processed
   ```

2. **避免默认参数导致的问题**
   ```bash
   # 错误做法
   python process_data.py

   # 正确做法
   python process_data.py --mode preprocess --img_size 640 --raw_dir data/raw
   ```

### 脚本执行

1. **保持命令一致性**
   - Windows PowerShell需使用:
     ```bash
     Remove-Item -Recurse -Force [路径]
     ```
     而非Linux的 `rm -rf [路径]`

2. **处理长路径**
   - 长路径应使用引号:
     ```bash
     Move-Item -Path "较长的路径名称/文件名" -Destination "目标位置"
     ```

3. **避免管道命令过长**
   - 拆分复杂命令，分步执行:
     ```bash
     # 不推荐
     Get-ChildItem -Path . -Filter "pattern" | Where-Object {条件} | ForEach-Object { 操作 }

     # 推荐
     $items = Get-ChildItem -Path . -Filter "pattern" | Where-Object {条件}
     foreach ($item in $items) { 操作 }
     ```

## 文件编辑操作

1. **先备份再修改**
   ```bash
   Copy-Item -Path [原始文件] -Destination [原始文件].bak
   ```

2. **确保编码一致**
   - 使用UTF-8编码保存文本文件，避免中文乱码:
     ```bash
     # PowerShell示例
     Get-Content -Path [文件] -Encoding UTF8 | Set-Content -Path [输出文件] -Encoding UTF8
     ```

## 环境管理

1. **使用虚拟环境隔离依赖**
   ```bash
   # 创建环境
   conda env create -f environment.yml

   # 激活环境
   conda activate yeast_cell
   ```

2. **在操作前确认当前路径**
   ```bash
   # 检查当前目录
   pwd

   # 确保在项目根目录
   cd /path/to/project
   ```

## 错误恢复步骤

1. **从备份恢复文件**
   ```bash
   # 从最近的备份恢复
   Get-ChildItem -Path "data_backup_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object {
       Copy-Item -Path "$($_.FullName)/*" -Destination "./data/" -Recurse -Force
   }
   ```

2. **恢复误删的脚本**
   - 从归档目录恢复:
     ```bash
     Copy-Item -Path "archive_*/scripts_backup/[脚本名]" -Destination "scripts/" -Force
     ```

## 版本控制注意事项

1. **在进行重大更改前提交当前更改**
   ```bash
   git add .
   git commit -m "保存当前状态"
   ```

2. **使用分支进行实验性更改**
   ```bash
   git checkout -b experimental
   # 进行更改...
   # 如果成功:
   git checkout main
   git merge experimental
   # 如果失败:
   git checkout main
   ```

## 常见问题解决

1. **PowerShell脚本执行策略问题**
   - 如果无法执行脚本，尝试:
     ```bash
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```

2. **路径过长导致的操作失败**
   - 使用绝对路径或简化目录结构
   - 避免深层嵌套目录

请根据本文档指南执行操作，避免因指令不明确或错误操作导致的问题。如有任何不确定的操作，请先执行测试或演示模式确认安全性。