# setup_log.md — 数据组环境搭建记录

## 环境信息
- 操作系统: Microsoft Windows 11 家庭版 中文版 10.0.26200
- Conda: conda 22.9.0 (Miniconda/Anaconda 安装于 C:\ProgramData\Anaconda3)
- 环境名称: cole-data
- Python: Python 3.10.20

## 关键包版本
- numpy 2.2.5
- pandas 2.3.3
- pyarrow 24.0.0
- jsonschema 4.25.1

## 激活方式
```powershell
$env:CONDA_EXE = 'C:\ProgramData\Anaconda3\Scripts\conda.exe'
Import-Module 'C:\ProgramData\Anaconda3\shell\condabin\Conda.psm1'
conda activate cole-data
```

或直接使用解释器路径:
`C:\ProgramData\Anaconda3\envs\cole-data\python.exe`

## 备注
- 环境创建时间: 2026-08-11 01:23
- 该环境仅用于数据组（阶段 D1-D4），不含环境组的 TensorFlow/COLE 旧栈依赖。
- 完整依赖锁定见同目录 environment_lock.yml。