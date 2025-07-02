# Grouped Gemm自动安装脚本

### 安装命令
```bash
pip install .
```
### 说明
该脚本会调用msopgen自动生成工程化算子，生成的代码位于/tmp/grouped_gemm_*目录下。
为了保证python虚拟环境的隔离性，工程化算子的.run包会被安装在当前python环境的site-package目录下。