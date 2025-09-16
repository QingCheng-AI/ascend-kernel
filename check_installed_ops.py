# python调用侧算子安装验证，无法验证底层npu算子是否安装成功
import os
import torch_npu
import cinfer_ascendc

try:
    with open(".installed_ops.txt", "r") as f:
        installed_ops = [line.strip() for line in f]
except:
    raise RuntimeError(
        "Failed to open .installed_ops.txt. Consider installing the package first."
    )

for op in installed_ops:
    if hasattr(cinfer_ascendc, op) == False:
        print("Check installed ops failed.")
        exit(1)
    else:
        print(f"{op} installed.")
print("Check installed ops passed.")
