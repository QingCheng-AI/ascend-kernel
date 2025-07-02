import numpy as np
import sys
import os
import torch

if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except:
        raise RuntimeError("Input args error: please enter the path to the data")
    file = f"{data_path}/y.bin"
    if not os.path.isfile(file):
        raise RuntimeError(f"Invalid case name:", file)
    y = np.fromfile(file, dtype=np.float16)
    y = torch.from_numpy(y).view(dtype=torch.bfloat16)
    print(f"output: ", y)
