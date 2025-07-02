import torch
import sys


def gen_tensor(low, high, shape, path, name, dtype=torch.float16):
    # 生成随机张量
    tensor = (high - low) * torch.rand(*shape) + low
    tensor = tensor.to(dtype)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)

    tensor.numpy().tofile(f"{path}/{name}.bin")


def gen_group_list(low, high, len, path, name, dtype=torch.int64):
    tensor = torch.linspace(low, high, len, dtype=dtype)
    tensor.numpy().tofile(f"{path}/{name}.bin")


if __name__ == "__main__":
    # print(len(sys.argv))
    #     print("Please enter the path to the data")
    try:
        data_path = sys.argv[1]
    except:
        raise RuntimeError("Input args error: please enter the path to the data")
    gen_tensor(-128, 128, (48, 7168), data_path, "x", torch.bfloat16)
    gen_tensor(-128, 128, (256, 7168, 256), data_path, "weight", torch.int8)
    gen_tensor(-2, 2, (256, 448, 512), data_path, "scale_offset", torch.bfloat16)
    gen_tensor(-0.05, 0.05, (256, 448, 512), data_path, "scale", torch.bfloat16)
    gen_group_list(0, 48, 256, data_path, "group_list", torch.int64)
