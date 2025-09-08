# setup.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11
import os
import torch_npu
import subprocess
import glob
import site
import tempfile

# 获取 pybind11 的头文件路径
pybind11_include = pybind11.get_include()

op_name = "grouped_gemm"

# 检查必要的环境变量
if "BASE_LIBS_PATH" not in os.environ:
    if "ASCEND_HOME_PATH" not in os.environ:
        if "ASCEND_AICPU_PATH" in os.environ:
            os.environ["ASCEND_HOME_PATH"] = os.environ["ASCEND_AICPU_PATH"]
        else:
            raise EnvironmentError(
                "请设置环境变量 ASCEND_HOME_PATH, ASCEND_AICPU_PATH 或 BASE_LIBS_PATH"
            )
else:
    os.environ["ASCEND_HOME_PATH"] = os.environ["BASE_LIBS_PATH"]
site_packages_path = site.getsitepackages()[0]


# 创建自定义的 BuildExtension 类
class CustomBuildExtension(BuildExtension):
    def build_extension(self, ext):
        # 清理之前可能残留的setup安装文件
        subprocess.run(["bash", "clean_build.sh"], check=True)
        # 安装ascendc算子
        if ext.name == "cinfer_ascendc":
            try:
                # 将ascend算子的相关代码和编译文件放在/tmp下
                tmp_dir = tempfile.mkdtemp(prefix="cinfer_ascendc_", dir="/tmp")
                subprocess.run(["cp", "-r", f"{op_name}.json", tmp_dir], check=True)
                subprocess.run(
                    [
                        "msopgen",
                        "gen",
                        "-i",
                        f"{tmp_dir}/{op_name}.json",
                        "-c",
                        "ai_core-Ascend910B,ai_core-Ascend910_93",
                        "-lan",
                        "cpp",
                        "-out",
                        f"{tmp_dir}/cinfer_ascendc_autogen",
                    ],
                    check=True,
                )
                subprocess.run(
                    ["cp", "-r", "op_host", f"{tmp_dir}/cinfer_ascendc_autogen"],
                    check=True,
                )
                subprocess.run(
                    ["cp", "-r", "op_kernel", f"{tmp_dir}/cinfer_ascendc_autogen"],
                    check=True,
                )
                cur_dir = os.getcwd()
                os.chdir(f"{tmp_dir}/cinfer_ascendc_autogen")
                # 执行 build.sh
                if os.path.exists("build.sh"):
                    subprocess.run(["bash", "build.sh"], check=True)
                # 删除可能已经安装的算子包
                subprocess.run(
                    ["rm", "-rf", f"{site_packages_path}/vendors"], check=True
                )
                # 执行所有 .run 文件
                run_files = glob.glob("./build_out/*.run")
                for run_file in run_files:
                    # subprocess.run(['bash', run_file], check=True)
                    subprocess.run(
                        ["bash", run_file, f"--install-path={site_packages_path}"],
                        check=True,
                    )
                os.chdir(cur_dir)
            except subprocess.CalledProcessError as e:
                print(f"执行脚本时出错: {e}")
                raise

        # 调用原始的构建方法
        super().build_extension(ext)


# 定义扩展模块
os.environ["CXX"] = "/usr/bin/c++"
torch_npu_path = os.path.dirname(torch_npu.__file__)
cinfer_ascendc = CppExtension(
    "cinfer_ascendc",  # 模块名称（需与 PYBIND11_MODULE 中一致）
    sources=[
        "./src/pybind.cpp",
        f"./src/{op_name}.cpp",
    ],  # 包含绑定代码的 C++ 文件
    include_dirs=[
        pybind11_include,
        os.path.expandvars("${ASCEND_HOME_PATH}/include"),
        os.path.expandvars("${ASCEND_HOME_PATH}/include/aclnn"),
        # 'os.path.expandvars(${ASCEND_HOME_PATH}/aarch64-linux/lib64)',
        os.path.expandvars(f"{site_packages_path}/vendors/customize/op_api/include"),
        # os.path.expandvars('${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include'),
        f"{torch_npu_path}/include",
    ],  # 包含 pybind11 头文件
    language="c++",
    extra_compile_args=[
        "-fPIC",
        "-O3",
        "-g",
        "-Wall",
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    extra_link_args=[
        f"-Wl,-rpath={os.path.expandvars(f'{site_packages_path}/vendors/customize/op_api/lib')}",
        # f"-Wl,-rpath={os.path.expandvars('${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib')}",
        # "-Wl,--no-as-needed",
    ],
    library_dirs=[
        os.path.expandvars("${ASCEND_HOME_PATH}/lib64"),
        # 'os.path.expandvars(${ASCEND_HOME_PATH}/aarch64-linux/lib64)',
        os.path.expandvars(f"{site_packages_path}/vendors/customize/op_api/lib"),
        # os.path.expandvars('${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib'),
        f"{torch_npu_path}/lib",
    ],
    libraries=[
        "cust_opapi",
        "torch_npu",
    ],
)

setup(
    name="cinfer_ascendc",
    ext_modules=[cinfer_ascendc],
    cmdclass={"build_ext": CustomBuildExtension},  # 使用自定义的 BuildExtension
    description="A Python interface for groupmatmul",
)
