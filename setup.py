import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension

name = "dump_collectives"
sources = ["./ProcessGroupDump.cpp"]
include_dirs = ["./", "./gloo"] # 缺少gloo头文件，需要单独下载一份

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name=name,
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args = ["-DUSE_C10D_GLOO"], # 启用gloo
    )
else:
    module = cpp_extension.CppExtension(
        name=name,
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args = ["-DUSE_C10D_GLOO"],
    )

setup(
    name = "Dump-Collectives",
    version = "0.0.1",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)