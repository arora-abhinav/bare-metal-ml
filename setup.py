from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bare_metal_ml._cpp",
        sources=["bare_metal_ml/cpp/bindings.cpp"],
        include_dirs=["bare_metal_ml/cpp"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="bare-metal-ml",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
