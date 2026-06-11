from setuptools import setup, find_packages

try:
    import sys
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    if sys.platform == "darwin":
        blas_link = ["-framework", "Accelerate"]
    else:
        blas_link = ["-lopenblas"]

    ext_modules = [
        Pybind11Extension(
            "bare_metal_ml._cpp",
            sources=["bare_metal_ml/cpp/bindings.cpp"],
            include_dirs=["bare_metal_ml/cpp"],
            extra_compile_args=["-O3", "-std=c++17"],
            extra_link_args=blas_link,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    ext_modules = []
    cmdclass = {}

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
