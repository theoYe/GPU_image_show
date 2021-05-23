import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

if sys.platform == 'win32':
    vc_version = os.getenv('VCToolsVersion', '')
    if vc_version.startswith('14.16.'):
        CXX_FLAGS = ['/sdl']
    else:
        CXX_FLAGS = ['/sdl', '/permissive-']
else:
    CXX_FLAGS = ['-g']

USE_NINJA = os.getenv('USE_NINJA') == '1'
ext_modules = []
# ext_modules = [
#     CppExtension(
#         'torch_test_cpp_extension.cpp', ['extension.cpp'],
#         extra_compile_args=CXX_FLAGS),
#     CppExtension(
#         'torch_test_cpp_extension.msnpu', ['msnpu_extension.cpp'],
#         extra_compile_args=CXX_FLAGS),
#     CppExtension(
#         'torch_test_cpp_extension.rng', ['rng_extension.cpp'],
#         extra_compile_args=CXX_FLAGS),
#     CppExtension('lltm_cpp', ['lltm.cpp'],
#         extra_compile_args=CXX_FLAGS),
#
# ]
#
# if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
#     extension = CUDAExtension(
#         'torch_test_cpp_extension.cuda', [
#             'cuda_extension.cpp',
#             'cuda_extension_kernel.cu',
#             'cuda_extension_kernel2.cu',
#         ],
#         extra_compile_args={'cxx': CXX_FLAGS,
#                             'nvcc': ['-O2']})
#     ext_modules.append(extension)
#
# if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
#     extension = CUDAExtension(
#         'torch_test_cpp_extension.torch_library', [
#             'torch_library.cu'
#         ],
#         extra_compile_args={'cxx': CXX_FLAGS,
#                             'nvcc': ['-O2']})
#     ext_modules.append(extension)

if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    extension = CUDAExtension(
        'lltm_cuda', [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
            'glad.cu'
        ],
        extra_compile_args={'cxx': CXX_FLAGS,
                            # windows 可以省略 -lglfw等
                            'nvcc': ['-O2', '-I', r'"C:\dev\vcpkg\installed\x64-windows\include"','-lglfw3dll','-L', r'"C:\dev\vcpkg\installed\x64-windows\lib"']},
        # extra_ldflags={r'/LIBPATH: "C:\dev\vcpkg\packages\glfw3_x64-windows\lib\glfw3dll.lib"'}
        library_dirs=["C:\dev\vcpkg\packages\glfw3_x64-windows\lib"],
        libraries=["glfw3dll"]

    )
    ext_modules.append(extension)

setup(
    name='torch_test_cpp_extension',
    packages=['torch_test_cpp_extension'],
    ext_modules=ext_modules,
    include_dirs='self_compiler_include_dirs_test',
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)},
)
