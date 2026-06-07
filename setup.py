from setuptools import setup, Extension

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'poker_cpp',
        ['cpp/bindings.cpp', 'cpp/poker_env.cpp'],
        include_dirs=[get_pybind_include(), 'cpp'],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name='poker_cpp',
    version='1.0',
    ext_modules=ext_modules,
)
