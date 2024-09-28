from setuptools import setup, Extension
import numpy

module = Extension(
    'euler_solver',
    sources=['src/euler_solver.c'],
    include_dirs=['src', numpy.get_include()]
)

setup(
    name='euler_solver',
    version='1.0',
    ext_modules=[module],
    description='A Python extension module for solving ODE using Euler method'
)

