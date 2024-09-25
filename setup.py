from setuptools import setup, Extension
import numpy

# Define the C extension module
APV = Extension(
    'APV',  # The name of the module to import in Python
    sources=['src/apv-module.c'],  # Path to your C source file
    include_dirs=[  # Include directories for header files
        numpy.get_include(),  # Include NumPy headers
        # Add other include directories here if necessary
    ],
)

# Setup function
setup(
    name='APV',  # Name of your package
    version='0.1',
    description='A Python extension module for APV',
    ext_modules=[APV],  # List of extension modules to build
)

