from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
include_gsl_dir = "/usr/include/gsl/"
lib_gsl_dir = "/usr/lib64/"

REQUIRED = [
    'numpy',
    'matplotlib',
    'cython',
    'gsl',
    'pyrat @ git+https://github.com/alecthomson/pyrat',
]
ext = Extension("grid_tools", ["fsclean/grid_tools.pyx"], include_dirs=
    [numpy.get_include(), include_gsl_dir], library_dirs=[lib_gsl_dir],
    libraries=["gsl", "gslcblas"])

setup(
    name='fsclean',
    ext_modules = cythonize([ext]),
    cmdclass={'build_ext': build_ext},
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    entry_points={
        'console_scripts': ['fsclean=fsclean.fsclean:main'],
    },
    include_dirs=[numpy.get_include(), include_gsl_dir], 
    library_dirs=[lib_gsl_dir],
    install_requires=REQUIRED
)
