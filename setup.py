from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="radvel-package",
    version="0.1",
    author="Erik Petigura, BJ Fulton",
    packages =['radvel'],
    ext_modules=cythonize("src/_kepler.pyx"),
    include_dirs=[numpy.get_include()]
)
