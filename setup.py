from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("radvel._kepler", ["src/_kepler.pyx"],)
]

setup(
    name="radvel-package",
    version="0.1",
    author="Erik Petigura, BJ Fulton",
    packages =['radvel'],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
