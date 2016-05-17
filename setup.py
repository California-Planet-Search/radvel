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
    include_dirs=[numpy.get_include()],
    data_files=[('radvel_example_data', ['example_data/164922_fixed.txt', 'example_data/164922.hdf', 'example_data/epic203771098.hdf'])]
)
