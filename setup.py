from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("radvel._kepler", ["src/_kepler.pyx"],)
]

setup(
    name = "radvel-package",
    version = "0.5b",
    author = "BJ Fulton, Erik Petigura",
    packages = find_packages(),
    ext_modules = cythonize(extensions),
    include_dirs = [numpy.get_include()],
    data_files = [
        (
            'radvel_example_data', 
            [
                'example_data/164922_fixed.txt', 
                'example_data/164922.hdf', 
                'example_data/epic203771098.hdf'
            ]
        )
    ],
    entry_points = {'console_scripts': ['radvel=radvel.cli:main']}
)

print find_packages(),
