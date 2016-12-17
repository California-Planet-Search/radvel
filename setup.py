from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

exec(open('radvel/__version__.py').read())

extensions = [
    Extension("radvel._kepler", ["src/_kepler.pyx"],)
]

setup(
    name = "radvel-package",
    version = __version__,
    author = "BJ Fulton, Erik Petigura",
    packages = find_packages(),
    ext_modules = cythonize(extensions),
    include_dirs = [numpy.get_include()],
    data_files = [
        (
            'radvel_example_data', 
            [
                'example_data/164922_fixed.txt', 
                'example_data/epic203771098.csv'
            ]
        )
    ],
    entry_points = {'console_scripts': ['radvel=radvel.cli:main']}
)

print find_packages(),
