from setuptools import setup, find_packages, Extension
import numpy
import Cython.Build as cb
import re


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

extensions = [Extension("radvel._kepler", ["src/_kepler.pyx"],)]

setup(
    name="radvel",
    version=get_property('__version__', 'radvel'),
    author="BJ Fulton, Erik Petigura, Sarah Blunt",
    packages=find_packages(),
    ext_modules=cb.cythonize(extensions),
    include_dirs=[numpy.get_include()],
    data_files=[
        (
            'radvel_example_data', 
            [
                'example_data/164922_fixed.txt', 
                'example_data/epic203771098.csv'
            ]
        )
    ],
    entry_points={'console_scripts': ['radvel=radvel.cli:main']},
    install_requires=[line.strip() for line in
                      open('requirements.txt', 'r').readlines()]

)
