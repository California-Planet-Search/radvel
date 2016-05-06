from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name="radvel-package",
    version="0.1",
    author="Erik Petigura, BJ Fulton",
    packages =['radvel'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "kepler_cext", 
            sources=["src/_kepler.pyx"],
            include_dirs=[numpy.get_include()]
            )
        ]
)

