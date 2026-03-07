from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension(
        "radvel._kepler",
        ["src/_kepler.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=extensions,
    package_data={"radvel": ["_kepler*.so", "_kepler*.pyd"]},
    data_files=[("radvel_example_data", [])],
)
