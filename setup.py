from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import re


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)


extensions = [Extension("radvel._kepler", ["src/_kepler.pyx"],)]

reqs = []
for line in open('requirements.txt', 'r').readlines():
    if not line.startswith('celerite'):
        reqs.append(line)

setup(
    name="radvel",
    version=get_property('__version__', 'radvel'),
    author="BJ Fulton, Erik Petigura, Sarah Blunt, Evan Sinukoff",
    packages=find_packages(),
    setup_requires=['numpy'],
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    data_files=[
        (
            'radvel_example_data', 
            [
                'example_data/164922_fixed.txt', 
                'example_data/epic203771098.csv',
                'example_data/k2-131.txt'
            ]
        )
    ],
    entry_points={'console_scripts': ['radvel=radvel.cli:main']},
    install_requires=reqs,
    include_package_data=True
)
