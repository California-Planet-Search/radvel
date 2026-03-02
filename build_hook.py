"""
Custom build hook for hatchling to handle Cython extensions.
This maintains backward compatibility while using modern build tools.
"""
import os
import subprocess
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SETUP_CODE = """
from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension("radvel._kepler", ["src/_kepler.pyx"], include_dirs=[np.get_include()])
]

setup(ext_modules=extensions, packages="radvel")
"""


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to handle Cython extensions and data files."""

    def initialize(self, version, build_data):
        """Initialize the build process."""

        # Import Cython and setuptools here to avoid import issues during build
        try:
            setup_path = "build_hook_setup.py"
            with open(setup_path, "w") as setup_file:
                setup_file.write(SETUP_CODE)

            # This part is a simplified version of hatch-cython
            # https://github.com/joshua-auchincloss/hatch-cython/blob/main/src/hatch_cython/plugin.py
            process = subprocess.run(
                [sys.executable, setup_path, "build_ext", "--inplace"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            stdout = process.stdout.decode("utf-8")
            if process.returncode:
                self.app.display_error(
                    f"cythonize exited non null status {process.returncode}"
                )
                self.app.display_error(stdout)
                msg = "failed compilation"
                raise Exception(msg)
            else:
                self.app.display_info(stdout)
                self.app.display_success("Successfully compiled Cython extensions")
        except Exception as e:
            self.app.display_error(f"Error during Cython compilation: {e}")
        finally:
            # Clean up the temporary setup file
            if os.path.exists(setup_path):
                os.remove(setup_path)
