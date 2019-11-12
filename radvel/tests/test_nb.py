import warnings
import os
import sys
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

warnings.simplefilter('ignore')


# def test_notebooks(nbdir='docs/tutorials/'):
#     """
#     Run though notebook tutorials
#     """
#
#     nbfiles = sorted(glob(os.path.join(nbdir, '*.ipynb')))
#     for nbfile in nbfiles:
#         print(nbfile)
#         with open(nbfile) as f:
#             nb = nbformat.read(f, as_version=4)
#
#         if sys.version_info[0] < 3:
#             ep = ExecutePreprocessor(timeout=900, kernel_name='python2')
#         else:
#             ep = ExecutePreprocessor(timeout=900, kernel_name='python3')
#
#         ep.preprocess(nb, {'metadata': {'path': nbdir}})
#
#
# if __name__ == '__main__':
#     test_notebooks()
