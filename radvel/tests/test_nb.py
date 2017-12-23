import warnings

import os
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

warnings.filterwarnings("ignore")
warnings.simplefilter('once', DeprecationWarning)


def test_notebooks(nbdir='tutorials/'):
    """
    Run though notebook tutorials
    """

    nbfiles = sorted(glob(os.path.join(nbdir, '*.ipynb')))
    for nbfile in nbfiles:
        print(nbfile)
        with open(nbfile) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': nbdir}})


if __name__ == '__main__':
    test_notebooks()
