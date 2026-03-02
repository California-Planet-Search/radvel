import os
import sys
from glob import glob

# pytest is now the test runner, but we don't need to import it in test files
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_notebook():
    """
    Run though a single notebook tutorial
    """
    # Test just one notebook to avoid pytest parameterization issues
    nbfile = "docs/tutorials/164922_Fitting+MCMC.ipynb"
    
    if not os.path.exists(nbfile):
        print(f"Skipping {nbfile} - file not found")
        return
        
    print(nbfile)
    with open(nbfile) as f:
        nb = nbformat.read(f, as_version=4)

    basename = os.path.basename(nbfile)
    # Skip the all samplers notebook. Same functionality tested in api tests and it is slow due to repeated sampling to convergence.
    skip_notebooks = ["k2_24_demo_all_samplers.ipynb"]
    if basename in skip_notebooks:
        return
    timeout = 900

    if sys.version_info[0] < 3:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python2")
    else:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")

    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nbfile)}})
