
# RadVel

General Toolkit for Modeling Radial Velocities.

[![Build Status](https://travis-ci.org/California-Planet-Search/radvel.svg?branch=master)](https://travis-ci.org/California-Planet-Search/radvel)
[![Coverage Status](https://coveralls.io/repos/github/California-Planet-Search/radvel/badge.svg?branch=master)](https://coveralls.io/github/California-Planet-Search/radvel?branch=master)
[![Documentation Status](https://readthedocs.org/projects/radvel/badge/?version=latest)](http://radvel.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/radvel.svg)](https://badge.fury.io/py/radvel)

## Attribution

Written by BJ Fulton, Erik Petigura, and Sarah Blunt. Fulton et al. (in prep.)

Please cite the following DOI if you wish to make use of this software in any publication.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.580821.svg)](https://doi.org/10.5281/zenodo.580821)

## Documentation

Documentation is available [here](http://radvel.readthedocs.io/)

## Features


With RadVel you can


- *Optimize*
  - leverages the suite of minimizers in [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- *Run MCMC*
  - leverages [emcee](http://dfm.io/emcee/) package for MCMC exploration of posterior
- *Visualize*
  - creates quicklook summary plots and statistics
 
RadVel is

- *Flexible*
  - fix/float parameters that are indexed as strings (emulates [lmfit](https://github.com/lmfit/lmfit-py/) API)
  - convert between different parameterizations e.g. `e omega <-> sqrtecosw sqrtesinw`
  - incorporate RVs from multiple telescopes
- *Extensible* 
  - Object-oriented programing makes adding new likelihoods, priors, etc. easy
- *Scriptable*
  - Code can be run through convenient Command-line Interface (CLI) 
- *Fast*
   - Kepler's equation solved in C (slower pure python solver available)
   - MCMC is multi-threaded

## Future Improvements

- Gaussian Process (GP) functionality

## Tutorials 

Follow examples in

- `radvel/tests/SyntheticData.ipynb`
- `radvel/tests/K2-24_Fitting+MCMC.ipynb`
- `radvel/tests/164922_Fitting+MCMC.ipynb`

