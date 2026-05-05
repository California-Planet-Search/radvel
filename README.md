
# RadVel

General Toolkit for Modeling Radial Velocities.

[![CI](https://github.com/California-Planet-Search/radvel/workflows/CI/badge.svg)](https://github.com/California-Planet-Search/radvel/actions)
[![Coverage Status](https://coveralls.io/repos/github/California-Planet-Search/radvel/badge.svg?branch=master)](https://coveralls.io/github/California-Planet-Search/radvel?branch=master)
[![Documentation Status](https://readthedocs.org/projects/radvel/badge/?version=latest)](http://radvel.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/radvel.svg)](https://badge.fury.io/py/radvel)
[![PyPI downloads](https://img.shields.io/pypi/dm/radvel.svg)](https://pypistats.org/packages/radvel)
[![ASCL:1801.012](https://img.shields.io/badge/ascl-1801.012-blue.svg?colorB=262255)](http://ascl.net/1801.012)

[![Powered by emcee](https://img.shields.io/badge/powered_by-emcee-EB5368.svg?style=flat)](https://emcee.readthedocs.io)
[![Powered by AstroPy](https://img.shields.io/badge/powered_by-AstroPy-EB5368.svg?style=flat)](http://www.astropy.org)
[![Powered by celerite](https://img.shields.io/badge/powered_by-celerite-EB5368.svg?style=flat)](https://celerite.readthedocs.io)

## Attribution

Written by BJ Fulton, Erik Petigura, Sarah Blunt, and Evan Sinukoff. [Fulton et al. (2018)](http://adsabs.harvard.edu/abs/2018PASP..130d4504F)

Please cite the [original publication](http://adsabs.harvard.edu/abs/2018PASP..130d4504F) and the following DOI if you make use of this software in your research.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.580821.svg)](https://doi.org/10.5281/zenodo.580821)

## Installation

### Quick Install

```bash
pip install radvel
```

### Development Install

```bash
git clone https://github.com/California-Planet-Search/radvel.git
cd radvel

# Install problematic dependencies via conda (recommended for macOS)
conda install pytables h5py

# Install in development mode
pip install -e .
```

### System Requirements

- **Python**: 3.8, 3.9, 3.11, 3.12
- **LaTeX**: Required for report generation (install TexLive or similar)
- **macOS users**: Consider using conda for `pytables` and `h5py` to avoid compilation issues

## Documentation

Documentation is available [here](http://radvel.readthedocs.io/)

## Run as a service (v1.6+)

RadVel 1.6 ships an HTTP service that wraps the entire CLI workflow
behind a JSON API plus an optional browser UI. Useful for non-Python
clients, long-running MCMC jobs, and reproducible deployments.

```bash
docker run --rm -p 8000:8000 -v "$PWD/.runs:/data" \
    ghcr.io/california-planet-search/radvel-api:1.6
open http://localhost:8000/ui
```

See the [service guide](http://radvel.readthedocs.io/en/latest/api_service.html)
and the [UI walkthrough](http://radvel.readthedocs.io/en/latest/ui_guide.html)
for details. The same image is what runs in production.

## Features

With RadVel you can


- *Optimize*
  - leverages the suite of minimizers in [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- *Run MCMC*
  - leverages the [emcee](http://dfm.io/emcee/) package for MCMC exploration of the posterior probability distribution
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
  - Code can be run through a convenient Command-line Interface (CLI) 
- *Fast*
   - Kepler's equation solved in C (slower Python solver also included)
   - MCMC is multi-threaded

## Tutorials 

Follow examples in

- `radvel/docs/tutorials/SyntheticData.ipynb`
- `radvel/docs/tutorials/K2-24_Fitting+MCMC.ipynb`
- `radvel/docs/tutorials/164922_Fitting+MCMC.ipynb`
- `radvel/docs/tutorials/GaussianProcess-tutorial.ipynb`