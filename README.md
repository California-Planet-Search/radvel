# RadVel

General Toolkit for Modeling Radial Velocities.

[![Build Status](https://travis-ci.org/California-Planet-Search/radvel.svg?branch=master)](https://travis-ci.org/California-Planet-Search/radvel)
[![Coverage Status](https://coveralls.io/repos/github/California-Planet-Search/radvel/badge.svg?branch=master)](https://coveralls.io/github/California-Planet-Search/radvel?branch=master)
[![Documentation Status](https://readthedocs.org/projects/radvel/badge/?version=latest)](http://radvel.readthedocs.io/en/latest/?badge=latest)

## Attribution

Written by BJ Fulton, Erik Petigura, and Sarah Blunt. Fulton et al. (in prep.)

Please cite the following DOI if you wish to make use of this software in any publication.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.580821.svg)](https://doi.org/10.5281/zenodo.580821)

## Documentation

Documentation is available on ReadTheDocs.org: http://radvel.readthedocs.io

## Features

- Object-oriented (i.e. models, likelihoods, priors, and posteriors are defined as objects)
- Extensible (i.e. naturally define new likelihoods, priors, parameterizatoins)
- Convenient API to fix/float parameters
- Easily plugs in to the the suite of `scipy.optimize` routines for max-likelihood fitting 
- Works with `emcee` MCMC
- parameters are represented as dicts not arrays
- Can handle data from multiple telescopes
- Easily convert between different parameterizations
- Computation of Kepler's equation (numerically intensive) written in C

## Future Improvements...

- PERF: Optimizations for low eccentricity orbits
- Add Gaussian Process (GP) functionality

## Tutorials 

Follow examples in

- `radvel/tests/SyntheticData.ipynb`
- `radvel/tests/EPIC-2037_Fitting+MCMC.ipynb`
- `radvel/tests/164922_Fitting+MCMC.ipynb`
