# RadVel

General Toolkit for Modeling Radial Velocities. Version 0.9.4

## Attribution

Written by BJ Fulton and Erik Petigura, Fulton and Petigura (in prep.).

Please cite the following DOI if you wish to make use of this software in any publication.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.580821.svg)](https://doi.org/10.5281/zenodo.580821)

## Documentation

Documentation is available on ReadTheDocs.org: http://radvel.readthedocs.io

[![Documentation Status](https://readthedocs.org/projects/radvel/badge/?version=master)](http://radvel.readthedocs.io/en/master/?badge=master)

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

- Bundle up examples in convenient test cases
- PERF: Optimizations for low eccentricity orbits
- Streamline API

## Tutorials 

Follow examples in

- `radvel/tests/SyntheticData.ipynb`
- `radvel/tests/EPIC-2037_Fitting+MCMC.ipynb`
- `radvel/tests/164922_Fitting+MCMC.ipynb`

You'll need the following dependencies

- emcee
- corner 
- pandas (to read in hdf5)
- matplotlib-1.5.0
- cython (tested with 0.22)
- pdflatex installed and in your system's path

