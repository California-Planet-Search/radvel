# radvel
General Toolkit for Modeling Radial Velocities

Note: Erik is still working on skeleton functionaliy. Afterwhich, BJ will take the lead on development

## Features

- Object-oriented (i.e. models, likelihoods, priors, and posteriors are defined as objects)
- Extensible (i.e. naturally define new likelihoods, priors, parameterizatoins)
- Convenient API to fix/float parameters
- Easily plugs in to the the suite of `scipy.optimize` routines for max-likelihood fitting 
- Works with `emcee` MCMC
- parameters are represented as dicts not arrays

## To Do

- Bundle up examples in convenient test cases
- Make likelihoods additive
- Test on data from multiple telescopes
- Rewrite `rv_drive` in C (for fun)
- Release publically (future) 

## Tutorials 

Follow examples in `radvel/tests/SyntheticData.ipynb`


