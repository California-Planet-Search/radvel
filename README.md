# radvel
General Toolkit for Modeling Radial Velocities

Note: Erik is still working on skeleton functionaliy. Afterwhich, BJ will take the lead on development.

Issues:

MCMC is buggy when we use it's used in a multi-threaded way. Is this becase we're creating multiple instances of the posterior object and it's not being passed around properly

## Features

- Object-oriented (i.e. models, likelihoods, priors, and posteriors are defined as objects)
- Extensible (i.e. naturally define new likelihoods, priors, parameterizatoins)
- Convenient API to fix/float parameters
- Easily plugs in to the the suite of `scipy.optimize` routines for max-likelihood fitting 
- Works with `emcee` MCMC
- parameters are represented as dicts not arrays
- Can handle data from multiple telescopes
- Easily convert between different parameterizations

## To Do

- Flesh out documentation
- Bundle up examples in convenient test cases
- Rewrite `rv_drive` in C (for fun)
- Release publically (future) 

## Tutorials 

Follow examples in

- `radvel/tests/SyntheticData.ipynb`
- `radvel/tests/EPIC-2037_Fitting+MCMC.ipynb`
- `radvel/tests/164922_Fitting+MCMC.ipynb`

You'll need the following dependencies

- emcee
- corner
- pandas (to read in hdf5)




