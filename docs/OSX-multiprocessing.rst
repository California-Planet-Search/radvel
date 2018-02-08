.. _OSX-multiprocessing:

multiprocessing on OSX
==========================

The Problem
+++++++++++
``numpy`` is linked to one of several C-based linear algebra libraries that
perform computations. On OSX, the default library for linear algebra computations
is called ``Accelerate``. ``Accelerate`` doesn't support computation on either
side of a fork, which means that if, for example, you try to calculate a maximum
a posteriori fit to a set of data using a Gaussian Process likelihood and then 
perform a parallelized Markov Chain Monte Carlo error analysis using ``multiprocessing``, 
you will get an unexplained segfault. They could at least throw an error, in my opinion, 
but oh well. 

This is a well-documented issue (you'll find several conversations about it by Googling 
"multiprocessing accelerate segfault") that Apple is opposed to fixing. 

To work around this issue, we suggest:

1. Running your parallel Gaussian Process MCMC computations on a non-OSX machine, 
if you have access to one. This is only a problem on OSX, so a Windows, 
Linux, etc. machine will do fine.

2. Running your Gaussian Process MCMC computations in serial.

3. Re-building your version of ``numpy`` using a 
different C-based linear algebra library. Instructions
are below. It's not as hard as it sounds!

Rebuilding ``numpy``
++++++++++++++++++++
``conda-forge`` provides a version of ``numpy`` built on ``openBLAS``, a
C-based linear algebra library that works just as well as ``Accelerate``,
and is in fact the default on several other OS types. 

First of all, we suggest doing all of this in a conda virtual environment, 
and subsequently running all of your ``radvel`` computations within that environment. 
Install ``anaconda`` or ``miniconda``, then see instructions `here
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
to learn more about virtual environments.

Step 0: optionally create and activate a ``conda`` virtual environment.

.. code-block:: bash
	
	$ conda create --name radvel_virtualenv --python=3.6 # create a virtual environment for best results
	$ source activate radvel_virtualenv # activate the newly minted virtual environment

Step 1: uninstall & reinstall ``numpy``:

.. code-block:: bash

	$ conda uninstall numpy # clear numpy installation from current environment
	$ conda install --channel "conda-forge" numpy # reinstall conda-forge version of numpy

Step 2: reinstall packages that rely on ``numpy``

.. code-block:: bash

	$ conda install scipy matplotlib cython astropy pandas
	$ pip install corner celerite emcee
	$ pip install radvel --upgrade  #reinstall radvel


That should be it! As always, please submit an issue on the ``radvel`` GitHub page if you run
into any problems.