.. _tutorials:

Tutorials
=========

First you need to create a planetary system configuration file. 
Use the examples within the `example_planets` directory as templates.

Use the ``radvel`` command line interface to execute a fit. The
``radvel`` binary should have been automatically placed in your system's path by the
``python setup.py install`` command (see :ref:`installation`).

See ``radvel --help`` for instructions. Here is an example workflow to
run a simple fit using the included `HD164922.py` example
configuration file:

Perform a maximum-likelihood fit. You should almost always do this first:

.. code-block:: bash
    radvel fit -s /path/to/HD164922.py


   
3. The results will be placed in a directory with the same name as
   your planet configuration file (without `.py`, e.g. `HD164922`).


For a more detailed tutorial in the use of the underlying API refer to
the example iPython notebooks in the `tests` directory.
