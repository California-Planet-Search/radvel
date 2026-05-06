from __future__ import annotations

import os
import shutil
from typing import Optional

import h5py
import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, read_hdf


from radvel.posterior import Posterior


def _run_dynesty(
    post: Posterior,
    output_dir: Optional[str] = None,
    sampler_type: str = "static",
    proceed: bool = False,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with `Dynesty <https://dynesty.readthedocs.io/>`_

    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "checkpoint_file" argument. A ``dynesty.save`` file is created automatically.
            When sampling is finished, the final state of the sampler is stored.
        proceed: Continue from previous run in output_dir if available.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments. This is not available for ``sampler='multinest'``.
            Defaults to ``None``.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - ``samples``: Samples array with shape ``(nsamples, nparams)``
            - ``lnZ``: Log of the Bayesian evidence
            - ``lnZ``: Statistical uncertainty on the evidence
            - ``sampler``: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.

    """
    from dynesty import DynamicNestedSampler, NestedSampler

    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}

    if sampler_type == "static":
        sampler_class = NestedSampler
    elif sampler_type == "dynamic":
        sampler_class = DynamicNestedSampler
    else:
        raise ValueError(
            f"Expected 'dynamic' or 'static' as sampler_type. Got {sampler_type}"
        )

    if "resume" in run_kwargs:
        raise ValueError("'resume' not supported for dynesty. Use radlvel's 'proceed' instead'")
    run_kwargs["resume"] = proceed

    if "checkpoint_file" in run_kwargs:
        raise ValueError(
            "checkpoint_file not supported for dynesty. Use radvel's output_dir instead."
        )
    if output_dir is not None:
        checkpoint_file = os.path.join(output_dir, "sampler.save")
        run_kwargs["checkpoint_file"] = checkpoint_file
    checkpoint_file = run_kwargs.get("checkpoint_file", None)

    if proceed and checkpoint_file is not None and os.path.exists(checkpoint_file):
        sampler = sampler_class.restore(checkpoint_file)
    else:
        sampler = sampler_class(
            post.likelihood_ns_array,
            post.prior_transform,
            len(post.name_vary_params()),
            **sampler_kwargs,
        )
        # Dynesty cannot resume when the file does not exist
        if (
            proceed
            and checkpoint_file is not None
            and not os.path.exists(checkpoint_file)
        ):
            run_kwargs["resume"] = False

    if checkpoint_file is not None and not os.path.exists(checkpoint_file):
        outdir = os.path.dirname(checkpoint_file)
        os.makedirs(outdir)

    sampler.run_nested(**run_kwargs)

    if checkpoint_file is not None:
        sampler.save(checkpoint_file)

    results = {
        "samples": sampler.results.samples_equal(),
        "lnZ": sampler.results["logz"][-1],
        "lnZerr": sampler.results["logzerr"][-1],
        "sampler": sampler,
    }

    return results


def _run_ultranest(
    post: Posterior,
    output_dir: Optional[str] = None,
    proceed: bool = False,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with `Ultranest <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`_

    The SliceSampler will be used automatically for more than 7 parameters.
    See `the Ultranest docs <https://johannesbuchner.github.io/UltraNest/example-sine-highd.html#Step-samplers-in-UltraNest>`_ for more information

    Parameters starting with ``tc`` or ``tp`` are assumed to by time of conjunction or time of periastron and are marked as ``wrapped_params`` automatically.

    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "log_dir" argument.
            The ultranest ``log_dir`` is automatically set to ``output_dir``.
        proceed: Continue from previous run in output_dir if available.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments.
            Defaults to ``None``.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - ``samples``: Samples array with shape ``(nsamples, nparams)``
            - ``lnZ``: Log of the Bayesian evidence
            - ``lnZ``: Statistical uncertainty on the evidence
            - ``sampler``: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.
    """
    from ultranest import ReactiveNestedSampler
    from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction

    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}

    if "log_dir" in sampler_kwargs:
        raise ValueError(
            "log_dir not supported for ultranest. Use radvel's output_dir instead."
        )
    if "resume" in sampler_kwargs:
        raise ValueError(
            "'resume' not supported for ultranest. Use radvel's 'proceed' instead."
        )
    sampler_kwargs["resume"] = proceed or 'overwrite'
    sampler_kwargs["log_dir"] = output_dir

    param_names = post.name_vary_params()
    wrapped_params = [pn.startswith(("tc", "tp")) for pn in param_names]
    # I guess simplest is to use overwrite or re-run
    sampler = ReactiveNestedSampler(
        param_names,
        post.likelihood_ns_array,
        post.prior_transform,
        wrapped_params=wrapped_params,
        **sampler_kwargs,
    )

    num_params = len(param_names)
    if num_params > 7:
        nsteps = len(param_names) * 2
        sampler.stepsampler = SliceSampler(
            nsteps=nsteps, generate_direction=generate_mixture_random_direction,
        )

    sampler.run(**run_kwargs)

    results = {
        "samples": sampler.results["samples"],
        "lnZ": sampler.results["logz"],
        "lnZerr": sampler.results["logzerr"],
        "sampler": sampler,
    }

    return results


def _run_multinest(
    post: Posterior,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    proceed: bool = False,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/pymultinest.html#>`_

    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "outputfiles_basename" argument.
            If ``output_dir`` is specified, sets ``outputfiles_basename`` to ``<output_dir>/out``
        proceed: Continue from previous run in output_dir if available.
        overwrite: Overwrite the output files if they exist. Defaults to ``False``.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - ``samples``: Samples array with shape ``(nsamples, nparams)``
            - ``lnZ``: Log of the Bayesian evidence
            - ``lnZ``: Statistical uncertainty on the evidence
    """
    import pymultinest

    run_kwargs = run_kwargs or {}

    # By default, assume we want a temporary output dir
    tmp = True

    if output_dir is None:
        output_dir = "tmpdir"
    else:
        # if an actual outupt dir was specified, it is not temporary
        tmp = False

    if "outputfiles_basename" in run_kwargs:
        raise ValueError(
            "outputfiles_basename not supported for multinest. Use radvel's output_dir instead."
        )
    run_kwargs["outputfiles_basename"] = os.path.join(output_dir, "out")

    if "resume" in run_kwargs:
        raise ValueError(
            "'resume' not supported for multinest. Use radvel's 'proceed' instead."
        )
    run_kwargs["resume"] = proceed

    os.makedirs(output_dir, exist_ok=tmp or overwrite or proceed)

    def loglike(p: ArrayLike, ndim: int, nparams: int) -> float:
        """Log-likelihood for multinest
        Must support ndim and nparams arguments
        and create a list-copy of the object to avoid segfault.
        """
        # This is required to avoid segfault
        # See here: https://github.com/JohannesBuchner/PyMultiNest/issues/41, which I semi-understand
        p = [p[i] for i in range(ndim)]
        return post.likelihood_ns_array(p)

    def prior_transform(u: ArrayLike, ndim: int, nparams: int) -> None:
        """Prior transform for multinest

        Multinest requires the prior transform to handle ndim, nparams arguments
        and to modify the array in-place
        """
        post.prior_transform(u, inplace=True)

    ndim = len(post.name_vary_params())

    pymultinest.run(loglike, prior_transform, ndim, **run_kwargs)

    a = pymultinest.Analyzer(
        outputfiles_basename=run_kwargs["outputfiles_basename"], n_params=ndim
    )

    results = {}
    results["samples"] = a.get_equal_weighted_posterior()[:, :-1]
    results["lnZ"] = a.get_stats()["global evidence"]
    results["lnZerr"] = a.get_stats()["global evidence error"]

    if tmp:
        shutil.rmtree(output_dir)

    return results


def _run_nautilus(
    post: Posterior,
    output_dir: Optional[str] = None,
    proceed: bool = False,
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling with `Nautilus <https://nautilus-sampler.readthedocs.io/en/latest/api_high.html>`_

    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored. Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "filepath argument.
            The nautilus output is automatically stored in ``nautilus_output.hdf5`` under that location.
        proceed: Continue from previous run in output_dir if available.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments.
            Defaults to ``None``.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the following keys:
            - ``samples``: Samples array with shape ``(nsamples, nparams)``
            - ``lnZ``: Log of the Bayesian evidence
            - ``lnZ``: Statistical uncertainty on the evidence
            - ``sampler``: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.
    """
    from nautilus import Sampler

    sampler_kwargs = sampler_kwargs or {}
    run_kwargs = run_kwargs or {}
    run_kwargs.setdefault("verbose", True)

    if "filepath" in sampler_kwargs:
        raise ValueError(
            "filepath not supported for nautilus. Use radvel's output_dir instead."
        )
    if output_dir is not None:
        sampler_kwargs["filepath"] = os.path.join(output_dir, "nautilus_output.hdf5")
    if "resume" in sampler_kwargs:
        raise ValueError(
            "'resume' not supported for ultranest. Use radvel's 'proceed' instead."
        )
    sampler_kwargs["resume"] = proceed

    ndim = len(post.name_vary_params())
    sampler = Sampler(
        post.prior_transform, post.likelihood_ns_array, n_dim=ndim, **sampler_kwargs
    )
    sampler.run(**run_kwargs)
    results = {
        "samples": sampler.posterior(equal_weight=True)[0],
        "lnZ": sampler.log_z,
        "lnZerr": sampler.n_eff**-0.5,
        "sampler": sampler,
    }
    return results


BACKENDS = {
    "dynesty-static": _run_dynesty,
    "dynesty-dynamic": _run_dynesty,
    "multinest": _run_multinest,
    "ultranest": _run_ultranest,
    "nautilus": _run_nautilus,
}


def run(
    post: Posterior,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    proceed: bool = False,
    sampler: str = "ultranest",
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """Run nested sampling

    Args:
        post: radvel posterior object
        output_dir: Output directory where the sampler checkpoints and results will be stored.
            Nothing is stored by default.
            **Note**: This replaces the sampler's built-in "checkpoint_file", "log_dir", or "outputfiles_basename" argument.
            Once you specify output there, everything is saved there automatically.
            A ``results.hdf5`` file will also be saved with the results dict, except for the sampler.
        overwrite: Overwrite the results.hdf5 if True. Will be enabled automatically when proceed=True.
        proceed: Resume from a previous run in the same output_dir if available. Also automatically enables overwrite.
        sampler: name of the sampler to use. Should be one of 'ultranest', 'dynesty-static', 'dynesty-dynamic', 'nautilus', or 'multinest'.
            Defaults to 'ultranest'.
        sampler_kwargs: Dictionary of keyword arguments passed to the 'sampler' object from the underlying nested sampling package at initialization.
            See each package's documentation to learn more on the available arguments. This is not available for ``sampler='multinest'``.
            Defaults to ``None``.
        run_kwargs: Dictionary of keyword arguments passed to the 'run' methods from the underlying nested sampling package.
            See each package's documentation to learn more on the available aruments.
    Returns:
        Dictionary of results with the keys below.

        - ``samples``: Samples dataframe with one column per parameters and lnprobability (log-posterior) for each set.
                       The samples are equally weighted, meaning they are equivalent to MCMC samples.
        - ``lnZ``: Log of the Bayesian evidence
        - ``lnZ``: Statistical uncertainty on the evidence
        - ``sampler``: Sampler object used by the nested sampling library. Provides more fine-grained access to the results.

    Link to each package's API documentation:

    - `Ultranest <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`_
    - `Nautilus <https://nautilus-sampler.readthedocs.io/en/latest/api_high.html>`_
    - `Dynesty <https://dynesty.readthedocs.io>`_
    - `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/pymultinest.html#>`_
    """
    post.check_proper_priors()

    # Proceed automatically enables overwrite.
    overwrite = overwrite or proceed

    if output_dir is not None:
        results_file = os.path.join(output_dir, "results.hdf5")
        if os.path.exists(results_file) and not overwrite:
            raise FileExistsError(
                f"Results file {results_file} exists and 'overwrite' is False."
            )

    sampler = sampler.lower()
    if sampler == "pymultinest":
        sampler = "multinest"

    # fmt: off
    if sampler == "ultranest":
        results = _run_ultranest(post, output_dir=output_dir, proceed=proceed, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "dynesty-static":
        results = _run_dynesty(post, sampler_type="static", proceed=proceed, output_dir=output_dir, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "dynesty-dynamic":
        results = _run_dynesty(post, sampler_type="dynamic", proceed=proceed, output_dir=output_dir, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    elif sampler == "multinest":
        if sampler_kwargs is not None and len(sampler_kwargs) > 0:
            raise TypeError("Argument sampler_kwargs is invalid for sampler 'multinest', only run_kwargs is supported")
        results = _run_multinest(post, output_dir=output_dir, proceed=proceed, overwrite=overwrite, run_kwargs=run_kwargs)
    elif sampler == "nautilus":
        results = _run_nautilus(post, output_dir=output_dir, proceed=proceed, sampler_kwargs=sampler_kwargs, run_kwargs=run_kwargs)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'. Available options are {list(BACKENDS.keys())}")
    # fmt: on
    
    df = DataFrame(results["samples"], columns=post.name_vary_params())
    lnprob_arr = np.empty(len(df))
    for i, row in df.iterrows():
        lnprob_arr[i] = post.logprob_array(row.values)
    df["lnprobability"] = lnprob_arr


    results["samples"] = df

    if output_dir is not None:
        with h5py.File(results_file, mode="w") as h5f:
            for key, val in results.items():
                if key == "sampler" or key == "samples":
                    continue
                h5f.create_dataset(key, data=val)
        results["samples"].to_hdf(results_file, key="samples", mode="a")

    return results


def load_results(results_file: str) -> dict:
    """Load nested sampling results dictionary

    Args:
        results_file: Path to hdf5 file containing the results.
    Returns:
        Dictionary with nested sampling results.
        Note that the ``sampler`` key is not saved, so it is not in the dictionary returned by this function.
    """
    results = {}
    with h5py.File(results_file) as h5f:
        results["lnZ"] = np.array(h5f["lnZ"]).item()
        results["lnZerr"] = np.array(h5f["lnZerr"]).item()
    results["samples"] = read_hdf(results_file, key="samples")
    return results
