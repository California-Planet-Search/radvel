---
name: fitting-rvs-with-radvel
description: Use when fitting radial-velocity (RV) data with RadVel (the radvel CLI) for Keplerian orbits, planet masses, MCMC uncertainties or a results PDF, or when a radvel command fails with "unrecognized arguments", a PermissionError in Docker, or a missing ic_compare.
---

# Fitting RVs with RadVel

Checked against radvel 1.6.7. Setup files: see `writing-radvel-setup-files`.

## Pipeline

Each step reads the setup file (`-s`) and the output dir (`-d`, default: the setup file name without `.py`). Output files are prefixed with the setup file name, not `starname`. Run the steps in order.

```
radvel fit    -s star.py                      # MAP fit, always first
radvel plot   -s star.py -t rv                # look before sampling
radvel mcmc   -s star.py --headless           # posteriors
radvel derive -s star.py                      # needs mcmc chains + stellar=dict(mstar=, mstar_err=)
radvel ic     -s star.py -t nplanets e trend  # silent; writes <dir>/<starname>_radvel.stat
radvel plot   -s star.py -t rv corner derived
radvel report -s star.py                      # <dir>/<starname>_results.pdf
```

## Flags that trip agents

| Wrong | Right |
|---|---|
| `-o DIR` (in the docs) | `-d DIR` |
| `--ensembles`, `--nrun` | `--nensembles`, `--nsteps` |
| `-t rv -t corner` (keeps only the last) | `-t rv corner` |
| `table -t ic_compare` before `ic` | run `radvel ic` first |

## Outputs

- `plot` writes PDFs only.
- `derive` prints no values: read `<dir>/<name>_derived_quantiles.csv`. `mpsini` is in Earth masses, `a` in AU.
- MCMC summaries (`_post_summary.csv`, chains) are in the fitting basis. In a `secosw sesinw` basis with both varying, get e from the chains: `e = secosw**2 + sesinw**2`. Fixed parameters are not in the chains (a circular fit has neither column). `radvel table -t params` and `report` give e and w, but in 1.6.7 they crash with "assignment destination is read-only" when a w posterior sits beyond 90 degrees.
- `fit`'s "Final loglikelihood" includes the priors. To compare fits with different priors, load `<dir>/<name>_post_obj.pkl` and use `post.likelihood.logprob()`.

## MCMC

- The defaults (`--maxGR 1.01 --minTz 1000`) stop the run once it converges. Only converged runs are publishable. Never loosen them for a final result.
- Runs can take tens of minutes: fix what you aren't measuring (`vary=False`), bound `tc` and `per` with priors, run it in the background. Exploring only? Lower `--nsteps` and label results "not converged".
- Each ensemble is one process, so keep `--nensembles` at or below your core count.
- `--save` lets `--proceed` resume, but a killed run leaves a corrupt chain file, so restart it clean.
- Near-equal period minima (aliases): MCMC stays in one and its error bar is too small. Fit from each minimum (start `per` there), compare the likelihoods, and report both if they are close.

## In Docker

`ghcr.io/california-planet-search/radvel-api:<version>` ships radvel and LaTeX. It runs as uid 10001, so bind mounts fail with `PermissionError`. Run as yourself:

```
docker run --rm --name radvel-mystar --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work --entrypoint radvel \
  ghcr.io/california-planet-search/radvel-api:1.6.7 fit -s star.py
```

`timeout docker run` stops only the client, and the container keeps running. To stop one, `docker kill radvel-mystar`. Never kill containers host-wide.

## Before quoting numbers

- Jitter well above the errors means a missing signal: periodogram the residuals.
- P and tc fixed from transits: refit with tc free. If lnL improves by more than about 5, the ephemeris and RVs disagree. Stop. Report both fits and the offset, and quote no mass until the Tc and time system (BJD vs BJD-2450000, `bjd0`) are checked.
- `radvel fit` prints `tc`/`tp` minus time_base when they are above 1e6 (full BJD) and unchanged otherwise. The CSVs in `-d` hold the values in your input time system.
- Circular orbits are a choice. Say so, or fit `secosw`/`sesinw`.
