---
name: writing-radvel-setup-files
description: Use when writing or fixing a RadVel setup file (the star.py passed to radvel -s), choosing a fitting basis, priors or instrument names, or when radvel fails with "must be converted to fitting basis", "No data found for instrument", or a KeyError on gamma_ or jit_.
---

# Writing RadVel setup files

Checked against radvel 1.6.7. Running the fit: see `fitting-rvs-with-radvel`.

A setup file is a Python module. radvel imports it and reads its globals. Working examples ship in the radvel repo under `example_planets/` (not in the Docker image): start from `epic203771098.py` (one instrument), `HD164922.py` (several), `k2-131.py` (GP).

## The contract

| Global | Required | Notes |
|---|---|---|
| `starname`, `nplanets` | yes | |
| `instnames` | yes | each must equal a value in `data.tel`, else "No data found for instrument" |
| `fitting_basis` | yes | one of `radvel.basis.BASIS_NAMES` |
| `params` | yes | a `radvel.Parameters` **already in** `fitting_basis` |
| `data` | yes | DataFrame with at least `time, mnvel, errvel, tel`; velocities in m/s. Keep extra columns `--decorr` needs (`decorr_vars`) |
| `priors` | yes | list, may be empty |
| `time_base` | yes | epoch for `dvdt`/`curv`; missing gives AttributeError. Use `time_base = 0.5 * (data.time.min() + data.time.max())` |
| `bjd0` | no | **plot axis offset only**. Converts nothing |
| `planet_letters` | no | `{1: 'b', 2: 'c'}` |
| `stellar` | for `derive` | `dict(mstar=, mstar_err=)` in Msun; without it masses are nan |
| `planet` | no | `rp1=, rp_err1=` (Earth radii) for densities |

Every instrument needs `params['gamma_<inst>']` and `params['jit_<inst>']`. Jitter is `jit_`, linear, never `logjit`. `dvdt` and `curv` always exist: set them `vary=False` unless you fit a trend.

Rename columns yourself: bundled `epic203771098.csv` has `t, vel, errvel` and no `tel`; `164922_fixed.txt` is whitespace-separated (`sep=r'\s+'`) with an extra `svalue` column. Read data relative to the setup file, `os.path.dirname(os.path.abspath(__file__))`, or `radvel.DATADIR` for bundled examples.

Instrument names may contain hyphens (`harps-n`). CPS files code Keck/HIRES before its Aug 2004 upgrade as `k`, after as `j`, and APF as `a`. Other unlabeled codes: identify them from each code's time range and say so.

## Basis and parameters

Start values can be in any basis, then convert:

```python
anybasis = radvel.Parameters(nplanets, basis='per tc e w k')
anybasis['per1'] = radvel.Parameter(value=20.885258)
# ... tc1, e1, w1, k1, dvdt, curv, gamma_j, jit_j
params = anybasis.basis.to_any_basis(anybasis, 'per tc secosw sesinw k')
params['per1'].vary = False   # set vary AFTER converting
```

Default to `per tc secosw sesinw k` with no prior on `k`. A flat K that may go negative keeps weak signals from being biased upward. `PositiveKPrior`, `Jeffreys` on `k`, and a `logk` basis are all priors on K: use one only on purpose and say so. Add `EccentricityPrior(nplanets)` whenever e can vary.

Circular: `secosw1`, `sesinw1` = 0, `vary=False`. Say so in the results.

Faster fit, non-GP likelihoods only: `gamma_<inst> = radvel.Parameter(0., vary=False, linear=True)` marginalizes gamma analytically. With a GP this would fix gamma at 0, so leave gamma varying.

## Time system

Data times, `tc`, `tp`, and Gaussian priors on them must use **one** system. Check the ranges before fitting: full BJD is about 2.45e6, BJD-2450000 is 4 digits, K2 BKJD (BJD-2454833) is about 2000-3500, TESS BTJD (BJD-2457000) is about 1300-4000. Ranges overlap, so check against the transit epoch. Unlabeled times: say which system you assumed and why. A wrong guess fits without error and gives nonsense phases.

Shift one side, then set `bjd0` to whatever offset remains so plot axes read right.

## Priors

`radvel.prior.Gaussian(name, mu, sigma)`, `.HardBounds(name, lo, hi)`, `.Jeffreys(name, lo, hi)`. Priors on `vary=False` parameters still count: a Gaussian adds a constant and constrains nothing, and a HardBounds that excludes the fixed value makes the posterior -inf. Transit P and tc are either fixed or free with a Gaussian prior, not both.

After `radvel fit`, check that no parameter sits on a HardBounds edge. Jitter at its cap means the bound is too tight or a signal is missing. Set jitter caps well above each instrument's RV rms, not near the errors.

## GP

Add `hnames` (instrument to hyperparameter names) and optionally `kernel_name`. Copy the structure from `example_planets/k2-131.py`.
