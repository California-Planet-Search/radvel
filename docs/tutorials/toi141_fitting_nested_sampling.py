# %%
from pandas import read_csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import radvel

# %%
data_df = read_csv(os.path.join(radvel.DATADIR, "rvs_toi141.dat"), sep=" ", names=["t", "rv", "erv", "inst"])
t, vel, errvel, inst = data_df.t.values, data_df.rv.values, data_df.erv.values, data_df.inst.values
min_time = np.min(t[inst == "FEROS"]) - 30.0
max_time = np.max(t[inst == "FEROS"]) + 30.0
t_mod = np.linspace(min_time, max_time, 1000)

inst_groups = data_df.groupby("inst").groups
inst_names = list(inst_groups.keys())

T_OFF = 2457000.0

# %%
def plot_data(t, vel, errvel, post=None):
    for inst in inst_names:
        inds = inst_groups[inst]
        if post is not None:
            vel = vel - post.params[f"gamma_{inst}"].value
            errvel_inflated = np.sqrt(errvel**2 + post.params[f"jit_{inst}"].value**2)
        plt.errorbar(t[inds] - T_OFF, vel[inds], yerr=errvel[inds], fmt=".", label=inst)
        if post is not None:
            plt.errorbar(t[inds] - T_OFF, vel[inds], yerr=errvel_inflated[inds], fmt="None", ecolor="r", capsize=0, zorder=-10)
    plt.legend()

# %%
plot_data(t, vel, errvel)
plt.show()

# %%
params = radvel.Parameters(1, basis="per tc e w k")

P = 1.007917
P_err = 0.000073
t0 = 2458325.5386
t0_err = 0.0011
params["per1"] = radvel.Parameter(value=P)
params["tc1"] = radvel.Parameter(value=t0)
params["e1"] = radvel.Parameter(value=0.0, vary=False)
params["w1"] = radvel.Parameter(value=90.0 * np.pi / 180.0, vary=False)
params["k1"] = radvel.Parameter(value=10.0)
params['dvdt'] = radvel.Parameter(value=0.,vary=False)
params['curv'] = radvel.Parameter(value=0.,vary=False)

model = radvel.RVModel(params)

# %%
likes = []
for inst in inst_names:
    like_inst = radvel.RVLikelihood(model, t, vel, errvel, suffix=f"_{inst}")
    indices = inst_groups[inst]
    like_inst.params['gamma_'+inst] = radvel.Parameter(value=np.mean(vel[indices]), vary=True)
    like_inst.params['jit_'+inst] = radvel.Parameter(value=np.mean(errvel[indices]), vary=True)
    likes.append(like_inst)

# %%
like = radvel.CompositeLikelihood(likes)

# %%
post = radvel.posterior.Posterior(like)

post.priors += [radvel.prior.Gaussian("per1", P, P_err)]
post.priors += [radvel.prior.Gaussian("tc1", t0, t0_err)]
post.priors += [radvel.prior.HardBounds("gamma_CORALIE14", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_CORALIE07", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_HARPS", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_FEROS", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("k1", 0.0, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_CORALIE14", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_CORALIE07", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_HARPS", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_FEROS", 1e-3, 100.0)]


res = optimize.minimize(
    post.neglogprob_array, post.get_vary_params(), method='Nelder-Mead',
    options=dict(maxiter=200, maxfev=100000, xatol=1e-8)
)
post.set_vary_params(res.x)

# %%
plot_data(t, vel, errvel)
plt.plot(t_mod - T_OFF, post.model(t_mod))
plt.show()

# %%
from typing import Optional, Union
import dynesty

def run_ns(
    post: radvel.posterior.Posterior,
    sampler_type: str = "static",
    sampler_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
) -> Union[dynesty.NestedSampler, dynesty.DynamicNestedSampler]:
    run_kwargs = run_kwargs or {}
    sampler_kwargs = sampler_kwargs or {}
    post.check_proper_priors()
    # TODO: Support dynamic
    if sampler_type == "static":
        sampler = dynesty.NestedSampler(
            post.likelihood.logprob_array, post.prior_transform, len(post.priors), **sampler_kwargs
        )
    elif sampler_type == "dynamic":
        sampler = dynesty.DynamicNestedSampler(post.likelihood.logprob_array, post.prior_transform, len(post.priors), **sampler_kwargs)
    else:
        raise ValueError("sampler must be 'static' or 'dynamic'")
    sampler.run_nested(**run_kwargs)
    return sampler

# %%
sampler = run_ns(post, sampler_kwargs={"nlive": 300})
sampler_single = sampler

# %%
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal

# %%
dyplot.runplot(sampler_single.results)
plt.show()

# %%
dyplot.traceplot(sampler_single.results, labels=post.name_vary_params())
plt.show()

# %%
dyplot.cornerplot(sampler_single.results, labels=post.name_vary_params(), show_titles=True)
plt.show()

# %%
def get_samples(results):
    weights = np.exp(results['logwt'] - results['logz'][-1])
    posterior_samples = resample_equal(results.samples, weights)
    return posterior_samples
samples_single = get_samples(sampler_single.results)
lnz_single = sampler_single.results.logz[-1]
lnz_err_single = sampler_single.results.logzerr[-1]

# %%
med_params = np.median(samples_single, axis=0)
post.set_vary_params(med_params)

# %%
from radvel.plot import orbit_plots
RVPlot = orbit_plots.MultipanelPlot(post)
RVPlot.plot_multipanel()
plt.show()

# %%
plot_data(t, vel, errvel, post=post)
plt.plot(t_mod - T_OFF, post.model(t_mod))
plt.ylim([-20,20])
plt.xlim([1365,1435])
plt.show()

# %%
params = radvel.Parameters(2, basis="per tc e w k")

P = 1.007917
P_err = 0.000073
t0 = 2458325.5386
t0_err = 0.0011
params["per1"] = radvel.Parameter(value=P)
params["tc1"] = radvel.Parameter(value=t0)
params["e1"] = radvel.Parameter(value=0.0, vary=False)
params["w1"] = radvel.Parameter(value=90.0 * np.pi / 180.0, vary=False)
params["k1"] = radvel.Parameter(value=10.0)
params["per2"] = radvel.Parameter(value=3.0)
params["tc2"] = radvel.Parameter(value=2458325+2)
params["e2"] = radvel.Parameter(value=0.0, vary=False)
params["w2"] = radvel.Parameter(value=90.0 * np.pi / 180.0, vary=False)
params["k2"] = radvel.Parameter(value=10.0)
params['dvdt'] = radvel.Parameter(value=0.,vary=False)
params['curv'] = radvel.Parameter(value=0.,vary=False)

model = radvel.RVModel(params)

# %%
likes = []
for inst in inst_names:
    like_inst = radvel.RVLikelihood(model, t, vel, errvel, suffix=f"_{inst}")
    indices = inst_groups[inst]
    like_inst.params['gamma_'+inst] = radvel.Parameter(value=np.mean(vel[indices]), vary=True)
    like_inst.params['jit_'+inst] = radvel.Parameter(value=np.mean(errvel[indices]), vary=True)
    likes.append(like_inst)

# %%
like = radvel.CompositeLikelihood(likes)

# %%
post = radvel.posterior.Posterior(like)

post.priors += [radvel.prior.Gaussian("per1", P, P_err)]
post.priors += [radvel.prior.Gaussian("tc1", t0, t0_err)]
post.priors += [radvel.prior.HardBounds("gamma_CORALIE14", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_CORALIE07", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_HARPS", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("gamma_FEROS", -100.0, 100.0)]
post.priors += [radvel.prior.HardBounds("k1", 0.0, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_CORALIE14", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_CORALIE07", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_HARPS", 1e-3, 100.0)]
post.priors += [radvel.prior.Jeffreys("jit_FEROS", 1e-3, 100.0)]
post.priors += [radvel.prior.HardBounds("per2", 1.0, 10.0)]
post.priors += [radvel.prior.HardBounds("tc2", 2458325.0, 2458330.0)]
post.priors += [radvel.prior.HardBounds("k2", 0.0, 100.0)]


res = optimize.minimize(
    post.neglogprob_array, post.get_vary_params(), method='Nelder-Mead',
    options=dict(maxiter=200, maxfev=100000, xatol=1e-8)
)
post.set_vary_params(res.x)

# %%
plot_data(t, vel, errvel)
plt.plot(t_mod - T_OFF, post.model(t_mod))
plt.ylim([-20,20])
plt.xlim([1365,1435])
plt.show()

# %%
sampler = run_ns(post, sampler_kwargs={"nlive": 300})
sampler_two = sampler

# %%
dyplot.runplot(sampler_two.results)
plt.show()

# %%
dyplot.traceplot(sampler_two.results, labels=post.name_vary_params())
plt.show()

# %%
dyplot.cornerplot(sampler_two.results, labels=post.name_vary_params(), show_titles=True)
plt.show()

# %%
samples_two = get_samples(sampler_two.results)
lnz_two = sampler_two.results.logz[-1]
lnz_err_two = sampler_two.results.logzerr[-1]

# %%
med_params = np.median(samples_two, axis=0)
post.set_vary_params(med_params)

# %%
plot_data(t, vel, errvel, post=post)
plt.plot(t_mod - T_OFF, post.model(t_mod), "k")
plt.ylim([-20,20])
plt.xlim([1365,1435])
plt.show()

