from matplotlib.pylab import *

def _errtot(params, errvel):
    jitter = 10**params['logjitter']
    return np.sqrt(jitter**2 + errvel**2)

def t_to_phase(params, t, num_planet):
    P = params['P%i' % num_planet].value
    tc = params['tc%i' % num_planet].value
    phase = np.mod(t - tc, P) 
    phase /= P
    return phase

def phase_to_t(params, phase, num_planet):
    P = params['P%i' % num_planet].value
    tc = params['tc%i' % num_planet].value
    t = phase * P
    t += tc
    return t

def plot_phasefolded(mod, params, num_planet, t, vel, errvel, extra=True):
    phasei = np.linspace(-0.25,1.25,1000)
    phase = t_to_phase(params, t, num_planet)
    ti = phase_to_t(params, phasei, num_planet)
    model = lambda t : mod.model_single_planet(params, t, num_planet)
    resid = mod.residuals(params, t, vel)
    
    errorbarkw = dict(yerr=errvel, fmt='o',capsize=0,ecolor='k',lw=0.5, ms=4, zorder=5)
    errorbar( phase, resid + model(t), **errorbarkw )
    plot(phasei, model(ti),zorder=6, color='Tomato')
    xlabel('Orbital Phase')
    
    if extra:
        errorbarkw['mfc'] = 'w'
        errorbar( phase -1, resid + model(t), **errorbarkw )
        errorbar( phase +1, resid + model(t), **errorbarkw )

    grid(zorder=0)

def plotfit(mod, params, t, vel, errvel, **kwargs):
    pad = 30
    nptsi = 1000
    ti = np.linspace(t[0] - pad, t[-1] + pad, nptsi)
    fit = mod.model(params, ti)
    plot(ti, fit, **kwargs)

    
def rvplot(mod, params, t, vel, errvel):
    """
    Make a plot showing the best fit rv models
    """
    # Constants
    phasepad = 0.25

    fig = plt.figure(figsize=(7.5,5))
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
    ax2 = plt.subplot2grid((2,2), (1,0))
    ax3 = plt.subplot2grid((2,2), (1,1), sharey=ax2, sharex=ax2)

    # Full time-series
    sca(ax1)
    
    resid = vel - mod.model(params, t)
    errtotal = _errtot(params, errvel)
    nresid = resid/errtotal
    chi2 = np.sum(nresid**2 )
    dof = vel.size - len(mod.vary_parameters) - 1
    redchi2 = chi2 / dof

    plotfit(mod, params, t, vel, errvel,color='Tomato')
    errorbar( t, vel, yerr=errtotal, fmt='.', capsize=0, color='k',zorder=1.9,lw=0.5)
    print "chi2 = %.1f" % redchi2

    grid()

    sca(ax2)
    plot_phasefolded(mod, params, 1, t, vel, errvel)
    ylabel('Radial Velocity (m/s)')
    sca(ax3)
    plot_phasefolded(mod, params, 2, t, vel, errvel)

    ylim(-15,15)
    xlim(0 - phasepad, 1 + phasepad)

from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

def add_at(ax, t, loc=2):
    import matplotlib.patheffects as PathEffects
    path_effects = [PathEffects.withStroke(linewidth=4,foreground="w")]
    fp = dict(size=16, color='k', path_effects=path_effects)
    _at = AnchoredText(t, loc=loc, prop=fp, frameon=False)
    ax.add_artist(_at)
    return _at
