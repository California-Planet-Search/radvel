from matplotlib.pylab import *
import pylab as pl
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable,AxesGrid
from matplotlib.ticker import NullFormatter
import radvel
from radvel.utils import *
from astropy.time import Time
import string
from matplotlib import rcParams

rcParams['font.size'] = 24

telfmts = {'j': 'ko', 'k': 'ks', 'a': 'gd', 'h': 'gs',
           'hires_rj': 'ko', 'hires_rk': 'ks', 'apf': 'gd', 'harps': 'gs'}
teldecode = {'a': 'APF', 'k': 'HIRES_k', 'j': 'HIRES_j'}
msize = 7
elinecolor = '0.6'


def _errtot(params, errvel):
    jitter = 10**params['logjitter']
    return np.sqrt(jitter**2 + errvel**2)

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


def add_at(ax, t, loc=2):
    import matplotlib.patheffects as PathEffects
    path_effects = [PathEffects.withStroke(linewidth=4,foreground="w")]
    fp = dict(size=16, color='k', path_effects=path_effects)
    _at = AnchoredText(t, loc=loc, prop=fp, frameon=False)
    ax.add_artist(_at)
    return _at

def _mtelplot(x, y, e, tel, ax):
        utel = np.unique(tel)
        for t in utel:
            xt = x[tel == t]
            yt = y[tel == t]
            et = e[tel == t]
            if t == '': t = 'j'
            if t == 'j' or t == 'k':
                ax.errorbar(xt,yt,yerr=et,fmt=telfmts[t], ecolor=elinecolor, markersize=msize, capsize=0, markeredgecolor=telfmts[t][0], markerfacecolor='none',
                            markeredgewidth=2)
            elif t not in telfmts.keys():
                ax.errorbar(xt,yt,yerr=et,fmt='o', ecolor=elinecolor, markersize=msize, capsize=0, markeredgewidth=0)
            else:
                ax.errorbar(xt,yt,yerr=et,fmt=telfmts[t], ecolor=elinecolor, markersize=msize, capsize=0, markeredgecolor=telfmts[t][0], markerfacecolor=telfmts[t][0],
                            markeredgewidth=3)

def rv_multipanel_plot(post, saveplot=None):

    if saveplot != None: resolution = 1e4
    else: resolution = 2000

    cpsparams = post.params.basis.to_cps(post.params)
    model = post.likelihood.model
    rvtimes = post.likelihood.x
    rvdat = post.likelihood.y
    rverr = post.likelihood.yerr
    n = model.num_planets
    e = 2450000

    if isinstance(post.likelihood, radvel.likelihood.CompositeLikelihood):
        like_list = post.likelihood.like_list
    else:
        like_list = [ post.likelihood ]
    
    periods = []
    for i in range(model.num_planets):
        periods.append(cpsparams['per%d' % (i+1)])
    longp = max(periods)
    shortp = min(periods)
        
    dt = max(rvtimes)-min(rvtimes)
    rvmodt = np.linspace(min(rvtimes)-0.05*dt,max(rvtimes)+0.05*dt+longp,resolution)

    rvmod2 = model(rvmodt)
    rvmod = model(rvtimes)

    rawresid = post.likelihood.residuals()
    resid = rawresid + cpsparams['dvdt']*(rvtimes-model.time_base) + cpsparams['curv']*(rvtimes-model.time_base)**2
    slope = cpsparams['dvdt']*(rvmodt-model.time_base) + cpsparams['curv']*(rvmodt-model.time_base)**2
    slope_low = cpsparams['dvdt']*(rvtimes-model.time_base) + cpsparams['curv']*(rvtimes-model.time_base)**2

    if n == 1: fig = pl.figure(figsize=(19.0,16.0))
    else: fig = pl.figure(figsize=(19.0,16.0+4*n))        
    rect = [0.07, 0.64, 0.865, 1./(n+1)]
    axRV = pl.axes(rect)
    pl.subplots_adjust(left=0.1,top=0.95,right=0.95)
    plotindex = 1
    pltletter = ord('a')
    ax = axRV
    
    #Unphased plot
    ax.axhline(0, color='0.5', linestyle='--', lw=2)
    ax.plot(rvmodt-e,rvmod2,'b-',linewidth=1, rasterized=False)
    ax.annotate("%s)" % chr(pltletter), xy=(0.01,0.85), xycoords='axes fraction', fontsize=28, fontweight='bold')
    pltletter += 1
    _mtelplot(rvtimes-e,rawresid+rvmod,rverr,post.likelihood.telvec, ax)
    ax.set_xlim(min(rvtimes-e)-0.01*dt,max(rvtimes-e)+0.01*dt)
    
    pl.setp(axRV.get_xticklabels(), visible=False)

    # Years on upper axis
    axyrs = axRV.twiny()
    axyrs.set_xlim(min(rvtimes-e)-0.01*dt,max(rvtimes-e)+0.01*dt)
    #yrticklocs = [date2jd(datetime(y, 1, 1, 0, 0, 0))-e for y in [1998, 2002, 2006, 2010, 2014]]
    yrticklocs = []
    yrticklabels = []
    for y in [1988,1992,1996,2000,2004,2008,2012,2016]:
        jd = Time("%d-01-01T00:00:00" % y, format='isot', scale='utc').jd - e
        if jd > ax.get_xlim()[0] and jd < ax.get_xlim()[1]:
            yrticklocs.append(jd)
            yrticklabels.append("%d" % y)
    axyrs.set_xticks(yrticklocs)
    axyrs.set_xticklabels(yrticklabels)    
    if len(yrticklabels) > 0:
        pl.xlabel('Year')
        axyrs.grid(False)

    
    ax.set_ylabel('RV [m s$^{-1}$]')
    ticks = ax.yaxis.get_majorticklocs()
    ax.yaxis.set_ticks(ticks[1:])

    divider = make_axes_locatable(axRV)
    axResid = divider.append_axes("bottom",size="50%",pad=0.0,sharex=axRV,sharey=None)
    ax = axResid

    #Residuals
    ax.plot(rvmodt-e,slope,'b-',linewidth=3)
    ax.annotate("%s)" % chr(pltletter), xy=(0.01,0.80), xycoords='axes fraction', fontsize=28, fontweight='bold')
    pltletter += 1

    _mtelplot(rvtimes-e,resid,rverr, post.likelihood.telvec,ax)
    ax.set_ylim(-9,9)
    ax.set_xlim(min(rvtimes-e)-0.01*dt,max(rvtimes-e)+0.01*dt)
    ticks = ax.yaxis.get_majorticklocs()
    ax.yaxis.set_ticks([ticks[0],0.0,ticks[-1]])
    xticks = ax.xaxis.get_majorticklocs()
    pl.xlabel('BJD$_{\\mathrm{TDB}}$ - %d' % e)
    ax.set_ylabel('Residuals')

    
    # Define the locations for the axes
    axbounds = ax.get_position().bounds
    bottom = axbounds[1]
    height = (bottom - 0.10) / n
    textloc = bottom / 2
    bottom -= height + 0.05
    left, width = 0.07, 0.75

    
    #Phase plots
    for i in range(n):
        pnum = i+1
        print "Planet %d" % pnum

        rvdat = rvdat.copy()

        rvmod2 = model(rvmodt, planet_num=pnum) - slope
        
        modph = t_to_phase(post.params, rvmodt, pnum, cat=True) - 1

        rvdat = rawresid + model(rvtimes, planet_num=pnum) - slope_low
        
        phase = t_to_phase(post.params, rvtimes, pnum, cat=True) - 1
        p2 = t_to_phase(post.params, rvtimes, pnum, cat=False) - 1

        rvdatcat = np.concatenate((rvdat,rvdat))
        rverrcat = np.concatenate((rverr,rverr))
        rvmod2cat = np.concatenate((rvmod2,rvmod2))

        bint, bindat, binerr = fastbin(phase+1, rvdatcat, nbins=25)
        bint -= 1.0

        rect = [left, bottom-(i)*height, (left+width)+0.045, height]
        if n == 1: rect[1] -= 0.03
        ax = pl.axes(rect)

        ax.axhline(0, color='0.5', linestyle='--', lw=2)
        ax.plot(sorted(modph),rvmod2cat[np.argsort(modph)],'b-',linewidth=3)
        ax.annotate("%s)" % chr(pltletter), xy=(0.01,0.85), xycoords='axes fraction', fontsize=28, fontweight='bold')
        pltletter += 1

        _mtelplot(phase,rvdatcat,rverrcat, np.concatenate((post.likelihood.telvec,post.likelihood.telvec)), ax)
        ax.errorbar(bint,bindat,yerr=binerr,fmt='ro', ecolor='r', markersize=msize*2.5, markeredgecolor='w', markeredgewidth=2)

        pl.xlim(-0.5,0.5)
        meanlim = np.mean([-min(rvdat), max(rvdat)])
        meanlim += 0.10*meanlim
        pl.ylim(-meanlim, meanlim)
        
        letters = string.lowercase
        planetletter = letters[i+1]
        labels = ['$P_{\\rm %s}$' % planetletter,'$K_{\\rm %s}$' % planetletter,'$e_{\\rm %s}$' % planetletter]
        units = ['days','m s$^{-1}$','']
        indicies = [0,4,2,2]
        spacing = 0.09
        xstart = 0.65
        ystart = 0.89
        
        if i < n-1:
            ticks = ax.yaxis.get_majorticklocs()
            ax.yaxis.set_ticks(ticks[1:-1])

        if n > 1: fig.text(0.01,textloc,'RV [m s$^{-1}$]',rotation='vertical',ha='center',va='center',fontsize=28)
        else: pl.ylabel('RV [m s$^{-1}$]')
        pl.xlabel('Phase')

        print_params = ['per', 'k', 'e']
        for l,p in enumerate(print_params):
            txt = ax.annotate('%s = %4.2f %s' % (labels[l],cpsparams["%s%d" % (print_params[l],pnum)] ,units[l]),(xstart,ystart-l*spacing),
                                    xycoords='axes fraction', fontsize=28)
        

    if saveplot != None:
        fig = pl.gcf()
        #fig.set_size_inches((10,8+1.5*n))
        pl.savefig(saveplot,dpi=150)
    else: pl.show()

