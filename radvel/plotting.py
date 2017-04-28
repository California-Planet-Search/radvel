import os
import string
import copy

import numpy as np
import pylab as pl
import matplotlib
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib import rcParams, gridspec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.time import Time
import corner

import radvel
from radvel.utils import t_to_phase, fastbin, round_sig, sigfig

latex = {
    'ms':'m s$^{\mathregular{-1}}$',
    'BJDTDB':'BJD$_{\mathregular{TDB}}$'
}

telfmts_default = {
    'j': dict(color='k',fmt='o',mfc='none',label='HIRES',mew=1),
    'k': dict(color='k',fmt='s',mfc='none',label='HIRES pre 2004',mew=1),
    'a': dict(color='g',fmt='d',label='APF'),
    'pfs': dict(color='magenta',fmt='p',label='PFS'),
    'h': dict(color='firebrick',fmt="s",label='HARPS'),
    'harps-n': dict(color='firebrick',fmt='^',label='HARPS-N'),
    'l': dict(color='g',fmt='*'),
}

telfmts_default['lick'] = telfmts_default['l']
telfmts_default['hires_rj'] = telfmts_default['j']
telfmts_default['hires'] = telfmts_default['j']
telfmts_default['hires_rk'] = telfmts_default['k']
telfmts_default['apf'] = telfmts_default['a']
telfmts_default['harps'] = telfmts_default['h']

cmap = matplotlib.cm.nipy_spectral
rcParams['font.size'] = 9
rcParams['lines.markersize'] = 5
rcParams['axes.grid'] = False
    
def _mtelplot(x, y, e, tel, ax, telfmts={}):
    """Plot data from from multiple telescopes

    x (array): Either time or phase
    y (array): RV
    e (array): RV error
    tel (array): telecsope string key
     telfmts (dict): dictionary of dictionaries corresponding to kwargs 
        passed to errorbar. Example:

        telfmts = {
             'hires': dict(fmt='o',label='HIRES',msize=),
             'harps-n' dict(fmt='s',)}  
    
    """

    lw = 1.0

    default_colors = ['orange', 'purple', 'magenta' , 'pink']
    ci = 0
    
    utel = np.unique(tel)
    for t in utel:
        xt = x[tel == t]
        yt = y[tel == t]
        et = e[tel == t]

        # Default formatting
        kw = dict(
            fmt='o', capsize=0, mew=0, 
            ecolor='0.6', lw = lw, color=default_colors[ci],
            label = t
        )

        # If not explicit format set, look among default formats
        telfmt = {}
        if not telfmts.has_key(t) and telfmts_default.has_key(t):
            telfmt = telfmts_default[t]
        if telfmts.has_key(t):
            telfmt = telfmts[t]
        if not telfmts.has_key(t) and not telfmts_default.has_key(t):
            ci += 1
        for k in telfmt:
            kw[k] = telfmt[k]

        pl.errorbar(xt, yt, yerr=et, **kw)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )

def rv_multipanel_plot(post, saveplot=None, telfmts={}, nobin=False, 
                       yscale_auto=False, yscale_sigma=3.0, nophase=False, 
                       epoch=2450000, uparams=None, phase_ncols=None, 
                       phase_nrows=None, legend=True, rv_phase_space=0.07):
    """Multi-panel RV plot to display model using post.params orbital paramters.

    Args:
        post (radvel.Posterior): Radvel posterior object. The model
            plotted will be generated from post.params
        saveplot (string, optional): Name of output file, will show as
             interactive matplotlib window if not defined.
        nobin (bool, optional): If True do not show binned data on
             phase plots. Will default to True if total number of
             measurements is less then 20.
        yscale_auto (bool, optional): Use matplotlib auto y-axis
             scaling (default: False)
        yscale_sigma (float, optional): Scale y-axis limits to be +/-
             yscale_sigma*(RMS of data plotted)
        telfmts (dict, optional): dictionary of dictionaries mapping
             instrument code to plotting format code.
        nophase (bool, optional): Will omit phase-folded plots if true
        epoch (float, optional): Subtract this value from the time axis for
            more compact axis labels (default: 245000)
        uparams (dict, optional): parameter uncertainties, must
           contain 'per', 'k', and 'e' keys.
        phase_ncols (int, optional): number of columns in the phase
            folded plots. Default behavior is 1.
        phase_nrows (int, optional): number of columns in the phase
            folded plots. Default is nplanets.
        legend (bool, optional): include legend on plot? (default: True)
        rv_phase_space (float, optional): verticle space between rv
            plot and phase-folded plots (in units of fraction of
            figure height)

    Returns:
        figure: current matplotlib figure object
        list: list of axis objects

    """
    figwidth = 7.5 # spans a page with 0.5in margins
    phasefac = 1.4
    ax_rv_height = figwidth * 0.6
    ax_phase_height = ax_rv_height / phasefac
    bin_fac = 1.75
    bin_markersize = bin_fac * rcParams['lines.markersize']
    bin_markeredgewidth = bin_fac * rcParams['lines.markeredgewidth']
    fit_linewidth = 3.0

    cpspost = copy.deepcopy(post) 
    model = cpspost.likelihood.model
    cpsparams = post.params.basis.to_cps(post.params)
    cpspost.params.update(cpsparams)
    rvtimes = cpspost.likelihood.x
    rvdat = cpspost.likelihood.y
    rverr = cpspost.likelihood.errorbars()
    num_planets = model.num_planets

    if nophase:
        num_planets = 1
        periods = [max(rvtimes) - min(rvtimes)]

    if phase_ncols is None:
        phase_ncols = 1
    if phase_nrows is None:
        phase_nrows = num_planets

    ax_phase_height /= phase_ncols
        
    e = epoch
    if len(post.likelihood.x) < 20: 
        nobin = True
    
    if saveplot != None: 
        resolution = 10000
    else: 
        resolution = 2000


    if isinstance(cpspost.likelihood, radvel.likelihood.CompositeLikelihood):
        like_list = cpspost.likelihood.like_list
    else:
        like_list = [ cpspost.likelihood ]
    
    if not nophase:
        periods = []
        for i in range(num_planets):
            periods.append(cpsparams['per%d' % (i+1)])
            
    longp = max(periods)
    shortp = min(periods)
        
    dt = max(rvtimes) - min(rvtimes)
    rvmodt = np.linspace(
        min(rvtimes) - 0.05 * dt, max(rvtimes) + 0.05 * dt + longp, int(resolution)
    )
    
    rvmod2 = model(rvmodt)
    rvmod = model(rvtimes)

    if ((rvtimes - e) < -2.4e6).any():
        plttimes = rvtimes
        mplttimes = rvmodt
    elif e == 0:
        e = 2450000
        plttimes = rvtimes - e
        mplttimes = rvmodt - e
    else:
        plttimes = rvtimes - e
        mplttimes = rvmodt - e

    rawresid = cpspost.likelihood.residuals()
    resid = (
        rawresid + cpsparams['dvdt']*(rvtimes-model.time_base) 
        + cpsparams['curv']*(rvtimes-model.time_base)**2
    )
    slope = (
        cpsparams['dvdt'] * (rvmodt-model.time_base) 
        + cpsparams['curv'] * (rvmodt-model.time_base)**2
    )
    slope_low = (
        cpsparams['dvdt'] * (rvtimes-model.time_base) 
        + cpsparams['curv'] * (rvtimes-model.time_base)**2
    )


    # Provision figure
    figheight = ax_rv_height + ax_phase_height * phase_nrows
    divide = 1 - ax_rv_height / figheight
    fig = pl.figure(figsize=(figwidth,figheight))
    fig.subplots_adjust(left=0.1)
    gs_rv = gridspec.GridSpec(1, 1)
    gs_rv.update(left=0.12, right=0.93, top=0.93,
                     bottom=divide+rv_phase_space*0.5)
    gs_phase = gridspec.GridSpec(phase_nrows, phase_ncols)
    if phase_ncols==1:
        gs_phase.update(left=0.12, right=0.93,
                            top=divide-rv_phase_space*0.5,
                            bottom=0.07,hspace=0.003)
    else:
        gs_phase.update(left=0.12, right=0.93,
                            top=divide-rv_phase_space*0.5,
                            bottom=0.07,hspace=0.25,wspace=0.25)

    axL = []
    axRV = pl.subplot(gs_rv[0, 0])
    plotindex = 1
    pltletter = ord('a')
    ax = axRV

    axL += [axRV]
   
    #Unphased plot
    ax.axhline(0, color='0.5', linestyle='--')
    ax.plot(mplttimes,rvmod2,'b-', rasterized=False, lw=0.5*fit_linewidth)

    def labelfig(ax, pltletter):
        text = "{})".format(chr(pltletter))
        add_anchored(
            text,loc=2,prop=dict(fontweight='bold',size='large'),frameon=False
        )

    labelfig(ax,pltletter)

    pltletter += 1
    _mtelplot(
        plttimes,rawresid+rvmod,rverr,cpspost.likelihood.telvec, ax, telfmts
    )
    ax.set_xlim(min(plttimes)-0.01*dt,max(plttimes)+0.01*dt)
    
    pl.setp(axRV.get_xticklabels(), visible=False)

    # Legend
    if legend:
        pl.legend(numpoints=1, fontsize='x-small', loc='best')
    
    # Years on upper axis
    axyrs = axRV.twiny()
#    axyrs.set_xlim(min(plttimes)-0.01*dt,max(plttimes)+0.01*dt)


    xl = np.array(list(ax.get_xlim())) + e
    decimalyear = Time(xl,format='jd',scale='utc').decimalyear
    axyrs.plot(decimalyear,decimalyear)
    axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
    axyrs.set_xlim(*decimalyear)
    axyrs.set_xlabel('Year', fontweight='bold')
    #axyrs.xaxis.set_major_locator(MaxNLocator(8))

    if not yscale_auto: 
        scale = np.std(rawresid+rvmod)
        ax.set_ylim(-yscale_sigma * scale , yscale_sigma * scale)

    ax.set_ylabel('RV [{ms:}]'.format(**latex), weight='bold')
    ticks = ax.yaxis.get_majorticklocs()
    ax.yaxis.set_ticks(ticks[1:])

    divider = make_axes_locatable(axRV)
    axResid = divider.append_axes(
        "bottom",size="50%",pad=0.0,sharex=axRV,sharey=None
    )
    ax = axResid
    axL += [axResid]

    #Residuals
    ax.plot(mplttimes,slope,'b-',lw=fit_linewidth)

    labelfig(ax,pltletter)

    pltletter += 1

    _mtelplot(plttimes,resid,rverr, cpspost.likelihood.telvec,ax, telfmts)
    if not yscale_auto: 
        scale = np.std(resid)
        ax.set_ylim(-yscale_sigma * scale, yscale_sigma * scale)

    ax.set_xlim(min(plttimes)-0.01*dt,max(plttimes)+0.01*dt)
    ax.yaxis.set_ticks([ticks[0],0.0,ticks[-1]])
    xticks = ax.xaxis.get_majorticklocs()
    pl.xlabel('JD - {:d}'.format(int(np.round(e))), weight='bold')
    ax.set_ylabel('Residuals', weight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(5,prune='both'))
    
    # Define the locations for the axes
    axbounds = ax.get_position().bounds
    bottom = axbounds[1]
    height = (bottom - 0.15) / num_planets
    textloc = bottom / 2
    bottom -= height + 0.05
    left, width = 0.10, 0.72

    #Phase plots
    for i in range(num_planets):
        if nophase: break
        
        pnum = i+1
        #print "Planet %d" % pnum

        rvdat = rvdat.copy()
        rvmod2 = model(rvmodt, planet_num=pnum) - slope
        modph = t_to_phase(cpspost.params, rvmodt, pnum, cat=True) - 1
        rvdat = rawresid + model(rvtimes, planet_num=pnum) - slope_low
        phase = t_to_phase(cpspost.params, rvtimes, pnum, cat=True) - 1
        p2 = t_to_phase(cpspost.params, rvtimes, pnum, cat=False) - 1
        rvdatcat = np.concatenate((rvdat,rvdat))
        rverrcat = np.concatenate((rverr,rverr))
        rvmod2cat = np.concatenate((rvmod2,rvmod2))
        bint, bindat, binerr = fastbin(phase+1, rvdatcat, nbins=25)
        bint -= 1.0

        i_row = i / phase_ncols
        i_col = i - i_row * phase_ncols
        ax = pl.subplot(gs_phase[i_row, i_col])
        axL += [ax]

        ax.axhline(0, color='0.5', linestyle='--', )
        ax.plot(sorted(modph),rvmod2cat[np.argsort(modph)],'b-',linewidth=fit_linewidth)
        labelfig(ax,pltletter)

        pltletter += 1
        telcat = np.concatenate((cpspost.likelihood.telvec,cpspost.likelihood.telvec))

        _mtelplot(phase,rvdatcat, rverrcat, telcat, ax, telfmts)
        if not nobin and len(rvdat) > 10: 
            ax.errorbar(
                bint, bindat, yerr=binerr, fmt='ro',mec='w', ms=bin_markersize, 
                mew=bin_markeredgewidth
            )

        pl.xlim(-0.5,0.5)

        if not yscale_auto: 
            scale = np.std(rvdatcat)
            pl.ylim(-yscale_sigma*scale, yscale_sigma*scale )
        
        letters = string.lowercase
        planetletter = letters[i+1]
        keys = [p+str(pnum) for p in ['per', 'k', 'e'] ]
        labels = [cpspost.params.tex_labels().get(k, k) for k in keys]
        if i < num_planets-1:
            ticks = ax.yaxis.get_majorticklocs()
            ax.yaxis.set_ticks(ticks[1:-1])

        pl.ylabel('RV [{ms:}]'.format(**latex), weight='bold')
        pl.xlabel('Phase', weight='bold')

        print_params = ['per', 'k', 'e']
        units = {'per':'days','k':latex['ms'],'e':''}

        anotext = []
        for l, p in enumerate(print_params):
            val = cpsparams["%s%d" % (print_params[l],pnum)]
            
            if uparams is None:
                _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$",""), val, units[p])
            else:
                if hasattr(post, 'medparams'):
                    val = post.medparams["%s%d" % (print_params[l],pnum)]
                else:
                    print "WARNING: medparams attribute not found in "+ \
                          "posterior object will annotate with "+ \
                          "max-likelihood values and reported uncertainties "+ \
                          "may not be appropriate."
                err = uparams["%s%d" % (print_params[l],pnum)]
                if err > 0:
                    val, err, errlow = radvel.utils.sigfig(val, err)
                    _anotext = '$\\mathregular{%s}$ = %s $\\mathregular{\\pm}$ %s %s' % (labels[l].replace("$",""), val, err, units[p])
                else:
                    _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$",""), val, units[p])

            anotext += [_anotext] 

        anotext = '\n'.join(anotext)
        add_anchored(
            anotext, loc=1, frameon=True, prop=dict(size='large', weight='bold'),
            bbox=dict(ec='none', fc='w', alpha=0.8)
        )

    if saveplot != None:
        pl.savefig(saveplot,dpi=150)
        print "RV multi-panel plot saved to %s" % saveplot
        
    return fig, axL

    
def corner_plot(post, chains, saveplot=None):
    """
    Make a corner plot from the output MCMC chains and a posterior object.

    Args:
        post (radvel.Posterior): Radvel posterior object
        chains (DataFrame): MCMC chains output by radvel.mcmc
        saveplot (str, optional):  Name of output file, will show as 
            interactive matplotlib window if not defined.

    Returns:
        None
    
    """
    labels = [k for k in post.vary.keys() if post.vary[k]]
    texlabels = [post.params.tex_labels().get(l, l) for l in labels]
    
    f = rcParams['font.size']
    rcParams['font.size'] = 12
    
    fig = corner.corner(
        chains[labels], labels=texlabels, label_kwargs={"fontsize": 14},
        plot_datapoints=False, bins=20, quantiles=[.16,.5,.84],
        show_titles = True, title_kwargs={"fontsize": 14}, smooth=True
    )
    
    if saveplot != None:
        pl.savefig(saveplot,dpi=150)
        print "Corner plot saved to %s" % saveplot
    else:
        pl.show()

    rcParams['font.size'] = f

def texlabel(key, letter):
    if key.count('mpsini')==1:
        return '$M_' + letter + '\\sin i$'
    if key.count('rhop')==1:
        return '$\\rho_' + letter + '$'
        
def corner_plot_derived_pars(chains, P, saveplot=None):
    """
    Make a corner plot from the output MCMC chains and a posterior object.

    Args:
        chains (DataFrame): MCMC chains output by radvel.mcmc
        pars (list): 
        saveplot (Optional[string]: Name of output file, will show as 
            interactive matplotlib window if not defined.

    Returns:
        None
    
    """

    # Determine which columns to include in corner plot
    labels = []
    texlabels = []
    for i in np.arange(1, P.nplanets +1, 1):
        letter = P.planet_letters[i]

        for key in 'mpsini rhop'.split():
            label = '{}{}'.format(key,i)
            
            is_column = list(chains.columns).count(label)==1
            if not is_column:
                break
            
            null_column = chains.isnull().any().ix[label]
            if null_column:
                break
                
            labels.append(label)
            texlabels.append(texlabel(label,letter))

    f = rcParams['font.size']
    rcParams['font.size'] = 12
    fig = corner.corner(
        chains[labels], labels=texlabels, label_kwargs={"fontsize": 14}, 
        plot_datapoints=False, bins=20, quantiles=[0.16,0.50,0.84],
        show_titles = True, title_kwargs={"fontsize": 14}, smooth=True
    )
    
    if saveplot != None:
        pl.savefig(saveplot,dpi=150)
        print "Corner plot saved to %s" % saveplot
    else: pl.show()
    rcParams['font.size'] = f
    
def trend_plot(post, chains, nwalkers, outfile=None):
    """MCMC trend plot

    Make a trend plot to show the evolution of the MCMC as a function of step number.

    Args:
        post (radvel.Posterior): Radvel Posterior object
        chains (DataFrame): MCMC chains output by radvel.mcmc
        nwalkers (int): number of walkers used in this particular MCMC run
        outfile (string): name of output multi-page PDF file

    Returns:
        None
        
    """

    labels = sorted([k for k in post.vary.keys() if post.vary[k]])
    texlabels = [post.params.tex_labels().get(l, l) for l in labels]
    colors = [ cmap(x) for x in np.linspace(0.05, 0.95, nwalkers)]

    quantiles = chains.quantile([0.1587, 0.5, 0.8413])

    with PdfPages(outfile) as pdf:
        for param,tex in zip(labels,texlabels):
            flatchain = chains[param].values
            wchain = flatchain.reshape((nwalkers,-1))
        
        
            fig = pl.figure(figsize=(18,10))
            for w in range(nwalkers):
                pl.plot(
                    wchain[w,:], '.', rasterized=True, color=colors[w], 
                    markersize=3
                )

            pl.xlim(0,wchain.shape[1])

            pl.xlabel('Step Number')
            pl.ylabel(tex)

            ax = pl.gca()
            ax.set_rasterized(True)

            pdf.savefig()
            pl.close()

def add_anchored(*args,**kwargs):
    """
    Parameters
    ----------
    s : string
        Text.

    loc : str
        Location code.

    pad : float, optional
        Pad between the text and the frame as fraction of the font
        size.

    borderpad : float, optional
        Pad between the frame and the axes (or *bbox_to_anchor*).

    prop : `matplotlib.font_manager.FontProperties`
        Font properties.
    """

    bbox = {}
    if kwargs.has_key('bbox'):
        bbox = kwargs.pop('bbox')
    at = AnchoredText(*args, **kwargs)
    if len(bbox.keys())>0:
        pl.setp(at.patch,**bbox)

    ax = pl.gca()
    ax.add_artist(at)
