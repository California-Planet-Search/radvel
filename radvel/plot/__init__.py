import numpy as np
from matplotlib import pyplot as pl
import matplotlib
from matplotlib.cm import nipy_spectral
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText

latex = {
    'ms': r'm s$^{\mathregular{-1}}$',
    'BJDTDB': r'BJD$_{\mathregular{TDB}}$'
}

telfmts_default = {
    'j': dict(color='k', marker=u'o', label='HIRES', mew=1),
    'k': dict(color='k', fmt='s', mfc='none', label='HIRES pre 2004', mew=1),
    'a': dict(color='g', fmt='d', label='APF'),
    'pfs': dict(color='magenta', fmt='p', label='PFS'),
    'h': dict(color='firebrick', fmt="s", label='HARPS'),
    'harps-n': dict(color='firebrick', fmt='^', label='HARPS-N'),
    'l': dict(color='g', fmt='*', label='LICK'),
}
telfmts_default['lick'] = telfmts_default['l']
telfmts_default['hires_rj'] = telfmts_default['j']
telfmts_default['hires'] = telfmts_default['j']
telfmts_default['hires_rk'] = telfmts_default['k']
telfmts_default['apf'] = telfmts_default['a']
telfmts_default['harps'] = telfmts_default['h']
telfmts_default['LICK'] = telfmts_default['l']
telfmts_default['HIRES_RJ'] = telfmts_default['j']
telfmts_default['HIRES'] = telfmts_default['j']
telfmts_default['HIRES_RK'] = telfmts_default['k']
telfmts_default['APF'] = telfmts_default['a']
telfmts_default['HARPS'] = telfmts_default['h']
telfmts_default['HARPS-N'] = telfmts_default['harps-n']
telfmts_default['PFS'] = telfmts_default['pfs']


cmap = nipy_spectral
rcParams['font.size'] = 9
rcParams['lines.markersize'] = 5
rcParams['axes.grid'] = False
default_colors = ['orange', 'purple', 'magenta', 'pink', 'green', 'grey', 'red']


def telplot(x, y, e, tel, ax, lw=1., telfmt={}):
    """Plot data from from a single telescope

    x (array): Either time or phase
    y (array): RV
    e (array): RV error
    tel (string): telecsope string key
    ax (matplotlib.axes.Axes): current Axes object
    lw (float): line-width for error bars
    telfmt (dict): dictionary corresponding to kwargs 
        passed to errorbar. Example:

        telfmt = dict(fmt='o',label='HIRES',color='red')
    """

    # Default formatting
    kw = dict(
        fmt='o', capsize=0, mew=0, 
        ecolor='0.6', lw=lw, color='orange',
    )

    # If not explicit format set, look among default formats
    if not telfmt and tel in telfmts_default:
        telfmt = telfmts_default[tel]

    for k in telfmt:
        kw[k] = telfmt[k]

    if not 'label' in kw.keys():
        if tel in telfmts_default:
            kw['label'] = telfmts_default[tel]['label']
        else:
            kw['label'] = tel
        
    pl.errorbar(x, y, yerr=e, **kw)


def mtelplot(x, y, e, tel, ax, lw=1., telfmts={}):
    """
    Overplot data from from multiple telescopes.

    x (array): Either time or phase
    y (array): RV
    e (array): RV error
    tel (array): array of telecsope string keys
    ax (matplotlib.axes.Axes): current Axes object
    telfmts (dict): dictionary of dictionaries corresponding to kwargs 
        passed to errorbar. Example:
        
        telfmts = {
             'hires': dict(fmt='o',label='HIRES'),
             'harps-n' dict(fmt='s')
        }
    """

    utel = np.unique(tel)

    ci = 0
    for t in utel:
        xt = x[tel == t]
        yt = y[tel == t]
        et = e[tel == t]

        telfmt = {}

        if t in telfmts:
            telfmt = telfmts[t]
            if 'color' not in telfmt:
                telfmt['color'] = default_colors[ci]
                ci +=1
        elif t not in telfmts and t not in telfmts_default:
            telfmt = dict(color=default_colors[ci])
            ci +=1 
        else:
            telfmt = {}

        telplot(xt, yt, et, t, ax, lw=1., telfmt=telfmt)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )

def add_anchored(*args, **kwargs):
    """
    Add text at a particular location in the current Axes

    Args:
        s (string): text
        loc (string): location code
        pad (float [optional]): pad between the text and the frame 
            as fraction of the font size
        borderpad (float [optional]): pad between the frame and the axes (or *bbox_to_anchor*)
        prop (matplotlib.font_manager.FontProperties): font properties
    """

    bbox = {}
    if 'bbox' in kwargs:
        bbox = kwargs.pop('bbox')
    at = AnchoredText(*args, **kwargs)
    if len(bbox.keys()) > 0:
        pl.setp(at.patch, **bbox)

    ax = pl.gca()
    ax.add_artist(at)

def labelfig(letter):
    """
    Add a letter in the top left corner in the current Axes

    Args:
        letter (int): integer representation of letter to be printed.
            Ex: ord("a") gives 97, so the input should be 97.
    """
    text = "{})".format(chr(letter))
    add_anchored(
        text, loc=2, prop=dict(fontweight='bold', size='large'),
        frameon=False
    )

