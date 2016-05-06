import subprocess
import copy
import numpy as np

import radvel

print_basis = 'per tc e w k'
planet_pars = ['per', 'tc', 'e', 'w', 'k']
inst_pars = ['gamma', 'jitter']
system_pars = ['dvdt', 'curv']
units = {'per': 'days',
         'tp': 'JD',
         'tc': 'JD',
         'e': '',
         'w': 'radians',
         'k': 'm s$^{-1}$',
         'logk': '$\\ln{m s^{-1}}$',
         'secosw': '',
         'sesinw': '',
         'gamma': 'm s$-1$',
         'jitter': 'm s$-1$',
         'dvdt': 'm s$-1$ day$^{-1}$',
         'curv': 'm s$-1$ day$^{-2}$'}


class RadvelReport():
    """Radvel report

    Class to handle the creation of the radvel summary PDF

    Args:
        planet (planet object): planet configuration object loaded in `kepfit.py` using `imp.load_source`
        post (radvel.posterior): radvel.posterior object containing the best-fit parameters in post.params
        chains (DataFrame): output DataFrame from a `radvel.mcmc` run
    """
    
    def __init__(self, planet, post, chains):
        self.planet = planet
        self.post = post
        self.chains = chains
        self.quantiles = chains.quantile([0.159, 0.5, 0.841])
        
        self.params = post.params
        self.latex_dict = self.params.tex_labels()
        self.starname = planet.starname
        
        printpost = copy.deepcopy(post)
        printpost.params = printpost.params.basis.to_cps(printpost.params)
        printpost.params = printpost.params.basis.from_cps(printpost.params, print_basis)


class TexTable(RadvelReport):
    """LaTeX table

    Class to handle generation of the LaTeX tables within the summary PDF.

    Args:
        report (radvel.report.RadvelReport): radvel report object
    """
    
    def __init__(self, report):
        self.report = report
        self.quantiles = report.quantiles
        self.fitting_basis = report.post.params.basis.name

    def preamble(self):
        return """
\\documentclass{emulateapj}
\\usepackage{graphicx,textcomp}
\\begin{document}"""

    def header(self):
        fstr = """
\\begin{deluxetable}{lrr}
\\tablecaption{%s Results}""" % self.report.starname
        fstr += """
\\tablehead{\\colhead{Parameter} & \\colhead{Value} & \\colhead{Units}}"""
        fstr += """
\\startdata"""
        return fstr

    def footer(self):
        fstr = """
    \\enddata
    \\tablenotetext{}{%d links saved}
    \\tablenotetext{}{Reference epoch for $\\gamma$,$\\dot{\\gamma}$,$\\ddot{\\gamma}$: %15.1f}
    \\end{deluxetable}
    """ % (len(self.report.chains),self.report.post.model.time_base)
        return fstr
    
    def row(self, param, unit):
        med = self.quantiles[param][0.5]
        low = self.quantiles[param][0.5] - self.quantiles[param][0.159]
        high = self.quantiles[param][0.841] - self.quantiles[param][0.5]

        tex = self.report.latex_dict[param]

        low = radvel.utils.round_sig(low)
        high = radvel.utils.round_sig(high)
        med, errlow, errhigh = radvel.utils.sigfig(med, low, high)

        if errhigh == errlow: errfmt = '$\pm %s$' % (errhigh)    
        else: errfmt = '$^{+%s}_{-%s}$' % (errhigh,errlow)

        row = "%s & %s %s & %s\\\\\n" % (tex,med,errfmt,unit)

        return row

    
    def data(self):
        tabledat = self.header()
        for n in range(1,self.report.planet.nplanets+1):
            for p in self.fitting_basis.split():
                unit = units.get(p, '')
                par = p+str(int(n))
                tabledat += self.row(par, unit)
        print tabledat

    def texdoc(self):
        return self.preamble() + self.header()
