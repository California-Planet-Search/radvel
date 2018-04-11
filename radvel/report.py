import subprocess
import copy
import os
import tempfile
import shutil
from jinja2 import Environment, PackageLoader, select_autoescape

import radvel

env = Environment(loader=PackageLoader('radvel', 'templates'))
print_basis = 'per tc e w k'
units = {
    'per': 'days',
    'tp': 'JD',
    'tc': 'JD',
    'e': '',
    'w': 'radians',
    'k': 'm s$^{-1}$',
    'logk': '$\\ln{(\\rm m\\ s^{-1})}$',
    'secosw': '',
    'sesinw': '',
    'gamma': 'm s$-1$',
    'jitter': 'm s$-1$',
    'logjit': '$\\ln{(\\rm m\\ s^{-1})}$',
    'jit': '$\\rm m\\ s^{-1}$',
    'dvdt': 'm s$^{-1}$ day$^{-1}$',
    'curv': 'm s$^{-1}$ day$^{-2}$',
    'gp_amp': 'm s$-1$',
    'gp_explength': 'days',
    'gp_per': 'days',
    'gp_perlength': ''
}

class RadvelReport():
    """Radvel report

    Class to handle the creation of the radvel summary PDF

    Args:
        planet (planet object): planet configuration object loaded in 
            `kepfit.py` using `imp.load_source` 
        post (radvel.posterior): 
            radvel.posterior object containing the best-fit parameters in 
                post.params 
        chains (DataFrame): output DataFrame from a `radvel.mcmc` run
    """
    
    def __init__(self, planet, post, chains, compstats=None):
        self.planet = planet
        self.post = post
        self.starname = planet.starname
        self.starname_tex = planet.starname.replace('_', '\\_')
        self.runname = self.starname_tex
                
        post.params = post.params.basis.to_synth(post.params)
        post.params = post.params.basis.from_synth(
            post.params, print_basis
        )
        self.latex_dict = post.params.tex_labels()

        for p in post.params.keys():
            if p not in chains.columns:
                chains[p] = post.params[p].value

        self.chains = post.params.basis.to_synth(
            chains, basis_name=planet.fitting_basis
        )
        self.chains = post.params.basis.from_synth(
            self.chains, print_basis
        )
        self.quantiles = self.chains.quantile([0.159, 0.5, 0.841])
        self.compstats = compstats
        self.num_planets = self.post.params.num_planets 

    def texdoc(self):
        """TeX for entire document

        Returns:
            str: TeX code for report
        """
        reportkw = {}
        reportkw['version'] = radvel.__version__

        # Render TeX for figures
        figtypes = ['rv_multipanel','corner','corner_derived_pars']
        for figtype in figtypes:
            infile = "{}_{}.pdf".format(self.runname,figtype)
            tmpfile = 'fig_{}.tex'.format(figtype)
            key = 'fig_{}'.format(figtype)
            if os.path.exists(infile):
                t = env.get_template(tmpfile)
                reportkw[key] = t.render(report=self,infile=infile)

        # Render TeX for tables
        textable = TexTable(self)
        reportkw['tab_rv'] = textable.tab_rv()
        reportkw['tab_params'] = textable.tab_params()
        reportkw['tab_prior_summary'] = textable.tab_prior_summary()

        if self.compstats is not None:
            reportkw['tab_comparison'] = textable.tab_comparison()

        t = env.get_template('report.tex')
        out = t.render(report=self, **reportkw)
        return out

    def compile(self, pdfname, latex_compiler='pdflatex', depfiles=[]):
        """Compile radvel report

        Compile the radvel report from a string containing TeX code
        and save the resulting PDF to a file.

        Args:
            pdfname (str): name of the output PDF file
            latex_compiler (str): path to latex compiler
            depfiles (list): list of file names of dependencies needed for 
                LaTex compilation (e.g. figure files)

        """
        texname = os.path.basename(pdfname).split('.')[0] + '.tex'
        current = os.getcwd()
        temp = tempfile.mkdtemp()
        for fname in depfiles:
            shutil.copy2(
                os.path.join(current, fname), os.path.join(temp, fname)
            )
        
        os.chdir(temp)

        f = open(texname, 'w')
        f.write(self.texdoc())
        f.close()

        shutil.copy(texname, current)

        try:
            for i in range(3):
                # LaTex likes to be compiled a few times
                # to get the table widths correct
                proc = subprocess.Popen(
                    [latex_compiler, texname], stdout=subprocess.PIPE, 
                )
                proc.communicate()  # Let the subprocess complete
        except (OSError):
            msg = """ 
WARNING: REPORT: could not run %s. Ensure that %s is in your PATH
or pass in the path as an argument
""" % (latex_compiler, latex_compiler)
            print(msg)
            return

        shutil.copy(pdfname, current)

        shutil.rmtree(temp)
        os.chdir(current)

class TexTable(RadvelReport):
    """LaTeX table

    Class to handle generation of the LaTeX tables within the summary PDF.

    Args:
        report (radvel.report.RadvelReport): radvel report object
    """
    
    def __init__(self, report):
        self.report = report
        self.post = report.post
        self.quantiles = report.quantiles
        self.fitting_basis = report.post.params.basis.name
    
    def _row(self, param, unit):
        """
        Helper function to output the rows in the parameter table
        """
        if unit == 'radians':
            par = radvel.utils.geterr(self.report.chains[param], angular=True)
            med, low, high = par 
        else:
            med = self.quantiles[param][0.5]
            low = self.quantiles[param][0.5] - self.quantiles[param][0.159]
            high = self.quantiles[param][0.841] - self.quantiles[param][0.5]

        maxlike = self.post.maxparams[param]
        tex = self.report.latex_dict[param]

        low = radvel.utils.round_sig(low)
        high = radvel.utils.round_sig(high)
        maxlike, errlow, errhigh = radvel.utils.sigfig(maxlike, low, high)
        med, errlow, errhigh = radvel.utils.sigfig(med, low, high)

        if min(errlow,errhigh) <= 1e-12:
            med = maxlike = r"$\equiv$ %s" % round(self.quantiles[param][0.5],4)
            errfmt = ''
        else:
            if errhigh == errlow: errfmt = '$\pm %s$' % (errhigh)    
            else: errfmt = '$^{+%s}_{-%s}$' % (errhigh,errlow)

        row = "%s & %s %s & %s & %s" % (tex,med,errfmt,maxlike,unit)
        return row

    def _data(self, basis):
        """
        Helper function to output the rows in the parameter table
        """
        suffixes = ['_'+j for j in self.report.post.likelihood.suffixes]
        rows = []
        for n in range(1,self.report.planet.nplanets+1):
            for p in basis.split(): # loop over variables
                unit = units.get(p, '')
                if unit == '':
                    for s in suffixes:
                        if s in p:
                            unit = units.get(p.replace(s, ''), '')
                            break
                        
                par = p+str(int(n))
                try:
                    row = self._row(par, unit)
                except KeyError:
                    row = self._row(p, unit)
                rows.append(row)
                
        return rows

    def tab_prior_summary(self):
        """Summary of priors
        """
        texdict = self.post.likelihood.params.tex_labels()
        prior_list = self.post.priors
        rows = []
        for prior in prior_list:
            row = prior.__str__()
            if not row.endswith("\\\\"):
                row = row + "\\\\"
            rows.append(row)

        tmpfile = 'tab_prior_summary.tex'
        t = env.get_template(tmpfile)
        out = t.render(rows=rows)
        return out

    def tab_rv(self):
        """Table of input velocities
        """

        nvels = len(self.post.likelihood.x)
        rows = []
        for i in range(nvels):
            t = self.post.likelihood.x[i]
            v = self.post.likelihood.y[i]
            e = self.post.likelihood.yerr[i]
            inst = self.post.likelihood.telvec[i]
            row = "{:.5f} & {:.2f} & {:.2f} & {:s}".format(t, v, e, inst)
            rows.append(row)

        tmpfile = 'tab_rv.tex'
        t = env.get_template(tmpfile)
        out = t.render(rows=rows)
        return out

    def tab_params(self):
        # Sort extra params
        ep = []
        order = ['gamma', 'dvdt', 'curv', 'jit']
        for o in order:
            op = []
            for p in self.post.likelihood.extra_params:
                if o in p:
                    op.append(p)
            if len(op)==0: 
                op = [o]
            [ep.append(i) for i in sorted(op)[::-1]]
        ep = ' '.join(ep)
        kw = {}
        kw['fitting_basis_rows'] = self._data(self.fitting_basis)
        kw['print_basis_rows'] = self._data(print_basis)
        kw['ep_rows'] = self._data(ep)
        kw['nlinks'] = len(self.report.chains)
        kw['time_base'] = self.report.post.likelihood.model.time_base
        tmpfile = 'tab_params.tex'
        t = env.get_template(tmpfile)
        out = t.render(**kw)
        return out

    def tab_comparison(self):
        """Model comparisons
        """
        statsdict = self.report.compstats

        if statsdict is None:
            return ""

        n_test = len(statsdict)
        coldefs = r"\begin{deluxetable*}{%s}" % ('l'+'r'*n_test)
        head = r"\tablehead{\colhead{Statistic}"
        for n in range(n_test):
            if n == n_test-1:
                head += r" & \colhead{{\bf %d planets (adopted)}}" % n

            else:
                head += r" & \colhead{%d planets}" % n
        head += "}"

        rows = []
        statkeys = statsdict[0].keys()
        for s in statkeys:
            row = "%s (%s) " % (s, statsdict[0][s][1])
            for n in range(n_test):
                row += " & %s" % statsdict[n][s][0]
            rows.append(row)

        t = env.get_template('tab_comparison.tex')
        out = t.render(coldefs=coldefs, head=head, rows=rows)
        return out
