import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('once', DeprecationWarning)

import radvel
import radvel.driver

class _args(object):
    def __init__(self):
        self.outputdir = '/tmp/'
        self.decorr = False

        self.nwalkers = 50
        self.nsteps = 1000
        self.ensembles = 8

def _standard_run(setupfn):
    args = _args()
    args.setupfn = setupfn

    radvel.driver.fit(args)
    radvel.driver.mcmc(args)
    radvel.driver.derive(args)

    args.type = ['nplanets']
    radvel.driver.bic(args)

    args.type = ['params', 'priors', 'nplanets']
    radvel.driver.tables(args)
    
    args.type = ['rv', 'corner', 'trend', 'derived']
    args.plotkw = {}
    radvel.driver.plots(args)

    args.comptype = 'bic'
    args.latex_compiler = 'pdflatex'
    radvel.driver.report(args)

        
def test_k2(setupfn='example_planets/epic203771098.py'):
    _standard_run(setupfn)

def test_hd(setupfn='example_planets/HD164922.py'):
    _standard_run(setupfn)
    
if __name__ == '__main__':
    test_k2()
