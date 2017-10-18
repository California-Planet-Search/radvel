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
        self.nsteps = 3000
        self.ensembles = 8

def _standard_run(setupfn):
    """
    Run through all of the standard steps
    """
    
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
    """
    Run through K2-24 example
    """
    
    _standard_run(setupfn)

def test_hd(setupfn='example_planets/HD164922.py'):
    """
    Check multi-instrument fit
    """
    
    args = _args()
    args.setupfn = setupfn

    radvel.driver.fit(args)
    
    args.type = ['rv']
    args.plotkw = {}
    radvel.driver.plots(args)

def test_basis():
    """
    Test basis conversions
    """
    
    basis_list = radvel.basis.BASIS_NAMES
    default_basis = 'per tc e w k'
    
    anybasis_params = radvel.Parameters(1, basis=default_basis)
    
    anybasis_params['per1'] = radvel.Parameter(value=20.885258)
    anybasis_params['tc1'] = radvel.Parameter(value=2072.79438)
    anybasis_params['e1'] = radvel.Parameter(value=0.01)
    anybasis_params['w1'] = radvel.Parameter(value=1.6)
    anybasis_params['k1'] = radvel.Parameter(value=10.0)

    anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
    anybasis_params['curv'] = radvel.Parameter(value=0.0)

    anybasis_params['gamma_j'] = radvel.Parameter(1.0)
    anybasis_params['jit_j'] = radvel.Parameter(value=2.6)
    
    for new_basis in basis_list:
        iparams = radvel.basis._copy_params(anybasis_params)
        if new_basis != default_basis:
            new_params = iparams.basis.to_any_basis(iparams, new_basis)
            tmp = radvel.basis._copy_params(new_params)
            
            old_params = tmp.basis.to_any_basis(tmp, default_basis)

            for par in iparams:
                before = iparams[par].value
                after = old_params[par].value
                assert (before - after) <= 1e-5,\
                    "Parameters do not match after basis conversion: \
{}, {} != {}".format(par, before, after) 

def test_kepler():
    """
    Profile and test C-based Kepler solver
    """
    radvel.kepler.profile()
    
if __name__ == '__main__':
    test_kepler()
