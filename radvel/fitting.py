from scipy import optimize
import numpy as np

def maxlike_fitting(mod, t, vel, errvel):
    print "initial paraeters"
    print mod.params0

    n = len(vel)

    # Maximizing likelihood is equivalent to minimizing negative likelihood
    def obj(*args):
        if np.isnan(args[0]).sum() > 0:
            import pdb;pdb.set_trace()
        return -1.0 * mod.lnprob_array(*args) 


    # If epsilon is too small, there will be no-disernable change in
    # the likelihood
    p0 = mod.params_to_array(mod.params0)
    p1, minneglnlike, outdict = optimize.fmin_l_bfgs_b(
        obj, p0, args=(t, vel, errvel,), approx_grad=1, epsilon=1e-4
        )

    maxlnlike = -1.0 * minneglnlike
    params_maxlike = mod.array_to_params(p1)
    k = mod.nvary_parameters
    print "max-likelihood parameters"
    print params_maxlike
    print "max-likelihood = {}".format( maxlnlike )
    print "n = {}, k = {}, BIC = {}".format( 
        n, k, BIC(maxlnlike, k,  n) 
        )
    return params_maxlike, maxlnlike
