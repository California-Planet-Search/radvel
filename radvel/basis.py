import numpy as np
import pandas as pd
from collections import OrderedDict
from radvel.orbit import timeperi_to_timetrans, timetrans_to_timeperi
import radvel.model

BASIS_NAMES = ['per tp e w k',  # The synth basis
               'per tc secosw sesinw logk',
               'per tc secosw sesinw k',
               'per tc ecosw esinw k',
               'per tc e w k',
               'logper tc secosw sesinw k',
               'logper tc secosw sesinw logk',
               'per tc se w k',
               'logper tp e w logk']

ECCENTRICITY_PARAMS_DICT = {
              'per tp e w k': 'e w',
              'per tc secosw sesinw logk': 'secosw sesinw',
              'per tc secosw sesinw k': 'secosw sesinw',
              'per tc ecosw esinw k': 'ecosw esinw',
              'per tc e w k': 'e w',
              'logper tc secosw sesinw k': 'secosw sesinw',
              'logper tc secosw sesinw logk': 'secosw sesinw',
              'per tc se w k': 'se w',
              'logper tp e w logk': 'e w'}

CIRCULAR_PARAMS_DICT = {
              'per tp e w k': 'per tp k',
              'per tc secosw sesinw logk': 'per tc logk',
              'per tc secosw sesinw k': 'per tc k',
              'per tc ecosw esinw k': 'per tc k',
              'per tc e w k': 'per tc k',
              'logper tc secosw sesinw k': 'logper tc k',
              'logper tc secosw sesinw logk': 'logper tc logk',
              'per tc se w k': 'per tc k',
              'logper tp e w logk': 'logper tp logk'}


def _print_valid_basis():
    print("Available bases:")
    print("\n".join(BASIS_NAMES))


def _copy_params(params_in):
    num_planets = params_in.num_planets
    basis = params_in.basis.name
    planet_letters = params_in.planet_letters
    params_out = radvel.model.Parameters(num_planets, basis=basis,
                                         planet_letters=planet_letters)
    params_out.update(params_in)
    
    return params_out


class Basis(object):
    """
    Object that knows how to convert between the various Keplerian bases

    Args:
        name (str): basis name
        num_planets (int): number of planets

    Attributes:

        synth_params (str): name of synth basis

    Note:
        Valid basis functions: \n
        'per tp e w k' (The synthesis basis) \n
        'per tc secosw sesinw logk'  \n 
        'per tc secosw sesinw k'  \n
        'per tc ecosw esinw k'  \n
        'per tc e w k' \n
        'logper tc secosw sesinw k'\n
        'logper tc secosw sesinw logk'\n
        'per tc se w k'
    """
    synth_params = 'per tp e w k'.split()

    def __init__(self, *args):
        self.name = None
        self.num_planets = 0
        if len(args) == 0:
            _print_valid_basis()
        #    return None
        
        name, num_planets = args

        if BASIS_NAMES.count(name) == 0:
            print("{} not valid basis".format(name))
            _print_valid_basis()
        #    return None

        self.name = name
        self.num_planets = num_planets
        self.params = name.split()

    def __repr__(self):
        return "Basis Object <{}>".format(self.name)

    def to_any_basis(self, params_in, newbasis):
        """Convenience function for converting Parameters object to an arbitraty basis

        Args:
            params_in (radvel.Parameters): radvel.Parameters object expressed in current basis
            newbasis (string): string corresponding to basis to switch into
        Returns:
            radvel.Parameters object expressed in the new basis

        """
        synth_params = self.to_synth(params_in)
        arbbasis_params = self.from_synth(synth_params, newbasis, keep=False)
        return arbbasis_params

    def to_synth(self, params_in, **kwargs):
        """Convert to synth basis

        Convert Parameters object with parameters of a given basis into the
        synth basis

        Args:
            params_in (radvel.Parameters or pandas.DataFrame):  radvel.Parameters object or pandas.Dataframe containing 
                orbital parameters expressed in current basis
            noVary (bool [optional]): if True, set the 'vary' attribute of the returned Parameter objects 
                to '' (used for displaying best fit parameters)

        Returns: 
            Parameters or DataFrame: parameters expressed in the synth basis

        """
        basis_name = kwargs.setdefault('basis_name', self.name)

        if isinstance(params_in, pd.core.frame.DataFrame):
            # Output by emcee
            params_out = params_in.copy()
        else:
            params_out = _copy_params(params_in)

        for num_planet in range(1, 1+self.num_planets):

            def _getpar(key):
                if isinstance(params_in, pd.core.frame.DataFrame):
                    return params_in['{}{}'.format(key, num_planet)]
                else:
                    return params_in['{}{}'.format(key, num_planet)].value

            def _setpar(key, new_value):
                key_name = '{}{}'.format(key, num_planet)

                if isinstance(params_in, pd.core.frame.DataFrame):
                    params_out[key_name] = new_value
                else:
                    if key_name in params_in:
                        local_vary = params_in[key_name].vary
                        local_mcmcscale = params_in[key_name].mcmcscale
                    elif kwargs.get('noVary', True):
                        local_vary = ''
                        local_mcmcscale = None
                    else:
                        local_vary = True
                        local_mcmcscale = None

                    params_out[key_name] = radvel.model.Parameter(value=new_value, 
                                                                  vary=local_vary,
                                                                  mcmcscale=local_mcmcscale)

            # transform into synth basis
            if basis_name == 'per tp e w k':
                # already in the synth basis
                per = _getpar('per')
                tp = _getpar('tp')
                e = _getpar('e')
                w = _getpar('w')   
                k = _getpar('k')
                
            if basis_name == 'per tc e w k':
                per = _getpar('per')
                tc = _getpar('tc')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                tp = timetrans_to_timeperi(tc, per, e, w)

            if basis_name == 'per tc se w k':
                # pull out parameters
                per = _getpar('per')
                tc = _getpar('tc')
                se = _getpar('se')
                w = _getpar('w')
                k = _getpar('k')
                e = se**2
                tp = timetrans_to_timeperi(tc, per, e, w)
    
            if basis_name == 'per tc secosw sesinw logk':
                # pull out parameters
                per = _getpar('per')
                tc = _getpar('tc')
                secosw = _getpar('secosw')
                sesinw = _getpar('sesinw')
                logk = _getpar('logk')

                k = np.exp(logk)
                e = secosw**2 + sesinw**2
                w = np.arctan2(sesinw, secosw)
                tp = timetrans_to_timeperi(tc, per, e, w)

            if basis_name == 'per tc secosw sesinw k':
                # pull out parameters
                per = _getpar('per')
                tc = _getpar('tc')
                secosw = _getpar('secosw')
                sesinw = _getpar('sesinw')
                k = _getpar('k')
            
                # transform into synth basis
                e = secosw**2 + sesinw**2
                w = np.arctan2(sesinw, secosw)
                tp = timetrans_to_timeperi(tc, per, e, w)

            if basis_name == 'logper tc secosw sesinw k':
                # pull out parameters
                logper = _getpar('logper')
                tc = _getpar('tc')
                secosw = _getpar('secosw')
                sesinw = _getpar('sesinw')
                k = _getpar('k')
            
                # transform into synth basis
                per = np.exp(logper)
                e = secosw**2 + sesinw**2
                w = np.arctan2(sesinw, secosw)
                tp = timetrans_to_timeperi(tc, per, e, w)

            if basis_name == 'logper tc secosw sesinw logk':
                # pull out parameters
                logper = _getpar('logper')
                tc = _getpar('tc')
                secosw = _getpar('secosw')
                sesinw = _getpar('sesinw')
                k = _getpar('logk')

                # transform into synth basis
                per = np.exp(logper)
                e = secosw ** 2 + sesinw ** 2
                k = np.exp(k)
                w = np.arctan2(sesinw, secosw)
                tp = timetrans_to_timeperi(tc, per, e, w)

            if basis_name == 'per tc ecosw esinw k':
                # pull out parameters
                per = _getpar('per')
                tc = _getpar('tc')
                ecosw = _getpar('ecosw')
                esinw = _getpar('esinw')
                k = _getpar('k')
            
                # transform into synth basis
                e = np.sqrt(ecosw**2 + esinw**2)
                w = np.arctan2(esinw, ecosw)
                tp = timetrans_to_timeperi(tc, per, e, w)

            if basis_name == 'logper tp e w logk':
                # pull out parameters
                logper = _getpar('logper')
                tp = _getpar('tp')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('logk')

                # transform into synth basis
                per = np.exp(logper)
                k = np.exp(k)
                
            # shoves synth parameters from namespace into param_out
            _setpar('per', per)
            _setpar('tp', tp)
            _setpar('e', e)
            _setpar('w', w)
            _setpar('k', k)

        if isinstance(params_out, radvel.model.Parameters):
            params_out.basis = Basis('per tp e w k', self.num_planets)
        return params_out

    def from_synth(self, params_in, newbasis,  **kwargs):
        """Convert from synth basis into another basis

        Convert instance of Parameters with parameters of a given basis into the synth basis

        Args:
            params_in (radvel.Parameters or pandas.DataFrame):  radvel.Parameters object or pandas.Dataframe containing 
                orbital parameters expressed in current basis
            newbasis (string): string corresponding to basis to switch into
            keep (bool [optional]): keep the parameters expressed in
                the old basis, else remove them from the output
                dictionary/DataFrame

        Returns:
            dict or dataframe with the parameters converted into the new basis
        """
        
        if newbasis not in BASIS_NAMES:
            print("{} not valid basis".format(newbasis))
            _print_valid_basis()
            return None
        
        if isinstance(params_in, pd.core.frame.DataFrame):
            # Output by emcee
            params_out = params_in.copy()
        else:
            params_out = _copy_params(params_in)

        for num_planet in range(1, 1+self.num_planets):

            def _getpar(key):
                if isinstance(params_in, pd.core.frame.DataFrame):
                    return params_in['{}{}'.format(key, num_planet)]
                else:
                    return params_in['{}{}'.format(key, num_planet)].value

            def _setpar(key, new_value):
                key_name = '{}{}'.format(key, num_planet)

                if isinstance(params_in, pd.core.frame.DataFrame):
                    params_out[key_name] = new_value
                else:
                    if key_name in params_in:
                        local_vary = params_in[key_name].vary
                        local_mcmcscale = params_in[key_name].mcmcscale
                    else:
                        local_vary = True
                        local_mcmcscale = None

                    params_out[key_name] = radvel.model.Parameter(value=new_value, 
                                                                  vary=local_vary, 
                                                                  mcmcscale=local_mcmcscale)

            def _delpar(key):
                if isinstance(params_in, OrderedDict):
                    del params_out['{}{}'.format(key, num_planet)]
                elif isinstance(params_in, pd.core.frame.DataFrame):
                    params_out.drop('{}{}'.format(key, num_planet))

            if newbasis == 'per tc e w k':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                tp = _getpar('tp')
                
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))
                _setpar('w', w)

                if not kwargs.get('keep', True):
                    _delpar('tp')

            if newbasis == 'per tc se w k':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                tp = _getpar('tp')
                
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))
                _setpar('w', w)
                _setpar('se', np.sqrt(e))

                if not kwargs.get('keep', True):
                    _delpar('tp')
                    _delpar('e')

            if newbasis == 'per tc secosw sesinw logk':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                if 'tp' in params_in.planet_parameters:
                    tp = _getpar('tp')
                else:
                    tc = _getpar('tc')
                    tp = timetrans_to_timeperi(tc, per, e, w)
                    _setpar('tp', tp)
                    
                _setpar('secosw', np.sqrt(e)*np.cos(w))
                _setpar('sesinw', np.sqrt(e)*np.sin(w))
                _setpar('logk', np.log(k))
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))

                if not kwargs.get('keep', True):
                    _delpar('tp')
                    _delpar('e')
                    _delpar('w')
                    _delpar('k')

                # basis_name = newbasis
                self.params = newbasis.split()
                
            if newbasis == 'per tc secosw sesinw k':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                if 'tp' in params_in.planet_parameters:
                    tp = _getpar('tp')
                else:
                    tc = _getpar('tc')
                    tp = timetrans_to_timeperi(tc, per, e, w)
                    _setpar('tp', tp)
                _setpar('secosw', np.sqrt(e)*np.cos(w))
                _setpar('sesinw', np.sqrt(e)*np.sin(w))
                _setpar('k', k)
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))

                if not kwargs.get('keep', True):
                    _delpar('tp')
                    _delpar('e')
                    _delpar('w')

                self.name = newbasis
                self.params = newbasis.split()

            if newbasis == 'logper tc secosw sesinw k':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                if 'tp' in params_in.planet_parameters:
                    tp = _getpar('tp')
                else:
                    tc = _getpar('tc')
                    tp = timetrans_to_timeperi(tc, per, e, w)
                    _setpar('tp', tp)
                _setpar('logper', np.log(per))
                _setpar('secosw', np.sqrt(e)*np.cos(w))
                _setpar('sesinw', np.sqrt(e)*np.sin(w))
                _setpar('k', k)
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))

                if not kwargs.get('keep', True):
                    _delpar('per')
                    _delpar('tp')
                    _delpar('e')
                    _delpar('w')

                self.name = newbasis
                self.params = newbasis.split()

            if newbasis == 'logper tc secosw sesinw logk':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                if 'tp' in params_in.planet_parameters:
                    tp = _getpar('tp')
                else:
                    tc = _getpar('tc')
                    tp = timetrans_to_timeperi(tc, per, e, w)
                    _setpar('tp', tp)
                _setpar('logper', np.log(per))
                _setpar('secosw', np.sqrt(e)*np.cos(w))
                _setpar('sesinw', np.sqrt(e)*np.sin(w))
                _setpar('logk', np.log(k))
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))

                if not kwargs.get('keep', True):
                    _delpar('per')
                    _delpar('tp')
                    _delpar('e')
                    _delpar('w')
                    _delpar('k')

                self.name = newbasis
                self.params = newbasis.split()

            if newbasis == 'per tc ecosw esinw k':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                if 'tp' in params_in.planet_parameters:
                    tp = _getpar('tp')
                else:
                    tc = _getpar('tc')
                    tp = timetrans_to_timeperi(tc, per, e, w)
                    _setpar('tp', tp)
                _setpar('ecosw', e*np.cos(w))
                _setpar('esinw', e*np.sin(w))
                _setpar('k', k)
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))

                if not kwargs.get('keep', True):
                    _delpar('tp')
                    _delpar('e')
                    _delpar('w')

                self.name = newbasis
                self.params = newbasis.split()

            if newbasis == 'logper tp e w logk':
                per = _getpar('per')
                e = _getpar('e')
                w = _getpar('w')
                k = _getpar('k')
                if 'tp' in params_in.planet_parameters:
                    tp = _getpar('tp')
                else:
                    tc = _getpar('tc')
                    tp = timetrans_to_timeperi(tc, per, e, w)
                    _setpar('tp', tp)

                _setpar('logper', np.log(per))
                _setpar('logk', np.log(k))
                _setpar('tc', timeperi_to_timetrans(tp, per, e, w))

                if not kwargs.get('keep', True):
                    _delpar('per')
                    _delpar('k')

                self.name = newbasis
                self.params = newbasis.split()
                
        params_out.basis = Basis(newbasis, self.num_planets)
                
        return params_out

    def get_eparams(self):
        """Return the eccentricity parameters for the object's basis

        Returns:
            the params which have to do with eccentricity 
        """

        assert BASIS_NAMES.count(self.name) == 1, "Invalid basis"

        eparamstring = ECCENTRICITY_PARAMS_DICT[self.name]
        eparamlist = eparamstring.split()
        assert len(eparamlist) == 2
    
        return eparamlist 

    def get_circparams(self):
        """Return the 3 parameters for a circular orbit of a plent in the object's basis

        Returns:
            the params for a circular orbit 
        """
        assert BASIS_NAMES.count(self.name) == 1, "Invalid basis"

        circparamstring = CIRCULAR_PARAMS_DICT[self.name]
        circparamlist = circparamstring.split()
        assert len(circparamlist) == 3
    
        return circparamlist
