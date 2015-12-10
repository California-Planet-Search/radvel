class ObservedRV(object):
    """
    Simple container object for RV data.
    t : time
    rv : radial velocity
    erv : error on rv
    """
    def __init__(self, t, rv, erv):
        self.t = t
        self.rv = rv
        self.erv = erv 
        self.nobs = len(t)
