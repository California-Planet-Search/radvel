#!/usr/bin/env python

def restrict(input, inmin, inmax, lims=None):
    import numpy as np
    
    min = inmin
    max = inmax
    n = input.size
    #inf = np.where(ravel(logical_not(finite(in))))[0]
    if lims == None:
        lims = 2
    elif lims == 3:   
        lims = 2
    else:
        nl = lims.size
    data = np.array(input,dtype=float)
    if n == 1:
        data = (input)*np.ones(2)
    range = max - min
    sdata = (data - min) / range
    w = np.where((data >= max) | (data < min))[0]
    nw = w.size
    while nw > 0:
        w = np.where((data >= max) | (data < min))[0]
        nw = w.size
        #if nw > 0: data[w] = data[w] + (np.floor(np.abs(data - min) / range) > 1) * range * (2 * (data[w] < max) - 1)
        data[w] = data[w] + (np.max(np.floor(np.abs(data - min) / range))) * range * (2 * (data[w] < max) - 1)
        #if nw - ninf == 0:  break
   
    return ((n == 1) and [data[0]] or [data])[0]

if __name__ == '__main__':
    import numpy as np
    
    x = np.arange(100,dtype=float) - 40
    y = restrict(x,20,80)
    print x
    print y
