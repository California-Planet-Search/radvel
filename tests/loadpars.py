import sqlite3
import rv.model
import pandas as pd 
from k2phot.config import bjd0
from scipy.io.idl import readsav
from k2phot.pdplus import LittleEndian as LE
import pandas as  pd
import rv.mock
from k2phot.config import bjd0
from astropy.time import Time
import kbcUtils
import numpy as np

fn = 'vstepic203771098.dat'
def read_vst(fn):
    d = readsav(fn)
    vst = pd.DataFrame(LE(d['cf3']))
    namemap = dict([(c,c.lower()) for c in vst.columns])
    vst = vst.rename(columns=namemap)
    vst['jd'] = Time(vst['jd'] + 2440000,format='jd').jd
    vst['kjd'] = vst.jd - bjd0


    vst['date'] = Time(vst.jd,format='jd',out_subfmt='date_hms').iso

    print "nobs = %i" % len(vst)
    print "first obs: = %s" % vst.iloc[0]['date']
    print "last obs:  = %s" % vst.iloc[-1]['date']

    return vst

vst = read_vst(fn)

