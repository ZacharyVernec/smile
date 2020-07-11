# Helper functions

import numpy as np

def truncatednormal(xmin, xmax, pmsigma=3, shape=(2,4)):
    '''the smaller the pmsigma, the closer the distribution is to uniform'''
    my_mean = (xmax+xmin)/2
    my_std = (xmax-xmin)/(2*pmsigma)
    
    vals = np.random.normal(my_mean, my_std, shape)
    invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(my_mean, my_std, len(invalid))
        invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
        
    return vals

def rgblist_to_rgbapop(rgblist, npersons, ndays, opacity=1.0):
    '''
    takes a list of rgb colors with shape=(len,3) and turns it into an array of rgba colors with shape=(npersons, ndays, 4)
    where values change row-by-row (different persons) but are same across different columns (different days)
    '''
    listcopies = ceil(npersons/len(rgblist))
    rgblist_extended = np.tile(rgblist, (listcopies,1))[:npersons]
    rgbpop = np.tile(rgblist_extended, (1, 1, ndays)).reshape(npersons, ndays, 3)
    opacitypop = np.full((npersons, ndays, 1), opacity)
    rgbapop = np.concatenate([rgbpop, opacitypop], axis=2)
    return rgbapop

def normalize(array):
    return (array - np.min(array))/(np.max(array)-np.min(array))