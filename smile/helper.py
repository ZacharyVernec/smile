"""Helper functions, independent of local application"""

# Standard library imports
from math import ceil
import warnings

# Third party imports
import numpy as np
from scipy import stats


#warnings
def warn(message):
    '''Standard UserWarning but without showing extra information (i.e. exclude filename, lineno, line)'''
    default_formatwarning = warnings.formatwarning
    def custom_formatwarning(msg, category, filename, lineno, line=None): 
        return default_formatwarning(msg, category, filename='', lineno='', line='')
    warnings.formatwarning = custom_formatwarning
    warnings.warn(message)
    warnings.formatwarning = default_formatwarning

    
#distributions
class BoundedNormal:
    '''Creates frozen instance of truncnorm but with bounds independent of loc and scale'''
    def __new__(cls, lower, upper, loc=0, scale=1):
      if np.any(np.atleast_1d(lower) > np.atleast_1d(upper)):
        raise ValueError()
      a = (lower - loc) / scale
      b = (upper - loc) / scale
      return stats.truncnorm(a, b, loc=loc, scale=scale)

class Mixture:
    """Mixture of BoundedNormal, partially imitates stats.rv_continuous"""
    def __init__(self, lower, upper, mix, locs, scales):
        if not self._argcheck(mix, locs, scales):
            raise ValueError("bad parameters")
        self.mix = np.array([*np.atleast_1d(mix), 1-np.sum(mix)])
        self.distribs = [BoundedNormal(lower, upper, loc=loc, scale=scale) for loc, scale in zip(locs, scales)]
        
    def _argcheck(self, mix, locs, scales):
        mix = np.atleast_1d(mix)
        dims_ok = (mix.ndim == 1) and (len(mix)+1 == len(locs) == len(scales))
        mix_ok = np.all(mix >= 0) and np.sum(mix) <= 1
        locs_ok = np.all(np.isfinite(locs))
        scales_ok = np.all(scales > 0) and np.all(np.isfinite(scales))
        return dims_ok and mix_ok and locs_ok and scales_ok
    
    def rvs(self, size=1, random_state=None):
        #flatten size but store as 'shape' for returning reshaped
        shape = size
        size = np.prod(shape)
        
        indices = stats.rv_discrete(values=(range(len(self.mix)), self.mix)).rvs(size=size, random_state=random_state)
        norm_variates = [distrib.rvs(size=size, random_state=random_state) for distrib in self.distribs]
        return np.choose(indices, norm_variates).reshape(shape)
        
    def pdf(self, x):
        return np.average([distrib.pdf(x) for distrib in self.distribs], axis=0, weights=self.mix)
    def cdf(self, x):
        return np.average([distrib.cdf(x) for distrib in self.distribs], axis=0, weights=self.mix)
    def sf(self, x):
        return np.average([distrib.sf(x) for distrib in self.distribs], axis=0, weights=self.mix)

def beta(shape=1, left_bound=0, interval_length=1, mode=0.5, a=1):
        '''
        Beta distribution parametrized by location 'left_bound', scale 'interval_length', 'mode', and 'a'
        graphed here: https://www.desmos.com/calculator/qnydwobgwp
        '''
        #TODO warn if mode not in [left_bound, left_bound+interval_length]
        #TODO allow parametrization with max rather than length
        
        # transformation x -> x' and inverse transformation x' -> x
        transform_x = lambda x: (x-left_bound)/interval_length #from unit interval to range
        untransform_xprime = lambda xprime: xprime*interval_length+left_bound #from range to unit interval
        
        # shape parameters
        scaled_mode = transform_x(mode)
        b = (1/scaled_mode-1)*a + 2-1/scaled_mode
        
        values_unitinterval = stats.beta.rvs(a, b, size=shape)
        
        return untransform_xprime(values_unitinterval)

# class to make variates streamable to be restarted
class Beta_rng:
    def __init__(self, seed, left_bound=0, interval_length=1, mode=0.5, a=1):
        self.left_bound=left_bound
        self.interval_length=interval_length
        self.mode = mode
        self.a = a
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
    def reseed(self, seed):
        self.seed = seed
        self.reset()

    def __transform_x__(self, x):
        return (x-self.left_bound)/self.interval_length
    def __untransform_xprime__(self, xprime):
        return xprime*self.interval_length+self.left_bound
    def gen(self, shape=1):
        scaled_mode = self.__transform_x__(self.mode)
        b = (1/scaled_mode-1)*self.a + 2 - 1/scaled_mode
        values_unitinterval = stats.beta.rvs(self.a, b, size=shape, random_state=self.rng)
        return self.__untransform_xprime__(values_unitinterval)

    
#matplotlib colors
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



#numpy
#currently mostly unused, too much trouble to rewrite smile.py (e.g. having population days and scores be twodarrays)
class twodarray(np.ndarray):
    '''numpy ndarray that always stays two-dimensional when sliced
       and that raises errors when any other method is used that would make it not 2d'''
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        assert(obj.ndim == 2)
        # Finally, we must return the newly created object:
        return obj
    def __array_finalize__(self, obj):
        #called whenever a twodarray would be returned
        if obj is None: #if in the middle of __new__()
            return #can't check ndim, will be checked at the end of __new__()
        else:
            assert(self.ndim == 2)
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''uses ufunc as a regular ndarray
           converts back to twodarray iff ndim=2'''
        def _replace_self(a):
            if a is self:
                return a.view(np.ndarray)
            else:
                return a
        
        inputs = tuple(_replace_self(inputarr) for inputarr in inputs)
        if 'out' in kwargs:
            kwargs['out'] = tuple(_replace_self(outputarr) for outputarr in outputarrs)
        res = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(res, np.ndarray) and res.ndim == 2:
            return twodarray(res)
        else: 
            return res
        
    def __getitem__(self, subscript):
        '''will keep ndim == 2
        supports:
            int
            slice
            ndarray
            (int, int)
            (int, slice)
            (int, np.ndarray)
            (slice, int)
            (slice, slice)
            (slice, np.ndarray)
            (np.ndarray, int)
            (np.ndarray, slice)
        doesn't support boolean fancy indexing
        doesn't support indexing by (rowcoords, colcoords)
        '''
        if isinstance(subscript, int): #if only an int
            subscript = (subscript, np.newaxis) 
        elif isinstance(subscript, slice):
            pass #no changes necessary
        elif isinstance(subscript, np.ndarray):
            pass #no changes nevessary
        elif isinstance(subscript, tuple):
            def areinstances(tup, list_of_classes):
                if len(tup) != len(list_of_classes):
                    raise ValueError("tup and tup_of_classes are of different lengths")
                for i in range(len(tup)): 
                    if not isinstance(tup[i], list_of_classes[i]): return False
                return True
            if areinstances(subscript, [int, int]):
                subscript = (np.newaxis, np.newaxis, *subscript)
            elif areinstances(subscript, [int, slice]):
                subscript = (np.newaxis, *subscript)
            elif areinstances(subscript, [int, np.ndarray]):
                subscript = (np.newaxis, *subscript)
            elif areinstances(subscript, [slice, int]):
                subscript = (*subscript, np.newaxis)
            elif areinstances(subscript, [slice, slice]):
                pass #no changes necessary
            elif areinstances(subscript, [slice, np.ndarray]):
                pass #no changes necessary
            elif areinstances(subscript, [np.ndarray, int]):
                subscript = (*subscript, np.newaxis)
            elif areinstances(subscript, [np.ndarray, slice]):
                pass #no changes necessary
            else:
                raise ValueError("Unknown subscript: {} with types {}".format(subscript, [type(el) for el in subscript]))
        else:
            raise ValueError("Unknown subscript: {} with types {}".format(subscript, [type(el) for el in subscript]))
        return super().__getitem__(subscript) #now slice like an ndarra
    #TODO add classmethods from_horizontal(listlike, nrows) and from_vertical(listlike, ncols), 
    #    which would tile/repeat a 1d array to make a towdarray of the required height or depth, respectively
    #TODO add classmethod .slicer() which would use __getitem__ as a twodarray but would return a regular ndarray for compatibility
    
def to_vertical(arraylike):
    try:
        return arraylike.reshape(-1, 1)
    except AttributeError:
        return np.array(arraylike).reshape(-1, 1)