"""Helper functions"""

# Standard library imports
from math import ceil

# Third party imports
import numpy as np


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
        
    def __getitem__(self, subscript):
        '''will keep ndim == 2'''
        if isinstance(subscript, int): #if only an int
            subscript = (subscript, np.newaxis) 
        elif isinstance(subscript, slice):
            pass #no changes necessary
        #else subscript is a tuple
        elif isinstance(subscript[0], int) and isinstance(subscript[1], int):
            subscript = (*subscript, np.newaxis, np.newaxis)
        elif isinstance(subscript[0], int) and isinstance(subscript[1], slice):
            subscript = (np.newaxis, *subscript)
        elif isinstance(subscript[0], slice) and isinstance(subscript[1], int):
            subscript = (*subscript, np.newaxis)
        elif isisnstance(sbuscript[0], slice) and isinstance(subscript[1], slice):
            pass #no changes necessary
        else:
            raise ValueError("Unknown subscript: {} with types {}".format(subscript, [type(el) for el in subscript]))
        return super().__getitem__(subscript) #now slice like an ndarray
    
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


def collocate_text(text_blocks, separator="\t", separatorlen=2):
    text_blocks_lines = [str(text_block).splitlines() for text_block in text_blocks]
    text_lines_arr = np.array(text_blocks_lines).T
    lines = []
    for line in text_lines_arr:
        lines.append((separator*separatorlen).join(line))
    return "\n".join(lines)

def print_collocated(text_blocks, separator="\t", separatorlen=2):
    print(collocate_text(text_blocks, separator=separator, separatorlen=separatorlen))
    
#TODO collocate_text is very similar to tile_text, where vseparator="\n" and vseparatorlen=1

def tile_text(text_blocks_2d, hseparator="\t", hseparatorlen=2, vseparator="\n", vseparatorlen=2):
    lines_of_blocks = [collocate_text(line_of_blocks, separator=hseparator, separatorlen=hseparatorlen) 
                       for line_of_blocks in text_blocks_2d]
    return (vseparator*vseparatorlen).join(lines_of_blocks)
          
def print_tiled(text_blocks_2d, **kwargs):
    print(tile_text(text_blocks_2d, **kwargs))