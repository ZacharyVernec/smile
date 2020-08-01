"""Helper functions, independent of local application"""

# Standard library imports
from math import ceil

# Third party imports
import numpy as np

def truncatednormal(xmin, xmax, pmsigma=3, shape=None):
    '''
    the smaller the pmsigma, the closer the distribution is to uniform
    pmsigma corresponds to what would be the z-score of the |xmax| and |xmin| if the distribution was not truncated
    shape of None returns a simple float
    '''
    my_mean = (xmax+xmin)/2
    my_std = (xmax-xmin)/(2*pmsigma)
    my_shape = shape if shape is not None else 1 #convert to array
    
    vals = np.random.normal(my_mean, my_std, my_shape)
    invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(my_mean, my_std, len(invalid))
        invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
        
    #possibly convert back from array
    vals = vals if shape is not None else vals[0]
        
    return vals
def truncatednormal_left(xmin, mean, pmsigma=3, shape=None):
    '''
    the smaller the pmsigma, the closer the distribution is to uniform
    pmsigma corresponds to what would be the z-score of the |xmax| and |xmin| if the distribution was not truncated
    shape of None returns a simple float
    '''
    my_mean = mean
    my_std = (mean-xmin)/pmsigma
    my_shape = shape if shape is not None else 1 #convert to array
    
    vals = np.random.normal(my_mean, my_std, my_shape)
    invalid = np.flatnonzero(vals < xmin)
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(my_mean, my_std, len(invalid))
        invalid = np.flatnonzero(vals < xmin)
        
    #possibly convert back from array
    vals = vals if shape is not None else vals[0]
        
    return vals
def truncatednormal_right(mean, xmax, pmsigma=3, shape=None):
    '''
    the smaller the pmsigma, the closer the distribution is to uniform
    pmsigma corresponds to what would be the z-score of the |xmax| and |xmin| if the distribution was not truncated
    shape of None returns a simple float
    '''
    my_mean = mean
    my_std = (xmax-mean)/pmsigma
    my_shape = shape if shape is not None else 1 #convert to array
    
    vals = np.random.normal(my_mean, my_std, my_shape)
    invalid = np.flatnonzero(vals > xmax)
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(my_mean, my_std, len(invalid))
        invalid = np.flatnonzero(vals > xmax)
        
    #possibly convert back from array
    vals = vals if shape is not None else vals[0]
        
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
    '''Combine multiline strings into one large string that looks like each strings was placed side-by-side'''
    text_blocks_lines = [str(text_block).splitlines() for text_block in text_blocks]
    text_lines_arr = np.array(text_blocks_lines).T
    lines = []
    for line in text_lines_arr:
        lines.append((separator*separatorlen).join(line))
    return "\n".join(lines)
def print_collocated(text_blocks, separator="\t", separatorlen=2):
    '''Print multiline strings side-by-side'''
    print(collocate_text(text_blocks, separator=separator, separatorlen=separatorlen))
#TODO collocate_text is very similar to tile_text, where vseparator="\n" and vseparatorlen=1
def tile_text(text_blocks_2d, hseparator="\t", hseparatorlen=2, vseparator="\n", vseparatorlen=2):
    '''Combine multiline strings into one large string that looks like each strings was placed in a grid'''
    lines_of_blocks = [collocate_text(line_of_blocks, separator=hseparator, separatorlen=hseparatorlen) 
                       for line_of_blocks in text_blocks_2d]
    return (vseparator*vseparatorlen).join(lines_of_blocks)  
def print_tiled(text_blocks_2d, **kwargs):
    '''Print multiline strings in a grid'''
    print(tile_text(text_blocks_2d, **kwargs))
    
def collocate_html(list_of_html, separatorpx=10):
    '''Combine html strings into div blocks arranged side-by-side'''
    html_string = f"<style> .collocationcontainer {{ display: grid; column-gap: {separatorpx}px;}} </style>"
    html_string += '<section class="collocationcontainer">'
    
    for i in range(len(list_of_html)):
        html_string += f'<div style="grid-column: {i+1};">'+list_of_html[i]+'</div>'
        
    html_string += '</section>'
    
    return html_string
def display_collocated(list_of_html, separatorpx=10):
    '''Display html side-by-side'''
    from IPython.core.display import display, HTML
    html_string = collocate_html(list_of_html, separatorpx=separatorpx)
    display(HTML(html_string))


#currently unused, too much trouble to rewrite smile.py (e.g. having population days and scores be twodarrays)
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
                    raise ValueError("tuple and tup_of_classes are of different classes")
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
        return super().__getitem__(subscript) #now slice like an ndarray
    
    #TODO add classmethods from_horizontal(listlike, nrows) and from_vertical(listlike, ncols), 
    #    which would tile/repeat a 1d array to make a towdarray of the required height or depth, respectively
    
def to_vertical(arraylike):
    try:
        return arraylike.reshape(-1, 1)
    except AttributeError:
        return np.array(arraylike).reshape(-1, 1)