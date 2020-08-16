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
    careful, the given mean is the mean of the pre-truncated distribution, and won't stay the mean
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
    careful, the given mean is the mean of the pre-truncated distribution, and won't stay the mean
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


def collocate_text(text_blocks, hseparator="\t", hseparatorlen=2, vseparator='\n', vseparatorlen=2):
    '''Combine multiline strings into one large string that looks like each strings was placed in a grid'''
    
    #make 2d if not already
    text_blocks = np.array(text_blocks)
    if text_blocks.ndim == 1:
        text_blocks = text_blocks.reshape(1, -1)
        
    #collocate each row
    row_strings = []
    for row in range(text_blocks.shape[0]):
        text_blocks_lines = [str(text_block).splitlines() for text_block in text_blocks[row]]
        text_lines_arr = np.array(text_blocks_lines).T
        lines = []
        for line in text_lines_arr:
            lines.append((hseparator*hseparatorlen).join(line))
        row_strings.append("\n".join(lines))
    
    #stack rows
    return (vseparator*vseparatorlen).join(row_strings) 
def print_collocated(text_blocks, **kwargs):
    '''Display multiline strings in a grid'''
    print(collocate_text(text_blocks, **kwargs))
    
def collocate_html(html_elements, hseparatorpx=10, vseparatorpx=10):
    '''Combine html strings into div blocks arranged in a grid'''
    
    #make 2d if not already
    html_elements = np.array(html_elements)
    if html_elements.ndim == 1:
        html_elements = html_elements.reshape(1, -1)
        
    html_string = f"<style> .collocationcontainer {{ display: grid; column-gap: {hseparatorpx}px; row-gap: {hseparatorpx}px;}} </style>"
    html_string += '<section class="collocationcontainer">'
    
    for row in range(html_elements.shape[0]):
        for col in range(html_elements.shape[1]):
            html_string += f'<div style="grid-row: {row+1}; grid-column: {col+1};">'+html_elements[row, col]+'</div>'
        
    html_string += '</section>'
    
    return html_string
def display_collocated(html_elements, **kwargs):
    '''Display html side-by-side'''
    from IPython.core.display import display, HTML
    html_string = collocate_html(html_elements, **kwargs)
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