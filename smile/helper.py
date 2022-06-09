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
#TODO have non-general call the general
def truncatednormal(xmin, xmax, pmsigma=3, shape=None):
    '''
    the smaller the pmsigma, the closer the distribution is to uniform
    pmsigma corresponds to what would be the z-score of the |xmax| and |xmin| if the distribution was not truncated
    shape of None returns a simple float
    '''
    mode = (xmax+xmin)/2
    untruncated_std = (xmax-xmin)/(2*pmsigma)
    arr_shape = shape if shape is not None else 1 #convert to array if needed
    
    vals = np.random.normal(mode, untruncated_std, arr_shape)
    invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(mode, untruncated_std, len(invalid))
        invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
        
    if shape is None: return vals[0] #convert back from array
    else: return vals
def truncatednormal_left(xmin, mode, pmsigma=3, shape=None):
    '''
    careful, the given mode will not be the mean of the truncated distribution sampled from
    the smaller the pmsigma, the closer the distribution is to uniform
    pmsigma corresponds to what would be the z-score of the |xmax| and |xmin| if the distribution was not truncated
    shape of None returns a simple float
    '''
    mode = mode
    untruncated_std = (mean-xmin)/pmsigma
    arr_shape = shape if shape is not None else 1 #convert to array if needed
    
    vals = np.random.normal(mode, untruncated_std, arr_shape)
    invalid = np.flatnonzero(vals < xmin)
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(mode, untruncated_std, len(invalid))
        invalid = np.flatnonzero(vals < xmin)
        
    if shape is None: return vals[0] #convert back from array
    else: return vals
def truncatednormal_right(mode, xmax, pmsigma=3, shape=None):
    '''
    careful, the given mode will not be the mean of the truncated distribution sampled from
    the smaller the pmsigma, the closer the distribution is to uniform
    pmsigma corresponds to what would be the z-score of the |xmax| and |xmin| if the distribution was not truncated
    shape of None returns a simple float
    '''
    mode = mode
    untruncated_std = (mean-xmin)/pmsigma
    arr_shape = shape if shape is not None else 1 #convert to array if needed
    
    vals = np.random.normal(mode, untruncated_std, arr_shape)
    invalid = np.flatnonzero(vals > xmax)
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(mode, untruncated_std, arr_shape)
        invalid = np.flatnonzero(vals > xmax)
           
    if shape is None: return vals[0] #convert back from array
    else: return vals
def truncatednormal_general(xmin, mode, xmax, untruncated_std, shape=None):
    '''
    mode and untruncated_std are the mean and stdev parameters of the pre-truncated distribution
    shape of None returns a simple float
    '''
    arr_shape = shape if shape is not None else 1 #convert to array if needed
    
    vals = np.random.normal(mode, untruncated_std, arr_shape)
    invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
    
    while(len(invalid) > 0):
        vals.flat[invalid] = np.random.normal(mode, untruncated_std, len(invalid))
        invalid = np.flatnonzero((vals < xmin) | (vals >= xmax))
        
    if shape is None: return vals[0] #convert back from array
    else: return vals
    
    
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



#printing and displaying

#subroutines
def _get_line_lengths(arg):
    #only works for string or 2darray with same number of lines per string
    assert(isinstance(arg, (str,np.ndarray)))
    
    if isinstance(arg, str):
        return [len(line) for line in arg.split('\n')]
    
    if isinstance(arg, np.ndarray):
        assert(arg.ndim == 2)
        lengths_as_lists = np.empty(arg.shape, dtype=object)
        for i in range(arg.size):
            lengths_as_lists.flat[i] = _get_line_lengths(arg.flat[i])
        lines_per_string = len(lengths_as_lists[0,0])
        lengths_as_dim = np.empty((*lengths_as_lists.shape, lines_per_string), dtype=int)
        for i,j in np.ndindex(lengths_as_lists.shape):
            lengths_as_dim[i,j,:] = lengths_as_lists[i,j]
        return lengths_as_dim
def _extend_row_to_length(rowlike, length):
    #makes sure a rowlike (list, flat array, or single item) is of right length and is list
    if isinstance(rowlike, np.ndarray):
        row = rowlike.tolist()
    elif not isinstance(rowlike, list):
        row = [rowlike]
    else: #already a list
        row = rowlike
    rowlength = len(row)
    assert(rowlength <= length)
    missinglength = length-rowlength
    return np.hstack([row, np.repeat(np.array(['']),missinglength)])
def _extend_lines_to_length(arg, lengths=None, maxlength=None):
    #takes string or 1darray of strings and makes each line of all items the same length
    #lengths may be given as an n+1dim array
    #maxlength may be given as an int
    if lengths is None:
        lengths = _get_line_lengths(arg)
    if maxlength is None:
        maxlength = np.amax(lengths)
        
    if isinstance(arg, (str, np.str_)):
        lines = arg.split('\n') #also removes '\n' from end of each line
        for i in range(len(lines)):
            lines[i] = lines[i]+' '*(maxlength-lengths[i])
        lines = [line+'\n' for line in lines] #add back line endings, with extra for last line
        block = ''.join(lines)
        return block[:-1] #remove last line ending which wasn't there originally
    if isinstance(arg, np.ndarray) and arg.ndim == 2:
        arg_extended = np.empty_like(arg).tolist() #otherwise there is issues with fixed-length strings in ndarray
        for i,j in np.ndindex(arg.shape):
            arg_extended[i][j] = _extend_lines_to_length(arg[i,j], lengths[i,j], maxlength=maxlength)
        return np.array(arg_extended)
def _to_2darray_of_strings(listarg):
    lengths = []
    for el in listarg:
        if isinstance(el, list):
            lengths.append(len(el))
        elif isinstance(el, np.ndarray):
            assert(el.ndim == 1)
            lengths.append(el.size)
        else: #not flat list-like
            lengths.append(-1)
            
    if max(lengths) == -1: #no element is listlike
        return np.array([listarg]) #keep, as a row ie as array of shape (1,size)
    else: #consider each element as a row of arbitrary length
        rowlength = max(lengths)
        return np.array([_extend_row_to_length(el, rowlength) for el in listarg])
def _get_number_lines(arg):
    #line is counted as number of '\n' + 1
    assert(isinstance(arg, (str,np.ndarray))) #only works for string or 2darray
    if isinstance(arg, str):
        return len(arg.split('\n'))
    elif isinstance(arg, np.ndarray):
        assert(arg.ndim == 2)
        numbers = np.empty_like(arg, dtype=int)
        for i in range(arg.size):
            numbers.flat[i] = _get_number_lines(arg.flat[i])
        return numbers
def _extend_blocks_to_length(arg, numbers=None):
    #makes all items in each row of 2darray arg with have same number of lines
    #numbers is the number of lines for each item, 
    # possibly given so don't have to recalculate
    assert(isinstance(arg, np.ndarray) and arg.ndim == 2)
    if numbers is None: 
        numbers = _get_number_lines(arg)
    assert(arg.shape == numbers.shape)
    
    maxlines = np.amax(numbers, axis=1)
    arg_extended = np.empty_like(arg).tolist() #otherwise there is issues with fixed-length strings in ndarray
    for i,j in np.ndindex(arg.shape):
        arg_extended[i][j] = arg[i,j] + '\n'*(maxlines[i]-numbers[i,j])
    return np.array(arg_extended)
def _collocate_equal_blocks(blocks, hseparator="\t", hseparatorlen=2, vseparator='\n', vseparatorlen=2):
    vseparation = vseparator*vseparatorlen
    hseparation = hseparator*hseparatorlen
    
    collocated_rows = []
    for row in blocks:
        collocated_lines = []
        row_split_lines = [block.split('\n') for block in row]
        row_joined_lines = [hseparation.join(line) for line in zip(*row_split_lines)]
        collocated_row = '\n'.join(row_joined_lines)
        collocated_rows.append(collocated_row)
    
    collocated_blocks = vseparation.join(collocated_rows)
    return collocated_blocks

def collocate_text(strings, hseparator="\t", hseparatorlen=2, vseparator='\n', vseparatorlen=2):
    '''Combine multiline strings into one large string that looks like each strings was placed in a grid'''
    blocks = _to_2darray_of_strings(strings)
    equal_blocks = _extend_lines_to_length(_extend_blocks_to_length(blocks))
    text = _collocate_equal_blocks(equal_blocks, 
                                  hseparator=hseparator, hseparatorlen=hseparatorlen, 
                                  vseparator=vseparator, vseparatorlen=vseparatorlen)
    return text
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