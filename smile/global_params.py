# Third party imports
import numpy as np


# Datatype parameters
scoretype = np.float32


 # Study parameters

VMIN = 0 #minimum possible visual score
SMIN = 0 #minimum possible symptom score
def get_MIN(scorename):
    if scorename == 'visual': return VMIN
    elif scorename in {'symptom', 'symptom_noerror'}: return SMIN
    else: raise ValueError(f"Scorename of {scorename} not known")

NDAYS = 160 #number of days in the study
FIRSTVISIT = 7
LASTVISIT = NDAYS-1 #TODO fix in sampling.py (sometimes used NDAYS instead)

assert(0 <= FIRSTVISIT)
assert(FIRSTVISIT <= LASTVISIT)
assert(LASTVISIT < NDAYS)
assert(all(isinstance(numb, int) for numb in [FIRSTVISIT, LASTVISIT, NDAYS]))


# Sampling parameters (used in SequentialMethodology)
# Temp day values for different cases 
# must be arbitrary distrinct int32s, all larger than NDAYS

_UNREACHED_TRADITIONAL = 2**16
_UNREACHED_SMILE = 2**16+1
_UNREACHED_MAGNITUDE = 2**16+2
_LIMITREACHED = 2**16+3
_ALREADYREACHED = 2**16+4

max_int = np.iinfo(np.array(0, dtype=int).dtype).max
assert(all(NDAYS <= val <= max_int 
           for val in [_UNREACHED_SMILE, _UNREACHED_MAGNITUDE, _LIMITREACHED]))


# Graphing parameters

lines_cmap_name = 'Dark2' #https://matplotlib.org/2.0.1/users/colormaps.html#qualitative
points_cmap_name = 'viridis' #https://matplotlib.org/2.0.1/users/colormaps.html#sequential

#TODO add title colors, hpadding and vpadding 