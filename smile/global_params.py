# Study parameters

VMIN = 6 #minimum possible visual score
SMIN = 0 #minimum possible symptom score

NDAYS = 160 #number of days in the study
FIRSTVISIT = 7
LASTVISIT = NDAYS-1

assert(0 <= FIRSTVISIT)
assert(FIRSTVISIT <= LASTVISIT)
assert(LASTVISIT < NDAYS)
assert(all(isinstance(numb, int) for numb in [FIRSTVISIT, LASTVISIT, NDAYS]))

# Graphing parameters

lines_cmap_name = 'Dark2' #https://matplotlib.org/2.0.1/users/colormaps.html#qualitative
points_cmap_name = 'viridis' #https://matplotlib.org/2.0.1/users/colormaps.html#sequential

#TODO add title colors, hpadding and vpadding 