# Standard library imports
import os
from datetime import datetime
import re

# Third party imports
import numpy as np
import dill

# Local application imports
#none

# Settings
seed = 3 # chosen by fair dice roll. guaranteed to be random. https://xkcd.com/221/
np.random.seed(seed)
np.set_printoptions(edgeitems=30, linewidth=100000)
pickle_pops_dir = 'D:\saved_populations_large'
pickle_regs_dir = 'D:\saved_regressions_large'

# Pickling functions
def dump_to_file(obj, filename, filesuffix='.pik', 
                 dirname=None, create_newdir=False, avoid_overwrite=True):
    #build path
    if dirname is not None: 
        if not os.path.isdir(dirname):
            if create_newdir:
                os.makedirs(dirname)
                print(f"Directory {dirname} was created")
            else:
                raise OSError(f"Directory {dirname} doesn't exist and needs to be created")
        filename = os.path.join(dirname, filename+filesuffix)
    else:
        filename = filename+filesuffix
    #check if will overwrite
    if os.path.isfile(filename) and avoid_overwrite:
        raise OSError(f"File {filename} already exists and would be overriden")
    else:
        with open(filename, 'wb') as f:
            dill.dump(obj, f, protocol=4)
def load_from_file(filename):
    with open(filename, 'rb') as f:
        return dill.load(f)


# Definitions

def regress(poplists, name_indexless, index=None):
    #poplists is a 2d array of poplists
    
    if index is not None: 
        suffix='_'+str(index)
        verbose = False
    else: 
        suffix=''
        verbose = True

    #preallocate arrays
    regresultslists = np.empty_like(poplists)

    #filter
    for i, j, k in np.ndindex(poplists.shape):
        if verbose: print(i, j)
        regresultslists[i, j, k] = poplists[i, j, k].regress_mixed()

    #pickle
    dump_to_file(regresultslists, 'regression_'+name_indexless+suffix, 
                 filesuffix='.pik', dirname=pickle_regs_dir)
    if verbose: print("Done regressing.")

#parameters
npersons=1000
npops=1000
slope_options = (1, 2, 3)
error_options = (0.3, 0.5)

#printing
print("Test parameters:")
print(f"npersons={npersons}")
print(f"npops={npops}")
print(f"slope_options={slope_options}")
print(f"error_options={error_options}")
print(f"Total populations: {npersons*npops*len(slope_options)*len(error_options)}")
print()
print("Log: ")

#timing
starttime = datetime.now()
print(f"Started at {starttime.strftime('%H:%M')}.")

try:
    nfiles_per_category = 1
    ncategories = 2 #poster and worddoc
    for i in range(nfiles_per_category):
        poplists = load_from_file(pickle_pops_dir+"\poster_sampled_poplists_"+str(i)+".pik")
        regress(poplists, "poster_sampled_poplists", index=i)
        poplists = load_from_file(pickle_pops_dir+"\worddoc_sampled_poplists_"+str(i)+".pik")
        regress(poplists, "worddoc_sampled_poplists", index=i)
        print(f"Done {ncategories*(i+1)}/{nfiles_per_category*ncategories}")
finally:
    #timing
    endtime = datetime.now()
    deltatime = int((endtime-starttime).total_seconds())
    print(f"Took {deltatime//3600} h {(deltatime%3600)//60} min {deltatime%60} s to run.")
