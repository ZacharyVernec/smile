#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
pickle_pops_dir = r'D:\saved_populations_5'
pickle_csv_dir = r'D:\saved_populations_5_csv'




def load_from_file(filename):
    with open(filename, 'rb') as f:
        return dill.load(f)
def dump_to_csv_file(df, filename, filesuffix='.csv', 
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
        raise OSError(f"File {filename} already exists and would be overwritten")
    else:
        df.to_csv(filename)




#parameters
npersons=1000 #per population
npops=1000 #across all files
slope_options = (1, 2, 3)
error_options = (0.3, 0.5)
method_options = ("traditional", "realistic")

def options_to_string(slope, error):
    return f"{slope}_{error}".replace(".","")



poplists = load_from_file(pickle_pops_dir+"\worddoc_poplists_0"+".pik")
for i,j in np.ndindex(poplists.shape):
    foldername = f"realistic_"+options_to_string(slope_options[i], error_options[j])
    poplist = poplists[i,j]
    for l in range(len(poplist)):
        pop = poplist[l]
        dump_to_csv_file(pop.to_dataframe(), foldername+f"_population_{l}", filesuffix='.csv', 
                         dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)


poplists = load_from_file(pickle_pops_dir+"\worddoc_filtered_poplists_0"+".pik")
for i,j in np.ndindex(poplists.shape):
    foldername = f"realistic_"+options_to_string(slope_options[i], error_options[j])
    poplist = poplists[i,j]
    for l in range(len(poplist)):
        pop = poplist[l]
        dump_to_csv_file(pop.to_dataframe(), foldername+f"_population_{l}_filtered", filesuffix='.csv', 
                         dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)


poplists = load_from_file(pickle_pops_dir+"\worddoc_sampled_poplists_0"+".pik")
for i,j, k in np.ndindex(poplists.shape):
    foldername = f"realistic_"+options_to_string(slope_options[i], error_options[j])
    poplist = poplists[i,j,k]
    for l in range(len(poplist)):
        pop = poplist[l]
        dump_to_csv_file(pop.to_dataframe(), foldername+f"_population_{l}_sampled_{method_options[k]}", filesuffix='.csv', 
                         dirname=os.path.join(pickle_csv_dir, foldername), create_newdir=True, avoid_overwrite=True)

