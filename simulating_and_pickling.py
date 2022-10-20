# Standard library imports
import os
from datetime import datetime

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import dill
from scipy import stats
import pandas as pd

# Local application imports
from smile.population import Population, PopulationList
from smile.sampling import *
from smile import helper
from smile.global_params import *
from smile.global_params import _LIMITREACHED

# Settings
seed = 542705034
np.random.seed(seed)
rng = np.random.default_rng(547263468)
np.set_printoptions(edgeitems=30, linewidth=100000)
pickle_old_dir = r'D:\Work\smile_desk\simulating_16'
pickle_new_dir = r'D:\Work\smile_desk\simulating_17'

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
        raise OSError(f"File {filename} already exists and would be overwritten")
    else:
        with open(filename, 'wb') as f:
            dill.dump(obj, f, protocol=4)
def load_from_file(filename):
    with open(filename, 'rb') as f:
        return dill.load(f)


# Definitions

# To reset between methodologies so all methodologies have same variates
#  while populations still have different variates
beta_rng = helper.Beta_rng(1234, 7, 28, 14, 2.9)
def _synced_delay_func(shape, firstday=7):
    return beta_rng.gen(shape).astype('int')-7+firstday
def get_synced_delay_func(firstday):
    return lambda shape: partial(_synced_delay_func, firstday=firstday)(shape)

def get_traditional_methodology(firstday=7):
    methodology = Methodology('traditonal')
    methodology.add_sampler(TraditionalSampler(day=0, delay=get_synced_delay_func(firstday)))
    methodology.add_sampler(TraditionalSampler(day=('sample', -1), delay=14))
    methodology.add_sampler(TraditionalSampler(day=('sample', -1), delay=14))

    return methodology
def get_realistic_methodology(scorename, firstday=7, delayless=False):
    title_prefix = 'delayless_' if delayless else ''
    methodology = Methodology(f'{title_prefix}realistic_{scorename}')

    first_delay = 0 if delayless else get_synced_delay_func(firstday)
    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=first_delay)
    methodology.add_sampler(sampler1)

    other_delay = 0 if delayless else lambda shape: helper.beta(shape, 0, 14, 4, 3.8).astype('int') #90% at 7
    #if_reached is irrelevant because index is previous sample
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename=scorename,
                            delay=other_delay, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1+get_MIN(scorename), scorename=scorename,
                                delay=other_delay, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology

def get_methodologies(firstday):
    return [
        get_traditional_methodology(firstday),
        get_realistic_methodology('symptom', firstday=firstday),
        get_realistic_methodology('symptom_noerror', firstday=firstday),
        get_realistic_methodology('visual', firstday=firstday)
    ]


# Simulate

#parameters
npersons=1000 #per population
npops=100 #across all files
slope_options = (1,)
error_options = (0.5,)
def options_to_string(slope, error):
    return f"{slope}_{error}".replace(".","")

    
npops_per_sim = 10
nsims = npops // npops_per_sim
#nsims = 1
npops_remainder = npops % npops_per_sim

#produce independent seeds for each call to simulate()
ss = np.random.SeedSequence(874586374) 
seeds = ss.spawn(nsims+1) #at least as many as calls to simulate()

for n in range(nsims):
    suffix='_'+str(n)
    seed = seeds[n]
    beta_rng.reseed(seed)

    #Full
    poplists = load_from_file(pickle_old_dir+"\poplists_"+str(n)+".pik")
    for i,j in np.ndindex(poplists.shape):
        foldername = f"populations_"+options_to_string(slope_options[i], error_options[j])
        poplist = poplists[i,j]

        for filterday in (3, 7, 10, 15):

            def get_filter(scorename):
                return {'filter_type':'ratio_early', 'copy':True, 'firstday':filterday, 
                        'index_day':0, 'recovered_ratio':0.3, 'scorename':scorename}
            scorenames = ('symptom', 'symptom_noerror', 'visual', 'visual_yeserror')
            filters = {scorename: get_filter(scorename) for scorename in scorenames}

            #preallocate
            filteredout = np.ones(npersons*npops_per_sim, dtype=bool)
            df_index = pd.MultiIndex.from_product([range(npops_per_sim), range(npersons)], names=['pop', 'person'])
            df = pd.DataFrame({scorename: filteredout.copy() for scorename in scorenames}, index=df_index)
            dfs = np.empty(poplists.shape, dtype=object)

            #filter
            for i, j in np.ndindex(poplists.shape): #Careful, reuses i,j from outer loop
                dfs[i,j] = df.copy()
                for scorename, filter_kwargs in filters.items():
                    filtered_poplist = poplists[i, j].filter(**filter_kwargs)
                    for k in range(npops_per_sim):
                        persons_valid = filtered_poplist[k].persons.flatten()
                        dfs[i,j][scorename].loc[k,persons_valid] = False #set filteredout to false

            #pickle
            dump_to_file(dfs, f'filteredout_persons{suffix}_{filterday}', dirname=pickle_new_dir, create_newdir=True)


            # Sampling

            methodologies = get_methodologies(filterday)
            #preallocate arrays
            sampled_poplists_shape = (*poplists.shape, len(methodologies))
            sampled_poplists = np.empty(sampled_poplists_shape, dtype=object)

            for k in range(len(methodologies)):
                beta_rng.reset()

                for i, j in np.ndindex(poplists.shape):
                    sampled_poplists[i, j, k] = methodologies[k].sample(poplists[i, j])

            #pickle
            dump_to_file(sampled_poplists, f'sampled_poplists{suffix}_{filterday}', dirname=pickle_new_dir, create_newdir=True)