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
pickle_dir = r'D:\Work\smile_desk\simulating_16'

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
#populations
def get_populations(slope_option, error_option, npersons=100, npops=100):
    
    # Define and set visual score function
    pop = Population(npersons, title=f'slope {slope_option} and error {error_option}')
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop.set_score_generator('visual', gen_visualscores)
    #parameter generators
    #Initial V_0 scaled to visualscore
    symptom_values = np.arange(20+1) #from symptom reference data
    counts = np.array([0, 34, 38, 26, 32, 30, 12, 14, 12, 12, 12, 7, 8, 5, 5, 5, 3, 1, 0, 3, 3], dtype=int)
    symptom_values, counts = symptom_values[2:], counts[2:] #exclude 0 and 1 values as already healthy
    visual_values = symptom_values / slope_option + VMIN #Since reference data is a measure of symptoms, apply inverse gen_symptomscores
    probs = counts / np.sum(counts)
    V_0 = helper.Discrete(values=(visual_values, probs))
    #Together give recovery tails of (0.29999967680454737, 0.09999995806927865)

    gen_r = lambda shape: 0.5014 / slope_option #Based on empirical mean of Mixture in previous version
    gen_v0 = lambda shape: V_0.rvs(size=shape, random_state=rng)
    pop.set_parameter_generator('r', gen_r, 'population')
    pop.set_parameter_generator('v0', gen_v0, 'person')
    
    #Define and set symptom score functions
    gen_symptomscores = lambda v,a: np.minimum(a*(v-VMIN), 20)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: slope_option
    pop.set_parameter_generator('a', gen_a, 'population')
    
    # Define and set error functions
    #Symptom
    pop.set_score_generator('symptom', lambda s,C: np.minimum(s*C, 20))
    gen_C_mul = lambda shape: np.random.uniform(1-error_option, 1+error_option, shape)
    pop.set_parameter_generator('C', gen_C_mul, 'day')
    #Visual (same error parameter)
    pop.set_score_generator('visual_yeserror', lambda v,C: np.maximum(v*C, VMIN))

    # Repeat
    pops = PopulationList.full(npops, pop)
    
    # Return
    return pops


#simulations

# To reset between methodologies so all methodologies have same variates
#  while populations still have different variates
beta_rng = helper.Beta_rng(1234, 7, 28, 14, 2.9)
def synced_delay_func(shape):
    return beta_rng.gen(shape).astype('int')

def get_traditional_methodology():
    methodology = Methodology('traditonal')
    methodology.add_sampler(TraditionalSampler(day=0, delay=synced_delay_func))
    methodology.add_sampler(TraditionalSampler(day=('sample', -1), delay=14))
    methodology.add_sampler(TraditionalSampler(day=('sample', -1), delay=14))

    return methodology
def get_realistic_methodology(scorename, delayless=False):
    title_prefix = 'delayless_' if delayless else ''
    methodology = Methodology(f'{title_prefix}realistic_{scorename}')

    first_delay = 0 if delayless else synced_delay_func
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


def simulate(npops, npersons, slope_options, error_options, index=None, seed=1234):
    if index is not None: 
        suffix='_'+str(index)
        verbose = False
    else: 
        suffix=''
        verbose = True

    #preallocate arrays
    poplists_shape = (len(slope_options), len(error_options))
    poplists = np.empty(poplists_shape, dtype=object)

    #create and generate
    for i, j in np.ndindex(poplists_shape):
        if verbose: print(i, j)
        options = (slope_options[i], error_options[j])
        poplists[i, j] = get_populations(*options, npersons, npops)
        poplists[i, j].generate()

    #pickle
    dump_to_file(poplists, 'poplists'+suffix, dirname=pickle_dir, create_newdir=True)
    if verbose: print("Done generation.")


    # Filtering

    #define
    def get_filter(scorename):
        return {'filter_type':'ratio_early', 'copy':True,
                'index_day':0, 'recovered_ratio':0.3, 'scorename':scorename}
    scorenames = ['symptom', 'symptom_noerror', 'visual', 'visual_yeserror']
    filters = {scorename: get_filter(scorename) for scorename in scorenames}

    #preallocate
    filteredout = np.ones(npersons*npops, dtype=bool)
    df_index = pd.MultiIndex.from_product([range(npops), range(npersons)], names=['pop', 'person'])
    df = pd.DataFrame({scorename: filteredout.copy() for scorename in scorenames}, index=df_index)
    dfs = np.empty(poplists_shape, dtype=object)

    #filter
    for i, j in np.ndindex(poplists_shape):
        if verbose: print(i, j)
        dfs[i,j] = df.copy()
        for scorename, filter_kwargs in filters.items():
            filtered_poplist = poplists[i, j].filter(**filter_kwargs)
            for k in range(npops):
                persons_valid = filtered_poplist[k].persons.flatten()
                dfs[i,j][scorename].loc[k,persons_valid] = False #set filteredout to false

    #pickle
    dump_to_file(dfs, 'filteredout_persons'+suffix, dirname=pickle_dir, create_newdir=True)
    if verbose: print("Done filtering.")


    # Sampling

    #create
    methodologies = [
        get_traditional_methodology(),
        get_realistic_methodology('symptom'),
        get_realistic_methodology('symptom_noerror'),
        get_realistic_methodology('visual'),
        get_realistic_methodology('symptom', delayless=True),
        get_realistic_methodology('symptom_noerror', delayless=True),
        get_realistic_methodology('visual', delayless=True)
    ]

    #preallocate arrays
    sampled_poplists_shape = (*poplists_shape, len(methodologies))
    sampled_poplists = np.empty(sampled_poplists_shape, dtype=object)

    #sample
    beta_rng.reseed(seed)
    for k in range(len(methodologies)):
        beta_rng.reset()
        for i, j in np.ndindex(poplists_shape):
            if verbose: print(i, j, k)
            sampled_poplists[i, j, k] = methodologies[k].sample(poplists[i, j])

    #pickle
    dump_to_file(sampled_poplists, 'sampled_poplists'+suffix, dirname=pickle_dir, create_newdir=True)
    if verbose: print("Done sampling.")


#parameters
npersons=1000
npops=100
slope_options = (1,)
error_options = (0.5,)

#printing
print("Test parameters:")
print(f"npersons={npersons}")
print(f"npops={npops}")
print(f"slope_options={slope_options}")
print(f"error_options={error_options}")
print(f"Total persons: {npersons*npops*len(slope_options)*len(error_options)}")
print()
print("Log: ")

#timing
starttime = datetime.now()
print(f"Started at {starttime.strftime('%H:%M')}.")

try:
    npops_per_sim = 10
    nsims = npops // npops_per_sim
    #nsims = 1
    npops_remainder = npops % npops_per_sim

    #produce independent seeds for each call to simulate()
    ss = np.random.SeedSequence(874586374) 
    seeds = ss.spawn(nsims+1) #at least as many as calls to simulate()

    for i in range(nsims):
        simulate(npops_per_sim, npersons, slope_options, error_options, index=i, seed=seeds[i])
        print(f"Done {npops_per_sim*(i+1)}/{npops}")
    if npops_remainder > 0:
        simulate(npops=npops_remainder, index=nsims, seed=seeds[-1])
        print(f"Done {npops_per_sim*nsims+npops_remainder}/{npops}")
finally:
    #timing
    endtime = datetime.now()
    deltatime = int((endtime-starttime).total_seconds())
    print(f"Took {deltatime//3600} h {(deltatime%3600)//60} min {deltatime%60} s to run.")
