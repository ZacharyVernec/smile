# Standard library imports
import os
from datetime import datetime

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import dill
from scipy import stats

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
pickle_dir = r'C:\Users\zachv\Desktop\smile_desk\simulating_14'

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
    #Rate
    R = helper.Mixture(lower=0, upper=2, 
        mix=0.545868, 
        locs=np.array([0.0773412, 1.0]), 
        scales=np.array([0.056342, 0.743547])) 
    #Initial
    unique = np.arange(20+1)
    counts = np.array([0, 34, 38, 26, 32, 30, 12, 14, 12, 12, 12, 7, 8, 5, 5, 5, 3, 1, 0, 3, 3], dtype=int)
    unique, counts = unique[2:], counts[2:] #exclude 0 and 1 values as already healthy
    probs = counts / np.sum(counts)
    V_0 = stats.rv_discrete(values=(unique, probs))
    #Together give recovery tails of (0.29999967680454737, 0.09999995806927865)

    gen_r = lambda shape: R.rvs(size=shape, random_state=rng)
    gen_v0 = lambda shape: V_0.rvs(size=shape, random_state=rng)
    pop.set_parameter_generator('r', gen_r, 'person')
    pop.set_parameter_generator('v0', gen_v0, 'person')
    
    #Define and set symptom score functions
    gen_symptomscores = lambda v,a: a*(v-VMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: slope_option
    pop.set_parameter_generator('a', gen_a, 'population')
    
    # Define and set error functions
    #Multiplicative
    pop.set_score_generator('symptom', lambda s,C: np.minimum(s*C, 30))
    gen_C_mul = lambda shape: np.random.uniform(1-error_option, 1+error_option, shape)
    pop.set_parameter_generator('C', gen_C_mul, 'day')

    # Repeat
    pops = PopulationList.full(npops, pop)
    
    # Return
    return pops
#simulations

# To reset between methodologies so all methodologies have same variates
#  while populations still have different variates
beta_rng = helper.Beta_rng(1234, 7, 28, 14, 2.9)
def first_delay_func(shape):
    return beta_rng.gen(shape).astype('int')

def get_traditional_methodology():
    methodology = Methodology('traditonal')
    methodology.add_sampler(TraditionalSampler(day=0, delay=first_delay_func))
    methodology.add_sampler(TraditionalSampler(day=('sample', -1), delay=14))
    methodology.add_sampler(TraditionalSampler(day=('sample', -1), delay=14))

    return methodology
def get_realistic_symptom_methodology():
    methodology = Methodology('realistic_symptom')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=first_delay_func)
    methodology.add_sampler(sampler1)

    #if_reached is irrelevant because index is previous sample
    other_delay_func = lambda shape: helper.beta(shape, 0, 14, 4, 3.8).astype('int') #90% at 7
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename='symptom',
                            delay=other_delay_func, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1, scorename='symptom',
                                delay=other_delay_func, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology
def get_realistic_symptom_noerror_methodology():
    methodology = Methodology('realistic_symptom_noerror')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=first_delay_func)
    methodology.add_sampler(sampler1)

    #if_reached is irrelevant because index is previous sample
    other_delay_func = lambda shape: helper.beta(shape, 0, 14, 4, 3.8).astype('int') #90% at 7
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename='symptom_noerror',
                            delay=other_delay_func, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1, scorename='symptom_noerror',
                                delay=other_delay_func, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology
def get_realistic_visual_methodology():
    methodology = Methodology('realistic_visual')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=first_delay_func)
    methodology.add_sampler(sampler1)

    #if_reached is irrelevant because index is previous sample
    other_delay_func = lambda shape: helper.beta(shape, 0, 14, 4, 3.8).astype('int') #90% at 7
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename='visual',
                            delay=other_delay_func, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1, scorename='visual',
                                delay=other_delay_func, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology
def get_delayless_realistic_symptom_methodology():
    methodology = Methodology('delayless_realistic_symptom')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=0)
    methodology.add_sampler(sampler1)

    #if_reached is irrelevant because index is previous sample
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename='symptom',
                            delay=0, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1, scorename='symptom',
                                delay=0, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology
def get_delayless_realistic_symptom_noerror_methodology():
    methodology = Methodology('delayless_realistic_symptom_noerror')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=0)
    methodology.add_sampler(sampler1)

    #if_reached is irrelevant because index is previous sample
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename='symptom_noerror',
                            delay=0, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1, scorename='symptom_noerror',
                                delay=0, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology
def get_delayless_realistic_visual_methodology():
    methodology = Methodology('delayless_realistic_visual')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    sampler1 = TraditionalSampler(day=0, delay=0)
    methodology.add_sampler(sampler1)

    #if_reached is irrelevant because index is previous sample
    sampler2 = SmileSampler(index=('sample', -1), ratio=0.5, scorename='visual',
                            delay=0, triggered_by_equal=True, min_triggered=2,
                            limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler2)

    #same delay as previous
    sampler3 = MagnitudeSampler(value=1, scorename='visual',
                                delay=0, triggered_by_equal=True, min_triggered=2,
                                limit=(LASTVISIT, 'clip'), if_reached='NaN')
    methodology.add_sampler(sampler3)

    return methodology


def simulate(npops, index=None, seed=1234):
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
    filter_kwargs = {'filter_type':'ratio_early', 'copy':True,
                     'index_day':0, 'recovered_ratio':0.3, 'scorename':'symptom'}

    #preallocate arrays
    filtered_poplists = np.empty_like(poplists)

    #filter
    for i, j in np.ndindex(poplists_shape):
        if verbose: print(i, j)
        filtered_poplists[i, j] = poplists[i, j].filter(**filter_kwargs)

    #pickle
    dump_to_file(filtered_poplists, 'filtered_poplists'+suffix, dirname=pickle_dir, create_newdir=True)
    if verbose: print("Done filtering.")


    # Sampling

    #create
    methodologies = [
        get_traditional_methodology(),
        get_realistic_symptom_methodology(),
        get_realistic_symptom_noerror_methodology(),
        get_realistic_visual_methodology(),
        get_delayless_realistic_symptom_methodology(),
        get_delayless_realistic_symptom_noerror_methodology(),
        get_delayless_realistic_visual_methodology()
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
            sampled_poplists[i, j, k] = methodologies[k].sample(filtered_poplists[i, j])

    #pickle
    dump_to_file(sampled_poplists, 'sampled_poplists'+suffix, dirname=pickle_dir, create_newdir=True)
    if verbose: print("Done sampling.")


#parameters
npersons=1000
npops=100
slope_options = (1, 2, 3)
error_options = (0.3, 0.5)

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
        simulate(npops=npops_per_sim, index=i, seed=seeds[i])
        print(f"Done {npops_per_sim*(i+1)}/{npops}")
    if npops_remainder > 0:
        simulate(npops=npops_remainder, index=nsims, seed=seeds[-1])
        print(f"Done {npops_per_sim*nsims+npops_remainder}/{npops}")
finally:
    #timing
    endtime = datetime.now()
    deltatime = int((endtime-starttime).total_seconds())
    print(f"Took {deltatime//3600} h {(deltatime%3600)//60} min {deltatime%60} s to run.")
