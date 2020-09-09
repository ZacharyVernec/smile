# Standard library imports
import os

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import dill

# Local application imports
from smile.population import Population, PopulationList
from smile.sampling import *
from smile.regression import RegressionResultList
from smile import helper
from smile.helper import truncatednormal
from smile.global_params import *

# Settings
seed = 3 # chosen by fair dice roll. guaranteed to be random. https://xkcd.com/221/
np.random.seed(seed)
np.set_printoptions(edgeitems=30, linewidth=100000)
pickle_dir = 'saved_populations'

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
def load_from_file(obj, filename):
    with open(filename, 'rb') as f:
        return dill.load(f)


# Populations

#definitions
def get_poster_populations(slope_option, error_option, npersons=100, npops=100):
    '''
    returns a PopulationList similar to the one described in the poster
    slope_option is 1, 2, or 3
    error_option is 30/100 or 50/100
    '''
    
    # Define and set visual score function
    pop = Population(npersons, f'poster with {slope_option} and {error_option}')
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 0.2
    gen_v0 = lambda shape: np.random.randint(14, 18+1, shape)
    pop.set_parameter_generator('r', gen_r, 'population')
    pop.set_parameter_generator('v0', gen_v0, 'person')

    # Define and set symptom score function
    gen_symptomscores = lambda v,a,s0: np.maximum(a*v+s0, SMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: slope_option
    gen_s0 = lambda shape: np.random.normal(6, 2, shape)
    pop.set_parameter_generator('a', gen_a, 'population')
    pop.set_parameter_generator('s0', gen_s0, 'person')

    # Define and set error functions
    #Multiplicative
    pop.set_score_generator('symptom', lambda s,C: s*C)
    gen_C_mul = lambda shape: np.random.uniform(1-error_option, 1+error_option, shape)
    pop.set_parameter_generator('C', gen_C_mul, 'day')

    # Repeat
    pops = PopulationList.full(npops, pop)
    
    # Return
    return pops
def get_worddoc_populations(slope_option, error_option, npersons=100, npops=100):
    
    # Define and set visual score function
    pop = Population(npersons, title=f'realistic with {slope_option} and {error_option}')
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: helper.beta(shape, 0, 2, 0.2, 1.4) #median at 0.4
    gen_v0 = lambda shape: helper.truncatednormal_general(14, 16, 18, 1, shape)
    pop.set_parameter_generator('r', gen_r, 'person')
    pop.set_parameter_generator('v0', gen_v0, 'person')
    
    #Define and set symptom score functions
    gen_symptomscores = lambda v,a: a*(v-VMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: slope_option
    pop.set_parameter_generator('a', gen_a, 'population')
    
    # Define and set error functions
    #Multiplicative
    pop.set_score_generator('symptom', lambda s,C: s*C)
    gen_C_mul = lambda shape: np.random.uniform(1-error_option, 1+error_option, shape)
    pop.set_parameter_generator('C', gen_C_mul, 'day')

    # Repeat
    pops = PopulationList.full(npops, pop)
    
    # Return
    return pops

#parameters
npersons=10
npops=10
slope_options = (1, 2, 3)
error_options = (0.3, 0.5)

#preallocate arrays
poplists_shape = (len(slope_options), len(error_options))
poster_poplists = np.empty(poplists_shape, dtype=object)
worddoc_poplists = np.empty(poplists_shape, dtype=object)

#create and generate
for i, j in np.ndindex(poplists_shape):
    options = (slope_options[i], error_options[j])
    poster_poplists[i, j] = get_poster_populations(*options, npersons, npops)
    worddoc_poplists[i, j] = get_worddoc_populations(*options, npersons, npops)
    poster_poplists[i, j].generate()
    worddoc_poplists[i, j].generate()
    
#pickle
dump_to_file(poster_poplists, 'poster_poplists', dirname=pickle_dir, create_newdir=True)
dump_to_file(worddoc_poplists, 'worddoc_poplists', dirname=pickle_dir)
    
    
# Filtering

#define
filter_kwargs = {'filter_type':'ratio_early', 'copy':True,
                 'index_day':0, 'recovered_ratio':0.7, 'scorename':'symptom'}

#preallocate arrays
poster_filtered_poplists = np.empty_like(poster_poplists)
worddoc_filtered_poplists = np.empty_like(worddoc_poplists)

#filter
for i, j in np.ndindex(poplists_shape):
    poster_filtered_poplists[i, j] = poster_poplists[i, j].filter(**filter_kwargs)
    worddoc_filtered_poplists[i, j] = worddoc_poplists[i, j].filter(**filter_kwargs)
    
#pickle
dump_to_file(poster_filtered_poplists, 'poster_filtered_poplists', dirname=pickle_dir)
dump_to_file(worddoc_filtered_poplists, 'worddoc_filtered_poplists', dirname=pickle_dir)
    
    
# Sampling

#define
def get_traditional_methodology():
    methodology = Methodology('traditonal')
    
    first_delay_func = lambda shape: helper.beta(shape, 7, 28, 14, 2.9).astype('int') #90% at 21
    
    methodology.add_sampler(TraditionalSampler(day=0, delay=first_delay_func))
    methodology.add_sampler(TraditionalSampler(day=('sample', 0), delay=14))
    methodology.add_sampler(TraditionalSampler(day=('sample', 0), delay=28))
    
    return methodology
def get_realistic_methodology():
    methodology = Methodology('realistic')

    #limit is irrelevant because max(day+delay) < NDAYS
    #if_reached is irrelevant because first sampling method
    first_delay_func = lambda shape: helper.beta(shape, 7, 28, 14, 2.9).astype('int') #90% at 21
    methodology.add_sampler(TraditionalSampler(day=0, delay=first_delay_func))

    #if_reached is irrelevant because index is previous sample
    other_delay_func = lambda shape: helper.beta(shape, 0, 14, 4, 3.8).astype('int') #90% at 7
    methodology.add_sampler(SmileSampler(index=('sample', -1), ratio=0.5, triggered_by_equal=True, scorename='symptom',
                                         delay=other_delay_func, limit=((-1, lambda prev_day: prev_day+28), 'clip'), if_reached='NaN'))

    #same delay as previous
    methodology.add_sampler(MagnitudeSampler(value=6, triggered_by_equal=False, scorename='symptom',
                                             delay=other_delay_func, limit=(LASTVISIT, 'clip'), if_reached='NaN'))
    
    return methodology

#create
methodologies = [get_traditional_methodology(), get_realistic_methodology()]

#preallocate arrays
sampled_poplists_shape = (*poplists_shape, len(methodologies))
poster_sampled_poplists = np.empty(sampled_poplists_shape, dtype=object)
worddoc_sampled_poplists = np.empty(sampled_poplists_shape, dtype=object)

#sample
for i, j in np.ndindex(poplists_shape):
    poster_sampled_poplists[i, j, 0] = methodologies[0].sample(poster_filtered_poplists[i, j])
    poster_sampled_poplists[i, j, 1] = methodologies[1].sample(poster_filtered_poplists[i, j])
    worddoc_sampled_poplists[i, j, 0] = methodologies[0].sample(worddoc_filtered_poplists[i, j])
    worddoc_sampled_poplists[i, j, 1] = methodologies[1].sample(worddoc_filtered_poplists[i, j])
    

#pickle
dump_to_file(poster_sampled_poplists, 'poster_sampled_poplists', dirname=pickle_dir)
dump_to_file(worddoc_sampled_poplists, 'worddoc_sampled_poplists', dirname=pickle_dir)