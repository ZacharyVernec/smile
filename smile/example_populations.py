# Third party imports
import numpy as np

# Local application imports
from smile.population import Population, PopulationList
from smile.helper import truncatednormal
from smile.global_params import VMIN, SMIN



# Various functions to exemplify how to create populations

def get_default_pop(size=10000):
    pop = Population(size, 'default pop')        

    #visual score
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 1
    pop.set_parameter_generator('r', gen_r , 'person')
    gen_v0 = lambda shape: 16
    pop.set_parameter_generator('v0', gen_v0, 'person')

    #symptom score no error
    gen_symptomscores = lambda v,a: a*(v-VMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: 1
    pop.set_parameter_generator('a', gen_a, 'population')

    #symptom error
    gen_error_mult = lambda s,C: s*C
    pop.set_score_generator('symptom', gen_error_mult)
    gen_C = lambda shape: 1
    pop.set_parameter_generator('C', gen_C, 'day')

    return pop

def get_poster_pop(size=10000):
    pop = Population(size, 'poster pop')        

    #visual score
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 2
    pop.set_parameter_generator('r', gen_r , 'population')
    gen_v0 = lambda shape: np.random.randint(14, 18+1, size=shape)
    pop.set_parameter_generator('v0', gen_v0, 'person')

    #symptom score no error
    gen_symptomscores = lambda v,a,s0: np.maximum(a*v+s0, SMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: 1
    pop.set_parameter_generator('a', gen_a, 'population')
    gen_s0 = lambda shape: np.random.normal(6, 2, size=shape)
    pop.set_parameter_generator('s0', gen_s0, 'person')

    #symptom error
    gen_error_mult = lambda s,C: s*C
    pop.set_score_generator('symptom', gen_error_mult)
    gen_C = lambda shape: 1 + np.random.choice([-1, 1], size=shape)*(np.random.randint(0, 30, size=shape)/100)
    pop.set_parameter_generator('C', gen_C, 'day')
    
    return pop

def get_prevsim_pop(size=10):
    pop = Population(size, 'previous simulation - basic pop')
    
    #visual score
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 1
    pop.set_parameter_generator('r', gen_r , 'population')
    gen_v0 = lambda shape: truncatednormal(14, 18, 3, shape)
    pop.set_parameter_generator('v0', gen_v0, 'person')

    #symptom score no error
    gen_symptomscores = lambda v,a: a*(v-VMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: random.choice([1, 2, 3], shape)
    pop.set_parameter_generator('a', gen_a, 'person')

    #symptom error
    gen_error_mult = lambda s,C: s*C
    pop.set_score_generator('symptom', gen_error_mult)
    gen_C = lambda shape: truncatednormal(0.8, 1.2, 3, shape)
    pop.set_parameter_generator('C', gen_C, 'day')
    
    return pop

def get_linpop(size=10000):
    pop = get_default_pop(size)
    pop.title = 'linear population with multiplicative error'
    
    #symptom error
    gen_error_mult = lambda s,C: s*C
    pop.set_score_generator('symptom', gen_error_mult)
    gen_C = lambda shape: truncatednormal(0.7, 1.3, 3, shape)
    pop.set_parameter_generator('C', gen_C, 'day')
    
    return pop

def get_exppop(size=10000):
    pop = get_linpop(size)
    pop.title = 'exponential population with multiplicative error'
    
    #symptom score no error
    gen_symptomscores = lambda v,a,v0,B: a*(v0-VMIN)*(B**v-B**VMIN)/(B**v0-B**VMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_B = lambda shape: np.full(shape=shape, fill_value=1.5)
    pop.set_parameter_generator('B', gen_B, 'population')
    
    return pop

def get_useful_poplists(size=100):
    '''returns a PopulationList with no errors, a PopulationList with additive errors, and a PopulationList with multiplicative errors'''
    
    # Define and set visual score function
    pop_with_visual_score = Population(size)
    gen_visualscores = lambda t,r,v0: np.maximum(-r*t+v0, VMIN)
    pop_with_visual_score.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: truncatednormal(1/15, 31/15, 1, shape)
    gen_v0 = lambda shape: truncatednormal(14, 18, 1, shape)
    pop_with_visual_score.set_parameter_generator('r', gen_r, 'person')
    pop_with_visual_score.set_parameter_generator('v0', gen_v0, 'person')

    #Define and set symptom score functions
    linear_pop = pop_with_visual_score.copy(newtitle='linear')
    gen_symptomscores = lambda v,a: a*(v-VMIN)
    linear_pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: 1
    linear_pop.set_parameter_generator('a', gen_a, 'population')

    exponential_pop = linear_pop.copy(newtitle='exponential')
    gen_symptomscores = lambda v,a,v0,B: a*(v0-VMIN)*(B**v-B**VMIN)/(B**v0-B**VMIN)
    exponential_pop.set_score_generator('symptom_noerror', gen_symptomscores)
    exponential_pop.set_parameter_generator('a', gen_a, 'population')

    B = 1.5
    exponential_quick_pop, exponential_slow_pop = exponential_pop.double(addtitle1='quick', addtitle2='slow')
    exponential_quick_pop.set_parameter_generator('B', lambda shape: B, 'population')
    exponential_slow_pop.set_parameter_generator('B', lambda shape: 1/B, 'population')

    # Define error functions
    gen_error_mul = lambda s,C: s*C
    gen_C_mul = lambda shape: truncatednormal(0.7, 1.3, 1, shape)
    gen_error_add = lambda s,C: np.maximum(s+C, SMIN)
    gen_C_add = lambda shape: truncatednormal(-1, 1, 1, shape)

    pops_beforeerror = PopulationList([linear_pop, exponential_quick_pop, exponential_slow_pop])
    pops_noerror = pops_beforeerror.copy(addtitle='no error')
    pops_mulerror = pops_beforeerror.copy(addtitle='multiplicative error')
    pops_adderror = pops_beforeerror.copy(addtitle='additive error')
    
    #Set error functions
    for (pop_no, pop_mul, pop_add) in zip(pops_noerror, pops_mulerror, pops_adderror):
        
        pop_no.set_score_generator('symptom', lambda s: s)

        pop_mul.set_score_generator('symptom', gen_error_mul)
        pop_mul.set_parameter_generator('C', gen_C_mul, 'day')

        pop_add.set_score_generator('symptom', gen_error_add)
        pop_add.set_parameter_generator('C', gen_C_add, 'day')
    
    #Return
    return pops_noerror, pops_mulerror, pops_adderror