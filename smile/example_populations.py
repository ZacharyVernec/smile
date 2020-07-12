
# Third party imports
import numpy as np

# Local application imports
from .smile import Population, PopulationList
from .helper import truncatednormal
from .global_params import *

# Various functions to exemplify how to create populations

def get_default_pop(size=10000):
    pop = Population(size, 'default pop')        

    #visual score
    gen_visualscores = lambda t,r,f0: np.maximum(-r*t+f0, FMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 1
    pop.set_parameter_generator('r', gen_r , 'person')
    gen_f0 = lambda shape: 16
    pop.set_parameter_generator('f0', gen_f0, 'person')

    #symptom score no error
    gen_symptomscores = lambda f,a: a*(f-FMIN)
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
    gen_visualscores = lambda t,r,f0: np.maximum(-r*t+f0, FMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 2
    pop.set_parameter_generator('r', gen_r , 'population')
    gen_f0 = lambda shape: np.random.randint(14, 18+1, size=shape)
    pop.set_parameter_generator('f0', gen_f0, 'person')

    #symptom score no error
    gen_symptomscores = lambda f,a,s0: np.maximum(a*f+s0, SMIN)
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
    gen_visualscores = lambda t,r,f0: np.maximum(-r*t+f0, FMIN)
    pop.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: 1
    pop.set_parameter_generator('r', gen_r , 'population')
    gen_f0 = lambda shape: truncatednormal(14, 18, 3, shape)
    pop.set_parameter_generator('f0', gen_f0, 'person')

    #symptom score no error
    gen_symptomscores = lambda f,a: a*(f-FMIN)
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
    gen_symptomscores = lambda f,a,f0,B: a*(f0-FMIN)*(B**f-B**FMIN)/(B**f0-B**FMIN)
    pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_B = lambda shape: np.full(shape=shape, fill_value=1.5)
    pop.set_parameter_generator('B', gen_B, 'population')
    
    return pop

def get_useful_poplists(size=100):
    '''returns a PopulationList with no errors, a PopulationList with additive errors, and a PopulationList with multiplicative errors'''
    
    pop_with_visual_score = Population(size)
    gen_visualscores = lambda t,r,f0: np.maximum(-r*t+f0, FMIN)
    pop_with_visual_score.set_score_generator('visual', gen_visualscores)
    gen_r = lambda shape: truncatednormal(1/15, 31/15, 1, shape)
    gen_f0 = lambda shape: truncatednormal(14, 18, 1, shape)
    pop_with_visual_score.set_parameter_generator('r', gen_r, 'person')
    pop_with_visual_score.set_parameter_generator('f0', gen_f0, 'person')

    linear_pop = pop_with_visual_score.copy(newtitle='linear')
    gen_symptomscores = lambda f,a: a*(f-FMIN)
    linear_pop.set_score_generator('symptom_noerror', gen_symptomscores)
    gen_a = lambda shape: 1
    linear_pop.set_parameter_generator('a', gen_a, 'population')

    exponential_pop = linear_pop.copy(newtitle='exponential')
    gen_symptomscores = lambda f,a,f0,B: a*(f0-FMIN)*(B**f-B**FMIN)/(B**f0-B**FMIN)
    exponential_pop.set_score_generator('symptom_noerror', gen_symptomscores)
    exponential_pop.set_parameter_generator('a', gen_a, 'population')

    B = 1.5
    exponential_quick_pop, exponential_slow_pop = exponential_pop.double(addtitle1='quick', addtitle2='slow')
    exponential_quick_pop.set_parameter_generator('B', lambda shape: B, 'population')
    exponential_slow_pop.set_parameter_generator('B', lambda shape: 1/B, 'population')

    gen_error_mul = lambda s,C: s*C
    gen_C_mul = lambda shape: truncatednormal(0.7, 1.3, 1, shape)
    gen_error_add = lambda s,C: np.maximum(s+C, SMIN)
    gen_C_add = lambda shape: truncatednormal(-1, 1, 1, shape)

    pops_noerror = PopulationList([linear_pop, exponential_quick_pop, exponential_slow_pop])
    pops_mulerror = PopulationList()
    pops_adderror = PopulationList()

    for i in range(len(pops_noerror)):
        pop_mul, pop_add = pops_noerror[i].double(addtitle1='multiplicative error', addtitle2='additive error')

        pop_mul.set_score_generator('symptom', gen_error_mul)
        pop_mul.set_parameter_generator('C', gen_C_mul, 'day')
        pops_mulerror.append(pop_mul)

        pop_add.set_score_generator('symptom', gen_error_add)
        pop_add.set_parameter_generator('C', gen_C_add, 'day')
        pops_adderror.append(pop_add)
        
    return pops_noerror, pops_mulerror, pops_adderror