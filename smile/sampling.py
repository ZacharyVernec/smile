
#TODO check score percentage not only first dip, but multiple consecutive days

# Standard library imports
from collections import UserList
from abc import ABC, abstractmethod #abstract base class
from functools import partial #for binding variables to functions

# Third party imports
import numpy as np
import numpy.ma as ma

# Local application imports
from smile.population import Population, PopulationList
from smile import helper
from smile.helper import warn
from smile.global_params import get_MIN, NDAYS, FIRSTVISIT, LASTVISIT
from smile.global_params import _UNREACHED_SMILE, _UNREACHED_MAGNITUDE, _LIMITREACHED, _ALREADYREACHED



class Methodology(ABC):
    def __init__(self, title=''):
        self.title = title
        super().__init__()
        
    def sample(self, pop_or_poplist):
        #if population is a PopulationList, apply the single-population version to all
        if isinstance(pop_or_poplist, PopulationList):
            return self.sample_populationlist(pop_or_poplist)
        elif isinstance(pop_or_poplist, Population):
            return self.sample_population(pop_or_poplist)
        else:
            raise TypeError(f"{pop_or_poplist} is a {type(pop_or_poplist)} when it should be a {Population} or a {PopulationList}.")
                
    def sample_populationlist(self, poplist):
        sampled_poplist = PopulationList([self.sample_population(pop) for pop in poplist])
        
        #setting title
        if len(set(sampled_poplist.titles)) == 1: #if only one unique title
            sampled_poplist.title = sampled_poplist.titles[0] #use that unique title
        else: #can only be unspecific
            sampled_poplist.title='sampled by '+self.title
            
        return sampled_poplist
            
    @abstractmethod
    def sample_population(self, population):
        pass
    
#TODO should warn how many people trigger limit or if_reached
class SequentialMethodology(Methodology):
    '''
    Each sampling method is added one at a time, in order
    '''

    def __init__(self, title=''):
        #title
        super().__init__(title=title)
        self.methods = []
                            
    def add_method(self, name, delay=0, limit=(LASTVISIT, 'raise'), if_reached='raise', **kwargs):
        '''
        name: What method is used for this sampler, can be traditional, smile, or magnitude.
        delay: can be an int or a callable with input 'shape'.
        limit: 2-tuple where the first entry determines the limit value 
                which is an int or tuple of ref to previous sample and function of that value,
            and the second entry determines what to do when the limit is reached
                which is to use NaN, to clip to the limit, or to raise an error.
        if_reached: determines what to do if this sample has already been reached at a previous session.
            Essentially, what to do if after the previous method's sample you tell the patient
            to 'call back when _addedmethod_' but they respond with 'oh but I've already _addedsampler_'
            Can be 'same', 'NaN', 'raise'
        kwargs: any keyword arguments relevant for the sampler type
        '''
        method = {'order':self.nmethods, 'name':name, 'delay':delay, 'limit':limit, 'if_reached':if_reached, **kwargs}
                            
        #check parameters
        #check name
        if method['name'] not in {'traditional', 'smile', 'magnitude'}:
            raise ValueError(f"name of {method['name']} not understood")
                            
        #check delay
        if isinstance(method['delay'], int):
            func = lambda shape, value: np.full(shape, value, dtype=int)
            #need to bind current value to future calls to avoid circular reference
            partial_func = partial(func, value=method['delay'])
            method['delay'] = lambda shape: partial_func(shape)
        elif callable(delay):
            if not method['delay'].__code__.co_varnames == ('shape',): 
                raise ValueError("The function for delay generation should only have 'shape' as an argument.")
        else: 
            raise TypeError(f"delay of {delay} is not an int nor is it callable")
                            
        #check limit
        if isinstance(method['limit'], tuple) and len(method['limit']) == 2:
            #limitval
            if method['limit'][0] is None:
                method['limit'] = (LASTVISIT, method['limit'][1])
            if isinstance(method['limit'][0], int):
                intvalue = method['limit'][0]
                method['limit'] = ((None, lambda val: intvalue), method['limit'][1])
            #limitvalfunc
            if isinstance(method['limit'][0], tuple) and len(method['limit'][0]) == 2:
                if isinstance(method['limit'][0][0], int):
                    if not(-method['order'] <= method['limit'][0][0] <= method['order']):
                        raise ValueError(f"index reference of {method['limit'][0][0]} does not refer to a previous sample.")
                elif method['limit'][0][0] is not None:
                    raise TypeError(f"index reference has value {method['limit'][0][0]} which is not an int")
            else:
                raise TypeError(f"limit value of {method['limit'][0]} should be a tuple of (reference_to_prev_sample, lambda)")
            #limitbehaviour
            if method['limit'][1] not in {'NaN', 'clip', 'raise'}:
                raise ValueError(f"limitbehaviour of {method['limit'][1]} not understood.")
        else:
            raise ValueError(f"limit of {method['limit']} should be a tuple of (limitvalue, limitbehaviour)")
                            
        #check if_reached
        if not method['if_reached'] in {'same', 'NaN', 'raise'}:
                raise ValueError(f"if_reached of {method['if_reached']} not known")
                            
        #add method
        self.methods.append(method)
    def add_method_traditional(self, day=0, delay=0, limit=(LASTVISIT, 'raise'), if_reached='raise'):
        '''day: which day of the simulation to sample'''
        
        self.add_method(name='traditional', limit=limit, if_reached=if_reached, day=day, delay=delay)
        method = self.methods[-1]
                            
        #check parameter
        
        if isinstance(method['day'], int):
            #convert to callable
            func = lambda shape, value: np.full(shape, value, dtype=int)
            #need to bind current value to future calls to avoid circular reference
            partial_func = partial(func, value=method['day'])
            method['day'] = lambda shape, prev_sampling_days: partial_func(shape)
            
        elif isinstance(method['day'], tuple): #check if refers to previous sample
            if len(method['day']) == 2 and method['day'][0] == 'sample':
                if isinstance(method['day'][1], int):
                    prev_ref = method['day'][1]
                    if -method['order'] <= prev_ref <= method['order']:
                        method['day'] = lambda shape, prev_sampling_days: prev_sampling_days[:, prev_ref]
                    else:
                        raise ValueError(f"index reference of {prev_ref} does not refer to a previous sample")
                else:
                    raise TypeError(f"index reference has value {method['day'][1]} which is not an int")
            else: 
                raise ValueError(f"day tuple of {method['day']} is defined wrong. "
                                 "It should have length 2 and it's first value should be the string 'sample'")
                
        #check if properly set as callable
        if method['day'].__code__.co_varnames != ('shape', 'prev_sampling_days'): 
            raise ValueError("The function for index day generation should only have 'shape' and 'prev_sampling_days' as an argument.")
    #TODO let index be a function of previous sample
    def add_method_smile(self, index=FIRSTVISIT, ratio=0.5, triggered_by_equal=True, scorename='symptom',
                         delay=0, limit=(LASTVISIT, 'raise'), if_reached='raise'):
        '''
        index: int of the day or 2-tuple where the first entry is the string 'sample'
            and the second entry determines which previous sample to reference (positive or negative int)
        ratio: what ratio triggers this smile milestone, should generally be between 0 and 1 for useful results
        triggered_by_equal: if True, use <= for trigger, if False, use < for trigger
        scorename: which score the ratio refers to
        delay, limit, if_reached: as in add_method
        '''
        
        self.add_method(name='smile', limit=limit, if_reached=if_reached, delay=delay,
                       index=index, ratio=ratio, triggered_by_equal=triggered_by_equal, scorename=scorename)
        method = self.methods[-1]
                            
        #check parameters
                            
        #set index_day as callable
        if isinstance(method['index'], int):
            func = lambda shape, value: np.full(shape, value, dtype=int)
            #need to bind current value to future calls to avoid circular reference
            partial_func = partial(func, value=method['index'])
            method['index'] = lambda shape, prev_sampling_days: partial_func(shape)
            
        elif isinstance(method['index'], tuple): #check if refers to previous sample
            if len(method['index']) == 2 and method['index'][0] == 'sample':
                if isinstance(method['index'][1], int):
                    prev_ref = method['index'][1]
                    if -method['order'] <= prev_ref <= method['order']:
                        method['index'] = lambda shape, prev_sampling_days: prev_sampling_days[:, prev_ref]
                    else:
                        raise ValueError(f"index reference of {prev_ref} does not refer to a previous sample")
                else:
                    raise TypeError(f"index reference has value {method['index'][1]} which is not an int")
            else: 
                raise ValueError(f"index tuple of {method['index']} is defined wrong. "
                                 "It should have length 2 and it's first value should be the string 'sample'")
        #check if properly set as callable
        if method['index'].__code__.co_varnames != ('shape', 'prev_sampling_days'): 
            raise ValueError("The function for index day generation should only have 'shape' and 'prev_sampling_days' as an argument.")
                                             
        #check ratio
        if not (0 < method['ratio'] < 1): 
            warn(f"ratio of {method['ratio']} may be unobtainable.")

        #check scorename
        if method['scorename'] not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"scorename of {method['scorename']} not understood")                             
    def add_method_magnitude(self, value=None, triggered_by_equal=True, scorename='symptom', 
                             delay=0, limit=(LASTVISIT, 'raise'), if_reached='raise'): #TODO remove defaults
        '''
        value: what value triggers this milestone (None means minimum possible given the scorename)
        triggered_by_equal: if True, use <= for trigger, if False, use < for trigger
        scorename: which score the value refers to
        delay, limit, if_reached: as in add_method
        '''
        
        self.add_method(name='magnitude', limit=limit, if_reached=if_reached, delay=delay,
                       value=value, triggered_by_equal=triggered_by_equal, scorename=scorename)
        method = self.methods[-1]
                        
        #check parameters
                                         
        #check scorename
        if method['scorename'] not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"scorename of {method['scorename']} not understood")
        #check value
        if method['value'] is None:
            method['value'] = get_MIN(method['scorename'])
        if method['triggered_by_equal']:
            if method['value'] < get_MIN(method['scorename']):
                warn(f"value of {method['value']} may be unobtainable since it is smaller than "
                     f"{scorename}'s min of MIN of {get_MIN([method['scorename']])}")
        else:
            if method['value'] <= get_MIN(method['scorename']):
                warn(f"value of {method['value']} may be unobtainable since it is smaller or equal to "
                     f"{scorename}'s MIN of {get_MIN(method['scorename'])}")
    
    @property
    def nmethods(self): return len(self.methods)
        
    def sample_population(self, population):
        #contains all days which will be sampled for all persons
        sampling_days = np.empty((population.npersons, self.nmethods), dtype=int)
        
        #populate sampling_days according to the methods
        for i, method in enumerate(self.methods):
            
            #get day each person calls in
            if method['name'] == 'traditional':
                sampling_days[:,i] = method['day']((population.npersons,), sampling_days[:,:i])
                #TODO check if int not outside NDAYS, FIRSTVISIT, LASTVISIT
                
            elif method['name'] == 'smile':
                smilescores = population.scores[method['scorename']] #scores which the method ratio refers to
                smilescore_lowerbound = get_MIN(method['scorename'])
                
                #get and check index days
                index_days = method['index']((population.npersons,), sampling_days[:,:i])
                #TODO check if int not outside NDAYS, FIRSTVISIT, LASTVISIT
                
                # Compute the scores which will trigger milestones
                smilescores_at_index = np.take_along_axis(smilescores, helper.to_vertical(index_days), axis=1) #column array
                smile_vals = (smilescores_at_index - smilescore_lowerbound)*method['ratio'] + smilescore_lowerbound #column array
                
                # Compute the days where the milestones are triggered
                comparison_array = (smilescores <= smile_vals) if method['triggered_by_equal'] else (smilescores < smile_vals)
                sampling_days_temp = np.argmax(comparison_array, axis=1) #the day at which the milestone is reached for each person
                sampling_days[:,i] = sampling_days_temp #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
                persons_reached_milestone = np.take_along_axis(comparison_array, 
                                                               helper.to_vertical(sampling_days_temp), 
                                                               axis=1) #record of which persons reached the milestones
                sampling_days[~persons_reached_milestone.flatten(), i] = _UNREACHED_SMILE
                if not np.all(persons_reached_milestone): warn("There is at least one person who didn't reach his milestone")
                    
            elif method['name'] == 'magnitude':
                smilescores = population.scores[method['scorename']] #scores which the method value refers to
                smilescore_lowerbound = get_MIN(method['scorename'])
                
                # Compute the days where the milestones are triggered
                comparison_array = (smilescores <= method['value']) if method['triggered_by_equal'] else (smilescores < method['value'])
                sampling_days_temp = np.argmax(comparison_array, axis=1) #the day at which the milestone is reached for each person
                sampling_days[:,i] = sampling_days_temp #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
                persons_reached_milestone = np.take_along_axis(comparison_array, 
                                                               helper.to_vertical(sampling_days_temp), 
                                                               axis=1) #record of which persons reached the milestones
                sampling_days[~persons_reached_milestone.flatten(), i] = _UNREACHED_MAGNITUDE
                if not np.all(persons_reached_milestone): warn("There is at least one person who didn't reach his milestone")
                    
            else:
                raise ValueError(f"name of {method['name']} not known")
                
            #add delay
            delay_gen = method['delay']
            persons_valid = sampling_days[:,i] < NDAYS
            npersons_valid = persons_valid.sum()
            sampling_days[persons_valid,i] += delay_gen((npersons_valid,))
            #TODO check if int
            
            #limit #TODO check if should be done before delay
            limitvaltuple, limitbehaviour = method['limit'] #unpack
            ref_index, limitvalfunc = limitvaltuple #unpack
            prev_sampling_days = sampling_days[:,:i]
            limitvals = limitvalfunc(prev_sampling_days[:,ref_index])
            #check where reached or exceeded limit
            reached_limit = sampling_days[:,i] > limitvals
            #act on limit
            if limitbehaviour == 'raise':
                if np.any(reached_limit):
                    raise IndexError("Reached limit") #TODO better error message
            if limitbehaviour == 'clip':
                sampling_days[:,i] = np.where(reached_limit, limitvals, sampling_days[:,i])
            if limitbehaviour == 'NaN':
                #will be masked with fill_value = NaN
                sampling_days[:,i] = np.where(reached_limit, _LIMITREACHED, sampling_days[:,i])
            #TODO add ('replace', replaceval) as a limitbehaviour option (where 'clip would be a special case')
            
            #check if_reached
            if i > 0:
                #checks if new sample is technically before previous (prevents backwards time-travel)
                already_reached = (sampling_days[:,i] <= sampling_days[:,i-1])
                if method['if_reached'] == 'same':
                    #fast forwards new sample to previous sample
                    sampling_days[:,i] = np.where(already_reached, sampling_days[:,i-1], sampling_days[:,i])
                if method['if_reached'] == 'NaN':
                    #will be masked with fill_value = NaN
                    sampling_days[:,i] = np.where(already_reached, _ALREADYREACHED, sampling_days[:,i]) 
                if method['if_reached'] == 'raise':
                    if np.any(already_reached): 
                        raise ValueError("Patient was already here when he arrived for his prev sample")
        
        #convert to mask with masked values being 0 (to not throw an error when using np.take_along_axis)
        mask = sampling_days >= NDAYS
        sampling_days = np.where(mask, 0, sampling_days)
        
        #Sample at sampling days
        new_scores = {scorename:np.take_along_axis(population.scores[scorename], sampling_days, axis=1) 
                      for scorename in population.scores} #both real and fake
        #mask the fake days
        new_days = ma.masked_array(sampling_days, mask=mask, fill_value=NDAYS) #Fake days will give index errors if used
        new_scores = {scorename: ma.masked_array(new_scores[scorename], mask=mask, fill_value=np.nan)
                      for scorename in new_scores}

        # Population
        sampled_population = population.copy(addtitle='\nsampled by '+self.title)
        sampled_population.days = new_days
        sampled_population.scores = new_scores
        return sampled_population
                                         
class RealisticMethodology(SequentialMethodology):
    '''Discussed on a phone call'''
    def __init__(self):
        super().__init__('realistic')
                                     
        #limit is irrelevant because max(day+delay) < NDAYS
        #if_reached is irrelevant because first sampling method
        first_delay_func = lambda shape: helper.beta(shape, 7, 28, 14, 2.9).astype('int') #90% at 21
        self.add_method_traditional(day=0, delay=first_delay_func)
        
        #if_reached is irrelevant because index is previous sample
        other_delay_func = lambda shape: helper.beta(shape, 0, 14, 4, 3.8).astype('int') #90% at 7
        self.add_method_smile(index=('sample', -1), ratio=0.5, triggered_by_equal=True, scorename='symptom',
                             delay=other_delay_func, limit=((-1, lambda prev_day: prev_day+28), 'clip'), if_reached='NaN')
        
        #same delay as previous
        self.add_method_magnitude(value=6, triggered_by_equal=False, scorename='symptom',
                                 delay=other_delay_func, limit=(LASTVISIT, 'clip'), if_reached='NaN')
        
class TraditionalMethodology(SequentialMethodology):
    def __init__(self):
        super().__init__('traditional')
        first_delay_func = lambda shape: helper.beta(shape, 7, 28, 14, 2.9).astype('int') #90% at 21
        self.add_method_traditional(day=0, delay=first_delay_func)
        self.add_method_traditional(day=('sample', 0), delay=14)
        self.add_method_traditional(day=('sample', 0), delay=28)
