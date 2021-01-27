
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


#TODO should warn how many people trigger limit or if_reached
#TODO replace 'order' with 'position'
class Methodology(ABC):
    '''
    Each sampling method is added one at a time, in order
    '''
    def __init__(self, title=''):
        self.title = title
        self.samplers = []
        
    @property
    def nsamplers(self): return len(self.samplers)
        
    def add_sampler(self, sampler):
        sampler._init_with_order(self.nsamplers)
        self.samplers.append(sampler)
        
    def sample(self, pop_or_poplist):
        #sample differently depending on if arg is a Population or a PopulationList
        if isinstance(pop_or_poplist, PopulationList):
            return self.sample_populationlist(pop_or_poplist)
        elif isinstance(pop_or_poplist, Population):
            return self.sample_population(pop_or_poplist)
        else:
            raise TypeError(f"{pop_or_poplist} is a {type(pop_or_poplist)} when it should be a {Population} or a {PopulationList}.")
                
    def sample_populationlist(self, poplist):
        #sample all individually and collect
        sampled_poplist = PopulationList([self.sample_population(pop) for pop in poplist])
        
        #setting title
        if len(set(sampled_poplist.titles)) == 1: #if only one unique title
            sampled_poplist.title = sampled_poplist.titles[0] #use that unique title
        else: #can only be unspecific
            sampled_poplist.title='sampled by '+self.title
            
        return sampled_poplist
            
    def sample_population(self, population):
        #contains all days which will be sampled for all persons
        sampling_days = np.empty((population.npersons, self.nsamplers), dtype=int)
        
        #populate sampling_days according to the methods
        population.sampling_summary = {'nsamplers':self.nsamplers, 'limit':[], 'if_reached':[]}
        for order, sampler in enumerate(self.samplers):
            sampler.sample(population, order, sampling_days)
        
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
        #move summary from population to its copy
        sampled_population.sampling_summary = population.sampling_summary
        del population.sampling_summary
        #return
        return sampled_population
    
    
class Sampler(ABC):
    def __init__(self, name, delay=0, limit=(LASTVISIT, 'raise'), if_reached='raise'):
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
        '''
        self.name = name
        
        #check delay
        if isinstance(delay, int):
            pass
        elif callable(delay):
            if not delay.__code__.co_varnames == ('shape',): 
                raise ValueError("The function for delay generation should only have 'shape' as an argument.")
        else: 
            raise TypeError(f"delay of {delay} is not an int nor is it callable")
        self.delay = delay
        
        #check and set limit
        if isinstance(limit, tuple) and len(limit) == 2:
            limitval, limitbehaviour = limit
            #limitval
            if limitval is None:
                limitval = LASTVISIT
            if isinstance(limit[0], int):
                pass 
            elif isinstance(limitval, tuple) and len(limitval) == 2:
                if not (isinstance(limitval[0], int) or limitval[0] is None):
                    raise TypeError(f"index reference has value {limitval[0]} which is not an int nor None")
                if not callable(limitval[1]):
                    raise TypeError(f"{limitval[1]} should be callable")
            else:
                raise TypeError(f"limit value of {limitval} should be a tuple of (reference_to_prev_sample, lambda)")
            #limitbehaviour
            if limitbehaviour not in {'NaN', 'clip', 'raise'}:
                raise ValueError(f"limitbehaviour of {limitbehaviour} not understood.")
        else:
            raise ValueError(f"limit of {limit} should be a tuple of (limitvalue, limitbehaviour)")
        self.limit = limitval, limitbehaviour
                            
        #check and set if_reached
        if not if_reached in {'same', 'NaN', 'raise'}:
            raise ValueError(f"if_reached of {if_reached} not known")
        self.if_reached = if_reached
        
    def _init_with_order(self, order):
        #limit
        if isinstance(self.limit, tuple) and isinstance(self.limit[0], tuple):
            if not(-order <= self.limit[0][0] <= order):
                raise ValueError(f"index reference of {self.limit[0][0]} does not refer to a previous sample.")
            
    @abstractmethod
    def sample(self, population, order, sampling_days):
        #finish all implementations by calling super().finish_sampling()
        pass
    
    def finish_sampling(self, population, order, sampling_days):
        
        #check if_reached
        if order > 0:
            #checks if new calling day is on the same day as the prev sample or before
            #TraditionalSampler is allowed to have new calling day on the same day as prev (but not before)
            if not isinstance(self, TraditionalSampler): #TODO make this check a parameter set in class definition
                already_reached = (sampling_days[:,order] <= sampling_days[:,order-1])
            else:
                already_reached = (sampling_days[:,order] < sampling_days[:,order-1])
            
            if np.any(already_reached): 
                warn(f"There are {already_reached.sum()} who had already reached their milestone")

            if self.if_reached == 'same':
                #fast forwards new sample to previous sample
                sampling_days[:,order] = np.where(already_reached, sampling_days[:,order-1], sampling_days[:,order])
            if self.if_reached == 'NaN':
                #will be masked with fill_value = NaN
                sampling_days[:,order] = np.where(already_reached, _ALREADYREACHED, sampling_days[:,order]) 
            if self.if_reached == 'raise':
                if np.any(already_reached): 
                    raise ValueError("Patient was already here when he arrived for his prev sample")
            #remember how many triggered if_reached
            population.sampling_summary['if_reached'].append((np.sum(already_reached), self.if_reached))
        else:
            #remember how many triggered if_reached
            population.sampling_summary['if_reached'].append((0, self.if_reached))
        
        #add delay
        persons_valid = sampling_days[:,order] < NDAYS
        npersons_valid = persons_valid.sum()
        if isinstance(self.delay, int):
            sampling_days[persons_valid,order] += self.delay
        elif callable(self.delay):
            sampling_days[persons_valid,order] += self.delay((npersons_valid,))

        #limit #TODO ask if should be done before delay
        limitval, limitbehaviour = self.limit #unpack
        if isinstance(limitval, int):
            limitvals = limitval #numpy will broadcast to the right shape
        elif isinstance(limitval, tuple):
            ref_index, limitvalfunc = limitval #unpack
            prev_sampling_days = sampling_days[:,:order]
            limitvals = limitvalfunc(prev_sampling_days[:,ref_index])
        #check where reached or exceeded limit
        reached_limit = sampling_days[:,order] > limitvals
        reached_limit = np.logical_and(reached_limit, persons_valid) #ignore those already_sampled
        #act on limit
        if limitbehaviour == 'raise':
            if np.any(reached_limit):
                raise IndexError("Reached limit") #TODO better error message
        if limitbehaviour == 'clip':
            sampling_days[:,order] = np.where(reached_limit, limitvals, sampling_days[:,order])
        if limitbehaviour == 'NaN':
            #will be masked with fill_value = NaN
            sampling_days[:,order] = np.where(reached_limit, _LIMITREACHED, sampling_days[:,order])
        #TODO add ('replace', replaceval) as a limitbehaviour option (where 'clip would be a special case')
        #remember how many triggered limit
        population.sampling_summary['limit'].append((np.sum(reached_limit), limitbehaviour))
        
class TraditionalSampler(Sampler):
    def __init__(self, day, **kwargs):
        '''
        day: which day of the simulation to sample
        kwargs: passed to parent class
        '''
        super().__init__(name='traditional', **kwargs)
        
        #check parameters
        
        if isinstance(day, int):
            pass
        elif isinstance(day, tuple): #check if refers to previous sample
            if len(day) == 2 and day[0] == 'sample':
                if not isinstance(day[1], int):
                    raise TypeError(f"index reference has value {day[1]} which is not an int")
            else: 
                raise ValueError(f"day tuple of {day} is defined wrong. "
                                 "It should have length 2 and it's first value should be the string 'sample'")
        elif callable(day):
            if day.__code__.co_varnames != ('shape', 'prev_sampling_days'): 
                raise ValueError("The function for day generation should only have 'shape' and 'prev_sampling_days' as an argument.")
        else:
            raise TypeError(f"day of {day} is of type {type(day)}, which is not int, tuple, or a callable")
        self.day = day
            
    def _init_with_order(self, order):
        super()._init_with_order(order)
        if isinstance(self.day, tuple):
            if -order <= self.day[1] <= order:
                #convert to callable
                func = lambda shape, prev_sampling_days, prev_ref: prev_sampling_days[:, prev_ref]
                #need to bind current value to future calls to avoid circular reference
                partial_func = partial(func, prev_ref=self.day[1])
                self.day = lambda shape, prev_sampling_days: partial_func(shape, prev_sampling_days)
            else:
                raise ValueError(f"day reference of {self.day[1]} does not refer to a previous sample")
                
    def sample(self, population, order, sampling_days):
        if isinstance(self.day, int):
            sampling_days[:,order] = self.day
        elif isinstance(self.day, tuple):
            prev_sampling_days = sampling_days[:,:order]
            sampling_days[:,order] = prev_sampling_days[:,self.day[1]]
        elif callable(self.day):
            prev_sampling_days = sampling_days[:,:order]
            #TODO check if ints not outside NDAYS, FIRSTVISIT, LASTVISIT
            sampling_days[:,order] = self.day((population.npersons,), prev_sampling_days)
            
        super().finish_sampling(population, order, sampling_days)
                
class SmileSampler(Sampler):
    def __init__(self, index=FIRSTVISIT, ratio=0.5, scorename='symptom', 
                 triggered_by_equal=True, min_triggered=1, **kwargs):
        '''
        index: int of the day or 2-tuple where the first entry is the string 'sample'
            and the second entry determines which previous sample to reference (nonzero int)
        ratio: what ratio triggers this smile milestone, between 0 and 1 for useful results
        scorename: which score the ratio refers to
        triggered_by_equal: if True, use <= for trigger, if False, use < for trigger
        min_triggered: the number of days in a row to fulfill the condition when sampling
        kwargs: passed to parent class
        '''
        super().__init__(name='smile', **kwargs)
                            
        #check parameters
                            
        #check index
        if isinstance(index, int):
            pass
        elif isinstance(index, tuple): #check if refers to previous sample
            if len(index) == 2 and index[0] == 'sample':
                if not isinstance(index[1], int):
                    raise TypeError(f"index reference has value {index[1]} which is not an int")
            else: 
                raise ValueError(f"index tuple of {index} is defined wrong. "
                                 "It should have length 2 and it's first value should be the string 'sample'")
        elif callable(index):
            if index.__code__.co_varnames != ('shape', 'prev_sampling_days'): 
                raise ValueError("The function for index day generation should only have 'shape' and 'prev_sampling_days' as an argument.")
        else:
            raise TypeError(f"index of {index} is of type {type(index)}, which is not int, tuple, or a callable")
        self.index = index
                                             
        #check and set ratio
        if not (0 < ratio < 1): 
            warn(f"ratio of {ratio} may be unobtainable.")
        self.ratio = ratio
        
        #check and set scorename
        if scorename not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"scorename of {scorename} not understood")
        self.scorename = scorename
        
        #check and set triggered_by_equal
        if not isinstance(triggered_by_equal, bool):
            raise TypeError(f"triggred_by_equal of {triggered_by_equal} should be a boolean")
        self.triggered_by_equal = triggered_by_equal
        
        #check and set min_triggered
        if not isinstance(min_triggered, int) or min_triggered < 1:
            raise TypeError(f"min_triggered of {min_triggered} should be an int of at least 1")
        self.min_triggered  = min_triggered
      
    def _init_with_order(self, order):
        super()._init_with_order(order)
        if isinstance(self.index, tuple):
            if (-order <= self.index[1] <= order):
                #convert to callable
                func = lambda shape, prev_sampling_days, prev_ref: prev_sampling_days[:, prev_ref]
                #need to bind current value to future calls to avoid circular reference
                partial_func = partial(func, prev_ref=self.index[1])
                self.index = lambda shape, prev_sampling_days: partial_func(shape, prev_sampling_days)
            else:
                raise ValueError(f"index reference of {self.index[1]} does not refer to a previous sample")
                
    def sample(self, population, order, sampling_days):
        smilescores = population.scores[self.scorename] #scores which the method ratio refers to
        smilescore_lowerbound = get_MIN(self.scorename)

        #get and check index days
        if isinstance(self.index, int):
            index_days = np.full((population.npersons,), self.index)
        elif isinstance(self.index, tuple):
            prev_sampling_days = sampling_days[:,:order]
            index_days = prev_sampling_days[:,self.day[1]]
        elif callable(self.index):
            prev_sampling_days = sampling_days[:,:order]
            #TODO check if int not outside NDAYS, FIRSTVISIT, LASTVISIT
            index_days = self.index((population.npersons,), prev_sampling_days)

        # Compute the scores which will trigger milestones
        smilescores_at_index = np.take_along_axis(smilescores, helper.to_vertical(index_days), axis=1) #column array
        smile_vals = (smilescores_at_index - smilescore_lowerbound)*self.ratio + smilescore_lowerbound #column array

        # Compute the days where the milestones are triggered
        comparison_array = (smilescores <= smile_vals) if self.triggered_by_equal else (smilescores < smile_vals)
        # Compute the days where the milestones are triggered consecutively
        if self.min_triggered == 1:
            pass #don't change comparison_array
        elif self.min_triggered > 1:
            triggered_in_a_row = np.ones_like(comparison_array[:,self.min_triggered-1:]) #initial
            for start in range(self.min_triggered):
                end = start + 1-self.min_triggered
                if end == 0: end = None
                triggered_in_a_row = triggered_in_a_row * comparison_array[:,start:end] # accumulate
            comparison_array[:,self.min_triggered-1:] = triggered_in_a_row #we only checked when enough days have passed
            comparison_array[:,:self.min_triggered-1] = False #the rest can't have had enough days in a row
          
        #only check on or after previous sample day by
        #setting the comparison values from days 0 to prev sample day (excluding end) to False
        for i in range(population.npersons):
            comparison_array[i,:sampling_days[i, order-1]] = False
        #if it is True on the same day as the previous sample day, the finish_sampling will consider it already_reached
            
        #the day at which the milestone is reached for each person
        sampling_days_temp = np.argmax(comparison_array, axis=1) 
        #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
        sampling_days[:,order] = sampling_days_temp 
        
        #record of which persons actually reached the milestones
        persons_reached_milestone = np.take_along_axis(comparison_array, 
                                                       helper.to_vertical(sampling_days_temp), 
                                                       axis=1)
        #give invalid day to those who didn't reach
        sampling_days[~persons_reached_milestone.flatten(), order] = _UNREACHED_SMILE 
        if not np.all(persons_reached_milestone): 
            warn(f"There are {(~persons_reached_milestone.flatten()).sum()} who didn't reach their milestone")
            
        super().finish_sampling(population, order, sampling_days)
                
class MagnitudeSampler(Sampler):
    def __init__(self, value=None, scorename='symptom', 
                 triggered_by_equal=True, min_triggered=1, **kwargs):
        '''
        value: what value triggers this milestone (None means minimum possible given the scorename)
        scorename: which score the value refers to
        triggered_by_equal: if True, use <= for trigger, if False, use < for trigger
        min_triggered: the number of days in a row to fulfill the condition when sampling
        kwargs: passed to parent class
        '''
        super().__init__(name='magnitude', **kwargs)
                                         
        #check and set scorename
        if scorename not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"scorename of {scorename} not understood")
        self.scorename = scorename
        
        #check and set triggered_by_equal
        if not isinstance(triggered_by_equal, bool):
            raise TypeError(f"triggred_by_equal of {triggered_by_equal} should be a boolean")
        self.triggered_by_equal = triggered_by_equal
        
        #check and set value
        if value is None:
            value = get_MIN(self.scorename)
        if self.triggered_by_equal:
            if value < get_MIN(self.scorename):
                warn(f"value of {value} may be unobtainable since it is smaller than "
                     f"{self.scorename}'s min of MIN of {get_MIN(self.scorename)}")
        else:
            if value <= get_MIN(self.scorename):
                warn(f"value of {value} may be unobtainable since it is smaller or equal to "
                     f"{self.scorename}'s MIN of {get_MIN(self.scorename)}")
        self.value = value
        
        #check and set min_triggered
        if not isinstance(min_triggered, int) or min_triggered < 1:
            raise TypeError(f"min_triggered of {min_triggered} should be an int of at least 1")
        self.min_triggered  = min_triggered
                
    def sample(self, population, order, sampling_days):
        smilescores = population.scores[self.scorename] #scores which the method value refers to
        smilescore_lowerbound = get_MIN(self.scorename)

        # Compute the days where the milestones are triggered
        comparison_array = (smilescores <= self.value) if self.triggered_by_equal else (smilescores < self.value)
        # Compute the days where the milestones are triggered consecutively
        if self.min_triggered == 1:
            pass #don't change comparison_array
        elif self.min_triggered > 1:
            triggered_in_a_row = np.ones_like(comparison_array[:,self.min_triggered-1:]) #initial
            for start in range(self.min_triggered):
                end = start + 1-self.min_triggered
                if end == 0: end = None
                triggered_in_a_row = triggered_in_a_row * comparison_array[:,start:end] # accumulate
            comparison_array[:,self.min_triggered-1:] = triggered_in_a_row #we only checked when enough days have passed
            comparison_array[:,:self.min_triggered-1] = False #the rest can't have had enough days in a row
            
            #only check on or after previous sample day by
        #setting the comparison values from days 0 to prev sample day (excluding end) to False
        for i in range(population.npersons):
            comparison_array[i,:sampling_days[i, order-1]] = False
        #if it is True on the same day as the previous sample day, the finish_sampling will consider it already_reached
            
        #the day at which the milestone is reached for each person
        sampling_days_temp = np.argmax(comparison_array, axis=1) 
        #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
        sampling_days[:,order] = sampling_days_temp 
        #record of which persons reached the milestones
        persons_reached_milestone = np.take_along_axis(comparison_array, 
                                                       helper.to_vertical(sampling_days_temp), 
                                                       axis=1)
        #give invalid day to those who didn't reach
        sampling_days[~persons_reached_milestone.flatten(), order] = _UNREACHED_MAGNITUDE
        if not np.all(persons_reached_milestone): 
            warn(f"There are {(~persons_reached_milestone.flatten()).sum()} who didn't reach their milestone")
            
        super().finish_sampling(population, order, sampling_days)