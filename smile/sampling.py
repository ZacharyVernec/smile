#TODO check score percentage not only first dip, but multiple consecutive days

# Standard library imports
from collections import UserList
from abc import ABC, abstractmethod #abstract base class

# Third party imports
import numpy as np
import numpy.ma as ma

# Local application imports
from smile.population import Population, PopulationList
from smile import helper
from smile.helper import warn
from smile.global_params import get_MIN, NDAYS, FIRSTVISIT, LASTVISIT



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
        
class TraditionalMethodology(Methodology):
    '''Sampling at fixed days'''
    def __init__(self, title='', sampling_days=[8, 15, 31]):
        
        super().__init__(title)
        
        self.sampling_days = sampling_days
        if not all(isinstance(day, int) for day in sampling_days):
            raise TypeError("Sampling days must be ints")
        if max(sampling_days) >= NDAYS:
            raise ValueError(f"There is a fixed sample day in {sampling_days} that is later than the simulation duration of {NDAYS}")
        if max(sampling_days) > LASTVISIT:
            warn(f"There is a fixed sample day in {sampling_days} that is later than the LASTVISIT of {LASTVISIT}")
        if min(sampling_days) < FIRSTVISIT:
            warn(f"There is a fixed sample day in {sampling_days} that is earlier than the FIRSTVISIT of {FIRSTVISIT}")
        
    def sample_population(self, population):

        # Sampling
        sampling_days = np.tile(self.sampling_days, (population.npersons,1)) #each person (row) has same sampling days
        sampling_scores = {scorename:np.take_along_axis(population.scores[scorename], sampling_days, axis=1) for scorename in population.scores}
        
        # Population
        samplepop = population.copy(addtitle='\nsampled by '+self.title)
        samplepop.days = sampling_days
        samplepop.scores = {scorename:sampling_scores[scorename] for scorename in samplepop.scores}
        return samplepop
    
class SmileMethodology(Methodology):
    '''Sampling at milestones'''
    def __init__(self, title='', smile_scorename='symptom',
                 index_day=0, sample_index=True,
                 milestone_ratios=[0.7, 0.4], delay=0):
        '''
        index_day determines which day's score will be used as a 'baseline' from which a milestone can be reached
        milestone_ratios should generally be between 0 and 1 for useful results
            if a ratio is not reached or a delay pushes it to > NDAYS, it gets a day value of NDAYS and a score value of np.NaN
        smile_scorename determines which score the milestone_ratios will be based on (can be symptom, visual, or symptom_noerror)
        '''
        super().__init__(title)
        
        #set index_day as callable
        if isinstance(index_day, int):
            self.index_day = lambda shape: np.full(shape, index_day, dtype=int)
        elif callable(index_day):
            if index_day.__code__.co_varnames == ('shape',): 
                self.index_day = index_day
            else: 
                raise ValueError("The function for index day generation should only have 'shape' as an argument.")
        else:
            raise ValueError(f"index_day of {index_day} is not an int nor is it callable")
        self.sample_index = sample_index
            
        #set delay as callable
        if isinstance(delay, int): 
            self.delay = lambda shape: np.full(shape, delay, dtype=int)
        elif callable(delay):
            if delay.__code__.co_varnames == ('shape',): 
                self.delay = delay
            else: 
                raise ValueError("The function for delay generation should only have 'shape' as an argument.")
        else: 
            raise ValueError(f"delay of {delay} is not an int nor is it callable")
            
        self.milestone_ratios = milestone_ratios
        if not all(0 < ratio < 1 for ratio in milestone_ratios): 
            warn(f"Some milestone_ratios in {milestone_ratios} may be unobtainable.")
            
        self.smile_scorename = smile_scorename
        if smile_scorename not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"smile_scorename of {smile_scorename} not understood")
            
    def sample_population(self, population):
        smilescores = population.scores[self.smile_scorename] #scores which the ratios refer to
        smilescore_lowerbound = get_MIN(self.smile_scorename)
            
        #get and check index days
        index_days = self.index_day((population.npersons, 1))
        earliest_index_day = np.min(index_days)
        if earliest_index_day >= NDAYS:
            raise ValueError(f"The index day {earliest_index_day} is later than the simulation duration of {NDAYS}")
        if earliest_index_day > LASTVISIT:
            warn(f"The index day {earliest_index_day} is later than the LASTVISIT of {LASTVISIT}")
        if earliest_index_day < FIRSTVISIT:
            warn(f"The index day {earliest_index_day} is earlier than the FIRSTVISIT of {FIRSTVISIT}")

        # Compute the scores which will trigger milestones
        smilescores_at_index = np.take_along_axis(smilescores, index_days, axis=1)
        smile_vals = (smilescores_at_index - smilescore_lowerbound)*self.milestone_ratios + smilescore_lowerbound
        #smile_vals are the score values to reach. Each row is a person, each column is an ordinal, and each value is the score the reach

        # Compute the days where the milestones are triggered
        milestone_days = ma.empty_like(smile_vals, dtype=int) #will hold the day each milestone_ratio is reached for each person
        milestone_days.fill_value = NDAYS
        #careful: values of 0 in milestone_days might represent 'day 0' or might represent 'never reached milestone'. 
        #The mask will hold the days where the milestone_ratios is not reached (exc. those stored as 0 meaning 'never reached')
        for milestone_col in range(smile_vals.shape[1]): 
            milestone_vals = helper.to_vertical(smile_vals[:,milestone_col])
            milestone_days_temp = np.argmax(smilescores <= milestone_vals, axis=1) #the day at which the milestone is reached for each person
            milestone_days[:,milestone_col] = milestone_days_temp #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
            milestones_reached_temp = np.take_along_axis(smilescores <= milestone_vals, helper.to_vertical(milestone_days_temp), axis=1).flatten() #record of which persons reached the milestones
            milestone_days[~milestones_reached_temp, milestone_col] = ma.masked
        if np.any(milestone_days.mask): warn("There is a milestone that was not reached.")
            
        #compute the days at which the scores will be sampled (i.e. include delay)
        delay = self.delay(milestone_days.shape)
        sampling_days = milestone_days + delay
        #exclude excessive days 
        exceed_study_duration = sampling_days > LASTVISIT #Will become fake days since can't be sampled
        if np.any(exceed_study_duration): warn("The delay is pushing a sampling day past the study duration.")
        sampling_days[exceed_study_duration] = ma.masked
        
        if self.sample_index == True: #include index_days as a real day
            sampling_days = ma.hstack([index_days, sampling_days])
            
        #make masked values 0 (to not throw an error when using np.take_along_axis)
        mask_temp = sampling_days.mask.copy() #since changing masked values will make them no longer masked
        sampling_days[mask_temp] = 0 #change masked values
        sampling_days[mask_temp] = ma.masked #put back the mask

        #Sample at sampling days
        milestone_scores = {scorename:np.take_along_axis(population.scores[scorename], sampling_days, axis=1) for scorename in population.scores} #both real and fake
        #mask the fake days
        milestone_days = ma.masked_array(sampling_days, fill_value=NDAYS) #Fake days will give index errors if used
        milestone_scores = {scorename: ma.masked_array(milestone_scores[scorename], sampling_days.mask, fill_value=np.nan) for scorename in milestone_scores}

        # Population
        samplepop = population.copy(addtitle='\nsampled by '+self.title)
        samplepop.days = milestone_days
        samplepop.scores = {scorename:milestone_scores[scorename] for scorename in samplepop.scores} #include index_day
        return samplepop
    
class MagnitudeMethodology(Methodology):
    '''Similar to SmileMethodology, but with score magnitude instead of ratio'''
    def __init__(self, title='', smile_scorename='symptom',
                 milestone_values=[None], delay=0):
        '''
        milestone_value of None corresponds to the smile_scorename's corresponding MIN
            if a value is not reached or a delay pushes it to > NDAYS, it gets a day value of NDAYS and a score value of np.NaN
        smile_scorename determines which score the milestone_ratios will be based on (can be symptom, visual, or symptom_noerror)
        '''
        super().__init__(title)
            
        #set delay as callable
        if isinstance(delay, int): 
            self.delay = lambda shape: np.full(shape, delay, dtype=int)
        elif callable(delay):
            if delay.__code__.co_varnames == ('shape',): 
                self.delay = delay
            else: 
                raise ValueError("The function for delay generation should only have 'shape' as an argument.")
        else: 
            raise ValueError(f"delay of {delay} is not an int nor is it callable")
            
        self.smile_scorename = smile_scorename
        if smile_scorename not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"smile_scorename of {smile_scorename} not understood")
            
        self.milestone_values = np.array([val if val is not None else get_MIN(self.smile_scorename) #convert None to scoreMIN
                                          for val in milestone_values], dtype=float)
        if np.min(self.milestone_values) < get_MIN(self.smile_scorename):
            raise ValueError(f"Some milestone_values in {milestone_values} may be unobtainable.")
            
    def sample_population(self, population):
        '''similar to SmileMethodology.sample_population()'''
        smilescores = population.scores[self.smile_scorename] #scores which the milestone_values refer to
        smilescore_lowerbound = get_MIN(self.smile_scorename)
        
        # Reshape the scores which will trigger milestones
        smile_vals = np.broadcast_to(self.milestone_values, (population.npersons, *self.milestone_values.shape))
        #smile_vals are the score values to reach. Each row is a person, each column is an ordinal, and each value is the score the reach

        # Compute the days where the milestones are triggered
        milestone_days = ma.empty_like(smile_vals, dtype=int) #will hold the day each milestone_ratio is reached for each person
        milestone_days.fill_value = NDAYS
        #careful: values of 0 in milestone_days might represent 'day 0' or might represent 'never reached milestone'. 
        #The mask will hold the days where the milestone_ratios is not reached (exc. those stored as 0 meaning 'never reached')
        for milestone_col in range(smile_vals.shape[1]): 
            milestone_vals = helper.to_vertical(smile_vals[:,milestone_col])
            milestone_days_temp = np.argmax(smilescores <= milestone_vals, axis=1) #the day at which the milestone is reached for each person
            milestone_days[:,milestone_col] = milestone_days_temp #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
            milestones_reached_temp = np.take_along_axis(smilescores <= milestone_vals, helper.to_vertical(milestone_days_temp), axis=1).flatten() #record of which persons reached the milestones
            milestone_days[~milestones_reached_temp, milestone_col] = ma.masked
        if np.any(milestone_days.mask): warn("There is a milestone that was not reached.")
            
        #compute the days at which the scores will be sampled (i.e. include delay)
        delay = self.delay(milestone_days.shape)
        sampling_days = milestone_days + delay
        #exclude excessive days 
        exceed_study_duration = sampling_days > LASTVISIT #Will become fake days since can't be sampled
        if np.any(exceed_study_duration): warn("The delay is pushing a sampling day past the study duration.")
        sampling_days[exceed_study_duration] = ma.masked
        
        #make masked values 0 (to not throw an error when using np.take_along_axis)
        mask_temp = sampling_days.mask.copy() #since changing masked values will make them no longer masked
        sampling_days[mask_temp] = 0 #change masked values
        sampling_days[mask_temp] = ma.masked #put back the mask

        #Sample at sampling days
        milestone_scores = {scorename:np.take_along_axis(population.scores[scorename], sampling_days, axis=1) for scorename in population.scores} #both real and fake
        #mask the fake days
        milestone_days = ma.masked_array(sampling_days, fill_value=NDAYS) #Fake days will give index errors if used
        milestone_scores = {scorename: ma.masked_array(milestone_scores[scorename], sampling_days.mask, fill_value=np.nan) for scorename in milestone_scores}

        # Population
        samplepop = population.copy(addtitle='\nsampled by '+self.title)
        samplepop.days = milestone_days
        samplepop.scores = {scorename:milestone_scores[scorename] for scorename in samplepop.scores}
        return samplepop
    
class RealisticMethodology(Methodology):
    '''
    Will help understand how to implement a sequential model for Methodology
    by checking how it cannibalizes code from other methodologies
    '''

    def __init__(self, title=''):
        #title
        super().__init__(title=title)
            
        #organizing
        self.methods = []
        self.methods.append({
            'order': 0, #Should be set automatically
            'name': 'traditional',
            'day': 0,
            'delay': lambda shape: helper.beta(shape, 7, 28, 14, 2.9), #90% at 21
            'limit': (None, None), #Irrelevant because max(day+delay) < NDAYS
            'if_reached': None #Irrelevant because first sampling method
        })
        self.methods.append({
            'order': 1, #Should be set automatically
            'name': 'smile',
            'index': ('sample', -1),
            'ratio': 0.5,
            'scorename': 'symptom',
            'delay': lambda shape: helper.beta(shape, 0, 14, 4, 3.8), #90% at 7
            'limit': ((('sample', -1), lambda val: val+28), 'clip'),
            'if_reached': None #Irrelevant becaue index is previous sample
        })
        self.methods.append({
            'order': 2, #Should be set automatically
            'name': 'magnitude',
            'value': 6,
            'scorename': 'symptom',
            'delay': lambda shape: helper.beta(shape, 0, 14, 4, 3.8), #same as prev
            'limit': ('NDAYS', 'clip'),
            'if_reached': 'NaN'
        })
        
        # check if methods have necessary entries of correct type
        for method in self.methods:
            #check name
            if method['name'] not in {'traditional', 'smile', 'magnitude'}:
                raise ValueError(f"name of {method['name']} not understood")
            #check delay
            if isinstance(method['delay'], int): 
                method['delay'] = lambda shape: np.full(shape, method['delay'], dtype=int)
            elif callable(delay):
                if not method['delay'].__code__.co_varnames == ('shape',): 
                    raise ValueError("The function for delay generation should only have 'shape' as an argument.")
            else: 
                raise TypeError(f"delay of {delay} is not an int nor is it callable")
            #check limit
            if not isinstance(method['limit'], int):
                raise TypeError(f"limit is not an int in {method}")
            #check if_reached
            if not method['if_reached'] in {None, 'NaN'}:
                raise ValueError(f"if_reached of {method['if_reached']} not known")
            #check method specific parameters
            parameterchecker_func = getattr(self, '_check_parameters_'+method['name'])
            parameterchecker_func(method)
    
    @staticmethod
    def _check_parameters_traditional(method):
        if not isinstance(method['day'], int):
                    raise TypeError("Sampling day must be int")
        if method['day'] >= NDAYS:
            raise ValueError(f"day of {method['day']} is later than the simulation duration of {NDAYS}")
        if method['day'] > LASTVISIT:
            warn(f"day of {method['day']} is later than the LASTVISIT of {LASTVISIT}")
        if method['day'] < FIRSTVISIT:
                    warn(f"day of {method['day']} is earlier than the FIRSTVISIT of {FIRSTVISIT}")
    #TODO let index be a function of previous sample
    @staticmethod
    def _check_parameters_smile(method):
        #set index_day as callable
        if isinstance(method['index'], int):
            method['index'] = lambda shape, prev_sampling_days: np.full(shape, method['index'], dtype=int)
        elif isinstance(method['index'], tuple): #check if refers to previous sample
            if len(method['index']) == 2 and method['index'][0] == 'sample':
                if isinstance(method['index'][1], int):
                        if -method['order'] < method['index'][1] <= -1:
                            method['index'] = lambda shape, prev_sampling_days: prev_sampling_days[:, method['index'][1]]
                        else:
                            raise ValueError("index reference of {method['index'][1]} does not refer to a previous sample"
                                             "i.e. it is not negative or it is negative but too large")
                else:
                    raise TypeError(f"index reference has value {method['index'][1]} which is not an int")
            else: 
                raise ValueError(f"index tuple of {method['index']} is defined wrong. "
                                 "It should have length 2 and it's first value should be the string 'sample'")
        #check if properly set as callable
        if callable(method['index']):
            if method['index'].__code__.co_varnames != ('shape', 'prev_sampling_days'): 
                raise ValueError("The function for index day generation should only have 'shape' and 'prev_sampling_days' as an argument.")
        else:
            raise TypeError(f"index of {method['index']} is not an int, a tuple, nor is it callable")
        #check ratio
        if not (0 < method['ratio'] < 1): 
            warn(f"ratio of {method['ratio']} may be unobtainable.")
        #check scorename
        if method['scorename'] not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"scorename of {method['scorename']} not understood")
    @staticmethod
    def _check_parameters_magnitude(method):
        #check scorename
        if method['scorename'] not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"scorename of {method['scorename']} not understood")
        #check value
        if method['value'] < get_MIN(method['scorename']): 
            warn(f"value of {method['value']} may be unobtainable since it is smaller than "
                 f"the MIN of {get_MIN([method['scorename']])}")
    
    @property
    def nmethods(self): return len(self.methods)
        
    #TODO use maskedarray functions, e.g. np.where should be ma.where 
    def sample_population(self, population):
        #contains all days which will be sampled for all persons
        sampling_days = ma.empty((population.npersons, self.nmethods), dtype=int)
        #TODO set sampling_days.fill_value
        
        #populate sampling_days according to the methods
        for i, method in enumerate(self.methods):
            
            #get day each person calls in
            if method['name'] == 'traditional':
                sampling_days[:,i] = method['day']
            elif method['name'] == 'smile':
                smilescores = population.scores[method['scorename']] #scores which the method ratio refers to
                smilescore_lowerbound = get_MIN(method['scorename'])
                
                #get and check index days
                index_days = method['index']((population.npersons,), sampling_days[:,i])
                #TODO check if not outside NDAYS, FIRSTVISIT, LASTVISIT
                
                # Compute the scores which will trigger milestones
                smilescores_at_index = np.take_along_axis(smilescores, helper.to_vertical(index_days), axis=1) #column array
                smile_vals = (smilescores_at_index - smilescore_lowerbound)*method['ratio'] + smilescore_lowerbound #column array
                
                # Compute the days where the milestones are triggered
                sampling_days_temp = np.argmax(smilescores <= smile_vals, axis=1) #the day at which the milestone is reached for each person
                sampling_days[:,i] = sampling_days_temp #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
                persons_reached_milestone = np.take_along_axis(smilescores <= smile_vals, 
                                                               helper.to_vertical(sampling_days_temp), 
                                                               axis=1) #record of which persons reached the milestones
                sampling_days[~persons_reached_milestone.flatten(), i] = 2**16 #arbitrary but distinct to represent 'unreached' #TODO classattribute
                sampling_days[~persons_reached_milestone.flatten(), i] = ma.masked
                if not np.all(persons_reached_milestone): warn("There is at least one person who didn't reach his milestone")
            elif method['name'] == 'magnitude':
                smilescores = population.scores[method['scorename']] #scores which the method value refers to
                smilescore_lowerbound = get_MIN(method['scorename'])
                
                # Compute the days where the milestones are triggered
                sampling_days_temp = np.argmax(smilescores <= method['value'], axis=1) #the day at which the milestone is reached for each person
                sampling_days[:,i] = sampling_days_temp #the day at which the milestone is reached for each person, inc. 0 for 'never reached'
                persons_reached_milestone = np.take_along_axis(smilescores <= method['value'], 
                                                               helper.to_vertical(sampling_days_temp), 
                                                               axis=1) #record of which persons reached the milestones
                sampling_days[~persons_reached_milestone.flatten(), i] = 2**16+1 #arbitrary but distinct to represent 'unreached' #TODO classattribute
                sampling_days[~persons_reached_milestone.flatten(), i] = ma.masked
                if not np.all(persons_reached_milestone): warn("There is at least one person who didn't reach his milestone")
            else:
                raise ValueError(f"name of {method['name']} not known")
                
            #add delay
            delay_gen = method['delay']
            sampling_days[:,i] += delay_gen((population.npersons,))
            
            #limit
            #TODO parse earlier, simplify, generalize 
            #TODO check if shouldn't includes delay
            limitval, limitbehaviour = method['limit'] #unpack
            #default
            if limitval is None:
                limitval = LASTVISIT
            if limitbehaviour is None:
                limitbehaviour = 'raise'
            #check limit
            if isinstance(limitval, int):
                #replacement value is arbitrary but distinct to represent 'reached' #TODO classattribute
                sampling_days[:,i] = np.where(sampling_days[:,i] > limitval, sampling_days[:,i], 2**16+2)
                #TODO mask
            elif isinstance(limitval, tuple): #check if refers to previous sample
                #TODO type and value checks, like for index of smile
                prev_val, func = limitval #unpack
                limitval = lambda shape, prev_sampling_days: prev_sampling_days[:, method['index'][1]]
                raise Exception("Not implemented yet")
                #TODO finish
            #act on limit
            if limitbehaviour == 'raise':
                if np.any(sampling_days[:,i] == 2**16+2): #same value as just above #TODO classattribute
                    raise IndexError("Reached limit") #TODO better error message
            if limitbehaviour == 'clip':
                sampling_days[:,i] = np.where(sampling_days[:,i] == 2**16+2, limitval ,sampling_days[:,i])
                #TODO implement when limitval is a function
            if limitbehaviour == 'NaN':
                raise Exception("Not implemented yet")
                #TODO or not TODO: could just be default and implemented later (when setting scores)
            #TODO add ('replace', replaceval) as a limitbehaviour option (where 'clip would be a special case')
            
            #TODO check if_reached
                
        #use sampling_days to return a sampled_population
        #TODO implement
        sampled_population = population.copy(addtitle='\nsampled by '+self.title)
        sampled_population.days = sampled_population.days #TODO change
        sampled_population.scores = sampled_population.scores #TODO change
        return sampled_population
    
#TODO optimize
class MixedMethodology(Methodology):
    '''
    Sampling according to many methodologies at once
    Careful,  may oversample some points
    '''
    def __init__(self, methodologies, title=''):
        warn("MixedMethodology may oversample some points\n"
             "e.g. if a SmileMethodology's index_day is also a sampling day of a TraditionalMethodology\n"
             "it will be sampled twice")
        
        super().__init__(title)
        
        #TODO prevent some oversampling by checking index_days and sampling_days
        
        self.methodologies = methodologies
        
    def sample_population(self, population):
        #one population per sampling methodology
        samplepops = PopulationList([methodology.sample_population(population) 
                                     for methodology in self.methodologies])
        
        #combine all sampled populations together
        samplepop = population.copy(addtitle='\nsampled by '+self.title)
        samplepop.days = ma.hstack(samplepops.days)
        samplepop.scores = {scorename:ma.hstack(scorevalues) for (scorename, scorevalues) in samplepops.dict_scores.items()}
        return samplepop
    
    def __getattr__(self, attrname):
        '''returns the attribute from any and all contained methodologies, or raises an AttributeError'''
        methodologies_attributes = {}
        for methname, meth in self.methodologies.items():
            try:
                methodologies_attributes[methname] = meth.__getattribute__(attrname)
            except AttributeError:
                pass
            
        if len(methodologies_attributes) == 0:
            raise AttributeError()
        else:
            return methodologies_attributes