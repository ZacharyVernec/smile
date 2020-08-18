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
        
    def sample(self, pop_or_poplist, filter_args=None):
        #if population is a PopulationList, apply the single-population version to all
        if isinstance(pop_or_poplist, PopulationList):
            return self.sample_populationlist(pop_or_poplist, filter_args=filter_args)
        elif isinstance(pop_or_poplist, Population):
            return self.sample_population(pop_or_poplist, filter_args=filter_args)
        else:
            raise TypeError(f"{pop_or_poplist} is a {type(pop_or_poplist)} when it should be a {Population} or a {PopulationList}.")
                
    def sample_populationlist(self, poplist, filter_args=None):
        sampled_poplist = PopulationList([self.sample_population(pop, filter_args=filter_args) for pop in poplist])
        
        #setting title
        if len(set(sampled_poplist.titles)) == 1: #if only one unique title
            sampled_poplist.title = sampled_poplist.titles[0] #use that unique title
        else: #can only be unspecific
            sampled_poplist.title='sampled by '+self.title
            
        return sampled_poplist
            
    @abstractmethod
    def sample_population(self, population, filter_args=None):
        #should have:
        #    #possible filtering
        #    if filter_args is not None: population = population.filter(**filter_args, copy=True)   
        #as first two lines
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
        
    def sample_population(self, population, filter_args=None):
        #possible filtering
        if filter_args is not None: population = population.filter(**filter_args, copy=True)

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
    def __init__(self, title='', index_day=0, delay=0, milestone_ratios=[0.7, 0.4], smile_scorename='symptom'):
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
            
    def sample_population(self, population, filter_args=None):
        #possible filtering
        if filter_args is not None: population = population.filter(**filter_args, copy=True)
        
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

        # Compute the days where the milestones are triggered #TODO use a masked array (possible complication using take_along_axis)
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
        
        #include index_days as a real day
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
    
#TODO last milestone is based on absolute symptom score
#TODO filter based on pecentage and absolute symptom score
class RealisticMethodology(SmileMethodology):
    '''Sampling similarly to what a real clinician would want, with simulated delay to get appointment'''
    
    def __init__(self, title='realistic methodology', index_day=0, milestone_ratios=[0.5], smile_scorename='symptom', 
                 min_delay=0, max_delay=21, mode=7, a=2.2):
        #a = 2.2 is determined numerically so that with default mode, the 90th percentile is at 21
        
        delay_generator = lambda shape: helper.beta(shape, min_delay, max_delay, mode, a).astype('int')
        super().__init__(title=title, index_day=index_day, delay=delay_generator, milestone_ratios=milestone_ratios, smile_scorename=smile_scorename)
    
class MixedMethodology(Methodology):
    '''Sampling at fixed days and at milestones (NOT UP TO DATE)'''
    def __init__(self, traditional_kwargs, smile_kwargs, title=''):
        warn("Mixed methodology only works with a traditional part and a smile part")
        super().__init__(title)
        
        #if a smilemethodology's index_day is also a sampling day of a traditionalmethodology, don't sample it twice
        try: traditional_kwargs['sampling_days'].copy().remove(smile_kwargs['index_day'])
        except ValueError: pass
        
        #add titles if not given
        if traditional_kwargs['title'] is None:
            traditional_kwargs['title'] = self.title+' (traditional part)'
        if smile_kwargs['title'] is None:
            smile_kwargs['title'] = self.title+' (smile part)'
        
        self.methodologies = {'traditional':TraditionalMethodology(**traditional_kwargs),
                              'smile':SmileMethodology(**smile_kwargs)}
        
    @classmethod
    def from_methodologies(cls, trad_meth, smile_meth, title=''):        
        return cls(traditional_kwargs=trad_meth.__dict__, smile_kwargs=smile_meth.__dict__, title=title)
    
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
        
    def sample_population(self, population, filter_args=None):
        #possible filtering
        if filter_args is not None: population = population.filter(**filter_args, copy=True)
            
        samplepops = PopulationList([methodology.sample_population(population, filter_args=None) 
                                     for methodology in self.methodologies.values()], 
                                    title='all samples')
        
        samplepop = population.copy(addtitle='\nsampled by '+self.title)
        samplepop.days = ma.hstack(samplepops.days)
        samplepop.scores = {scorename:ma.hstack(scorevalues) for (scorename, scorevalues) in samplepops.dict_scores.items()}
        return samplepop