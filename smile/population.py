# Standard library imports
from collections import UserList
from copy import copy
import warnings
from abc import ABC, abstractmethod #abstract base class

# Third party imports
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Reload library
from importlib import reload

# Local application imports
import smile.regression; reload(smile.regression)
from smile.regression import RegressionResult, RegressionResultList
import smile.helper; reload(smile.helper)
from smile import helper
import smile.global_params; reload(smile.global_params)
from smile.global_params import VMIN, SMIN, NDAYS, FIRSTVISIT, LASTVISIT

#custom warnings
def warn(message):
    '''Standard UserWarning but without showing extra information (filename, lineno, line)'''
    default_formatwarning = warnings.formatwarning
    def custom_formatwarning(msg, category, filename, lineno, line=None): 
        return default_formatwarning(msg, category, filename='', lineno='', line='')
    warnings.formatwarning = custom_formatwarning
    warnings.warn(message)
    warnings.formatwarning = default_formatwarning



class Population:
    def __init__(self, npersons=10000, title=''):
        
        self.title = title
        self.initial_data_shape = (self.initial_npersons, self.initial_ndays) = (npersons, NDAYS)
        
        self.parameter_generators = {}
        self.parameters = {}
        self.function_generators = {}
        
        self.days = np.tile(np.arange(self.initial_ndays), (self.initial_npersons,1))
        self.scores = {'visual':None, 
                       'symptom_noerror':None, 
                       'symptom':None}
        
    @property
    def npersons(self): return self.days.shape[0]
    @property
    def ndays(self): return self.days.shape[1]
    @property
    def data_shape(self): return self.days.shape
    @property
    def nfiltered(self): return self.initial_npersons - self.npersons
    @property
    def ratio_filtered(self):
        try:
            return self.nfiltered/self.initial_npersons
        except ZeroDivisionError:
            return 0
    @property
    def nsampled(self): return self.initial_ndays - self.ndays
    @property
    def ratio_sampled(self):
        try:
            return self.nsampled/self.initial_ndays
        except ZeroDivisionError:
            return 0
        
    # Data generation methods

    def set_parameter_generator(self, paramname, func, paramtype):
        '''
        func must be a numpy function with the "shape" argument of the numpy function as the only argument to func
        paramtype must be either "population", "person", or "day"
        '''
        if func.__code__.co_varnames != ('shape',):
            raise ValueError("The function for parameter generation should only have 'shape' as an argument. "+
                             "Currently, it's arguments are: "
                            +", ".join(["'"+str(arg)+"'" for arg in func.__code__.co_varnames]))
        
        if paramtype == 'population':
            shape = (1,)
        elif paramtype == 'person':
            shape = (self.npersons, 1)
        elif paramtype == 'day':
            shape = self.days.shape # also equivalent to self.data_shape
        self.parameter_generators[paramname] = lambda: func(shape)
    def generate_parameters(self):
        '''
        generates all parameters
        if population was already generated, is resets the scores (since they were based on previous parameters)
        '''
        #generate
        for paramname in self.parameter_generators:
            self.parameters[paramname] = np.array(self.parameter_generators[paramname]())
        #reset previous scores
        self.scores = {scorename:None for scorename in self.scores}   
    
    def set_score_generator(self, scorename, func):
        '''scorename is either visual, symptom_noerror, or symptom'''
        if scorename not in self.scores: warn("Scorename '{}' not known. Known options are: {}".format(scorename, self.scores.keys()))
        self.function_generators[scorename] = func
    def generate_from_score_generator(self, scorename):
        '''scorename is either visual, symptom_noerror, or error'''
        try:
            func = self.function_generators[scorename]
        except KeyError as e:
            warn("There is no generator for '{}' score attached to this Population.".format(scorename))
            func = lambda: np.array([])
        paramnames = func.__code__.co_varnames
        paramvals = []
        for paramname in paramnames:
            #reserved parameters
            if paramname == 't' or paramname == 'day':
                paramvals.append(self.days)
            elif paramname == 'v' or paramname == 'visual':
                paramvals.append(self.scores['visual'])
            elif paramname == 's' or paramname == 'symptom_noerror':
                paramvals.append(self.scores['symptom_noerror'])
            #custom parameters
            else:
                paramvals.append(self.parameters[paramname])
        return func(*paramvals)
    
    #def generate_parameters(self) is above
    def generate(self, generate_parameters=True):
        if generate_parameters: self.generate_parameters()
        
        for scorename in self.scores: #cannot be done by dict comprehension since later dict values depend on previous ones
            self.scores[scorename] = self.generate_from_score_generator(scorename).astype(float)
        
        minvisualscore = np.min(self.scores['visual'], initial=VMIN) #initial arg to avoid error of min on empty array
        if minvisualscore < VMIN: 
            warn("visual score in {} has min={}, which is below VMIN={}".format(self.title, minvisualscore, VMIN))
        minsymptomscore = np.min(self.scores['symptom'], initial=SMIN) #initial arg to avoid error of min on empty array
        if minsymptomscore < SMIN: 
            warn("symptom score in {} has  min={}, which is below SMIN={}".format(self.title, minsymptomscore, SMIN))
            
        #if all parameters are 'population', the generation process will only have created a single row
        #so, repeat that row 'self.npersons' times to create the full matrix            
        for scorename in self.scores:
            if self.scores[scorename].shape != self.data_shape and self.scores[scorename].size > 0: #if score array is wrong shape but nonzero
                self.scores[scorename] = np.broadcast_to(self.scores[scorename], self.data_shape) #change shape by broadcasting
            elif self.scores[scorename].size == 0: #if score array is empty
                self.scores[scorename] = self.scores[scorename].reshape(self.data_shape) #change shape by adding empty axes
                
    # Statistical methods
    
    #TODO more complex cases than linear
    #TODO remove repetition between functions
    def regress(self, method='population', y='symptom', x='visual'):
        #TODO use getattr(self, 'regress_'+method)
        if method == 'persons': return self.regress_persons(x=x, y=y)
        elif method == 'population': return self.regress_population(x=x, y=y)
        elif method =='mixed': return self.regress_mixed(x=x, y=y)
        else: raise ValueError("Unknown regression method: {}".format(method))
    def regress_persons(self, x='visual', y='symptom'):
        '''Simple linear regression on each person in self, independently'''
        warn('Deprecated')
        poplist = self.to_populationlist()
        #regress each person
        regresults = poplist.regress_populations(y=y, x=x)
        return regresults #TODO return as Result not Resultslist
    def regress_linear(self, x='visual', y='symptom'):
        '''Simple linear regression on self'''
        # Argument parsing # TODO make into helper function for clutter reduction
        y_possibilities = {'symptom'} #TODO add more possibilities
        x_possibilities = {'visual'} #TODO add more possibilities
        if y not in y_possibilities:
            raise ValueError('Dependent variable {} not recognized. Use one of {} instead.'.format(y, y_possibilities))
        if x not in x_possibilities:
            raise ValueError('Independent variable {} not recognized. Use one of {} instead.'.format(x, yx_possibilities))
            
        y, X = dmatrices(y+' ~ '+x, data=self.to_dataframe(), return_type='dataframe') #split into endogenous and exogenous
        model = sm.OLS(y, X) #define model
        result = model.fit() #fit model
        
        return RegressionResult(result, self)
    def regress_mixed(self, x='visual', y='symptom', random_effect='both'):
        '''Mixed effects linear regression on self, with random intercept and slope
        random_effect can be 'intercept', 'slope', or 'both'
        '''
        
        # Argument parsing # TODO make into helper function for clutter reduction
        y_possibilities = {'symptom'} #TODO add more possibilities
        x_possibilities = {'visual'} #TODO add more possibilities
        if y not in y_possibilities:
            raise ValueError('Dependent variable {} not recognized. Use one of {} instead.'.format(y, y_possibilities))
        if x not in x_possibilities:
            raise ValueError('Independent variable {} not recognized. Use one of {} instead.'.format(x, x_possibilities))
            
        df = self.to_dataframe()
        #check for NaN, will decide later if should be dropped when specifying model
        null_count = df.isnull().sum().sum()
        if null_count > 0: 
            warn('Population {} has {} NaN values'.format(self.title, null_count))
        missing='drop'
            
        #regress
        if random_effect == 'intercept':
            model = smf.mixedlm(f' {y}~{x} ', df, groups=df['person'], missing=missing) 
        elif random_effect == 'slope':
            model = smf.mixedlm(f' {y}~{x} ', df, groups=df['person'], re_formula=f' ~{x}+0', missing=missing) 
        elif random_effect == 'both':
            model = smf.mixedlm(f' {y}~{x} ', df, groups=df['person'], re_formula=f' ~{x}', missing=missing) 
        else:
            raise ValueError(f"random_effect of {random_effect} not understood")
        #TODO check notes of https://www.statsmodels.org/stable/generated/statsmodels.formula.api.mixedlm
        result = model.fit() #fit model
        
        return RegressionResult(result, self)
    
    # Other methods
        
    def copy(self, newtitle=None, addtitle=None):
        '''not fully deep, but allows re-generation and filtering'''
        newpop = copy(self) #python shallow copy
        if newtitle is not None:
            newpop.title = newtitle
        if addtitle is not None:
            newpop.title += ' '+addtitle
        newpop.parameter_generators = copy(self.parameter_generators)
        newpop.function_generators = copy(self.function_generators)
        newpop.parameters = copy(self.parameters)
        newpop.days = copy(self.days)
        newpop.scores = copy(self.scores)
        return newpop
    def double(self, newtitle1=None, addtitle1=None, newtitle2=None, addtitle2=None):
        newpop2 = self.copy(newtitle=newtitle2, addtitle=addtitle2)
        newpop1 = self
        if newtitle1 is not None:
            newpop1.title = newtitle1
        if addtitle1 is not None:
            newpop1.title += ' '+addtitle1
        return newpop1, newpop2
    def __getitem__(self, subscript): #for slicing like a numpy 2d array of (persons, days)
        '''
        Returns a new population by slicing the days and scores as specified
        (in a numpy-like fashion)
        keeping arrays two-dimensional
        '''
        newpop = self.copy()
        newpop.parameters = {paramname:np.array(helper.twodarray(paramval)[subscript]) if paramval.ndim > 0 else paramval #slice as twodarray but keep as ndarray
                             for paramname, paramval in newpop.parameters.items()}
        newpop.scores = {scorename:np.array(helper.twodarray(scoreval)[subscript]) #slice as twodarray but keep as ndarray
                         for scorename, scoreval in newpop.scores.items()}
        newpop.days = np.array(helper.twodarray(newpop.days)[subscript]) #slice as twodarray but keep as ndarray
        return newpop
    def to_dataframe(self):
        data_dict = {
            'person': np.broadcast_to(np.arange(self.npersons), (self.ndays, self.npersons)).T, # same shape matrix as days or scores, with values that indicate person index
            'day': ma.filled(self.days),
            **{scorename: ma.filled(scoreval) for scorename, scoreval in self.scores.items()}
        }
        dataflat_dict = {dataname: data.flatten() for (dataname,data) in data_dict.items()}
        df = pd.DataFrame(dataflat_dict)
        df.index.name = 'observation'
        df.name = self.title
        return df
    def to_populationlist(self):
        return PopulationList([self[i] for i in range(self.npersons)], title=self.title+' (as PopulationList)')
    
    #removing outliers
    def filter(self, copy=False, scorename='symptom', recovered_score=None, firstday=FIRSTVISIT, lastday=NDAYS, drop_na=False):
        if copy==False: pop=self
        elif copy==True: pop=self.copy(addtitle='filtered')
        else: raise ValueError()
            
        #TODO simplify retrieval of MINs
        if recovered_score is None:
            if scorename == 'visual': recovered_score = VMIN
            elif scorename == 'symptom' or scorename == 'symptom_noerror': recovered_score = SMIN
        
        persons_recovered_early = np.any(pop.scores[scorename][:,:firstday] <= recovered_score, axis=1)
        persons_recovered_late = np.min(pop.scores[scorename][:,:lastday], axis=1) > recovered_score
        persons_with_na = np.any(np.isnan(pop.scores[scorename]), axis=1)
        
        persons_excluded = np.logical_or(persons_recovered_early, persons_recovered_late)
        if drop_na: persons_excluded = np.logical_or(persons_excluded, persons_with_na)
        persons_included = np.logical_not(persons_excluded)

        #take only the included and recalculate size
        pop.scores = {scorename:pop.scores[scorename][persons_included] for scorename in pop.scores}
        pop.days = pop.days[persons_included]
        
        return pop #may be self or a copy
    
    #plotting
    def plot(self, ax, ndays=None, npersons=None, x='day', y='symptom', viztype='lines', vizcolor='person'):
        #x and y are either 'day' or a scorename (either 'visual', 'symptom_noerror', or 'symptom')
        #viztype is either 'lines', 'points', or 'both'
        #vizcolor is either 'person' (each person is a color) or 'day' (each day is a color)
        #TODO raise exception for non-existant viztype
        
        if ndays is None: ndays=self.ndays
        if npersons is None: npersons=self.npersons
            
        if (viztype=='lines' or viztype=='both') and vizcolor=='day':
            warn('vizcolor of "day" can only be applied to points, not lines')
        if vizcolor not in {'person', 'day'}:
            raise ValueError()
        if viztype not in {'lines', 'points', 'both'}:
            raise ValueError()
            
        #abscissas
        if x=='day':
            x = self.days[:npersons, :ndays]
            xlabel='days since concussion'
        elif x in self.scores:
            xlabel = x+' scores'
            x = self.scores[x][:npersons, :ndays]
        else:
            raise ValueError()
            
        #ordinates
        if y=='day':
            x = self.days[:npersons, :ndays]
            ylabel='days since concussion'
        elif y in self.scores:
            ylabel = y+' scores'
            y = self.scores[y][:npersons, :ndays]
        else:
            raise ValueError()
            
        #titles and labels
        ax.set_title(self.title, wrap=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        
        #plotting
        if viztype=='lines' or viztype=='both':
            points = np.stack([x, y], axis=2)
            colors = mpl.cm.get_cmap(lines_cmap_name).colors # https://matplotlib.org/2.0.1/users/colormaps.html
            ax.add_collection(LineCollection(points, colors=colors))
            
        if viztype=='points' or viztype=='both':
            if vizcolor == 'person':
                colors = np.array(mpl.cm.get_cmap(lines_cmap_name).colors) # not the right shape
                colors = helper.rgblist_to_rgbapop(colors, npersons, ndays)
                colors = colors.reshape(npersons*ndays, 4) #scatter converts the 2d arrays x and y to flat arrays, and colors should respect that flatness
            elif vizcolor == 'day':
                colors = self.days[:npersons, :ndays]
                cmap = mpl.cm.get_cmap(points_cmap_name) # https://matplotlib.org/2.0.1/users/colormaps.html
                colors = cmap(helper.normalize(colors)) # converts scalars to rgba
                colors = colors.reshape(npersons*ndays, 4) #scatter converts the 2d arrays x and y to flat arrays, and colors should respect that flatness
            else:
                raise ValueError("vizcolor of '{}' unknown".format(vizcolor))
            ax.scatter(x, y, facecolors='none', edgecolors=colors)
        ax.autoscale()

#The following are useful for defining the PopulationList class

def assertPopulation(obj):
    """Check if object is a Population"""
    if not isinstance(obj, Population):
        raise AssertionError("object {} is {} instead of {}".format(obj, type(obj), repr(Population)))
    else:
        return True #if no error
def assertListlikeOfPopulations(listlike):
    """Check if all objects are Populations"""
    if isinstance(listlike, PopulationList):
        return True #short circuit since must contain only Populations
    else:
        for (i, obj) in enumerate(listlike):
            try: 
                assertPopulation(obj)
            except AssertionError as e:
                newmessage = "At index {}, {}".format(i, e.args[0])
                #append update error message
                if len(e.args) >= 1:
                    e.args = (newmessage,) + e.args[1:]
                raise #re-raise error with updated message
        return True #if no error

class PopulationList(UserList):
    def __init__(self, listlike=[], title=''):
        assertListlikeOfPopulations(listlike)
        super().__init__(listlike)
        self.title = title
    #overriden methods
    def append(self, other):
        assertPopulation(other)
        super().append(other)
    def extend(self, listlike):
        assertListlikeOfPopulations(listlike)
        super().extend(listlike)
    def insert(self, i, obj):
        assertPopulation(obj)
        super().insert(i, obj)
        
    # Properties which iterate over the respective attributes for all Populations in the list
    
    @property
    def titles(self): return [pop.title for pop in self]
    @property
    def npersons(self): return [pop.npersons for pop in self]
    @property
    def ndays(self): return [pop.ndays for pop in self]
    @property
    def data_shapes(self): return [pop.data_shape for pop in self]
    @property
    def nfiltered(self): return [pop.nfiltered for pop in self]
    @property
    def ratio_filtered(self): return [pop.ratio_filtered for pop in self]
    @property
    def nsampled(self): return [pop.nsampled for pop in self]
    @property
    def ratio_sampled(self): return [pop.ratio_sampled for pop in self]
    @property
    def parameter_generators(self): return [pop.parameter_generators for pop in self]
    @property
    def parameters(self): return [pop.parameters for pop in self]
    @property
    def function_generators(self): return [pop.function_generators for pop in self]
    @property
    def days(self): return [pop.days for pop in self]
    @property
    def score_dicts(self):
        '''list of the "scores" dict of all populations'''
        return [pop.scores for pop in self]
    @property
    def dict_scores(self):
        '''dict where each entry is a scoretype, and each value is a list of that score for each population'''
        return {scorename:[pop.scores[scorename] for pop in self] for scorename in ['visual', 'symptom_noerror', 'symptom']} #TODO change iterator
    
    # Data generation methods
    
    def generate_parameters(self):
        for pop in self: pop.generate_parameters()
    def generate(self, generate_parameters=True):
        for pop in self: pop.generate(generate_parameters=generate_parameters)
            
    # Statistical methods
            
    def regress(self, method='population', **kwargs):
        #TODO use getattr(self, 'regress_'+method)
        if method == 'persons': return self.regress_persons(**kwargs)
        elif method == 'population': return self.regress_population(**kwargs)
        elif method =='mixed': return self.regress_mixed(**kwargs)
        else: raise ValueError("Unknown regression method: {}".format(method))
    def regress_persons(self, **kwargs):
        #deprecated
        return [pop.regress_persons(**kwargs) for pop in self]
    def regress_linear(self, **kwargs):
        return RegressionResultList([pop.regress_linear(**kwargs) for pop in self], title=self.title+'\nregressed with linear')
    def regress_mixed(self, **kwargs):
        return RegressionResultList([pop.regress_mixed(**kwargs) for pop in self], title=self.title+'\nregressed with mixed effects')
            
    # Other methods
    
    def copy(self, newtitle=None, addtitle=None):
        #copy
        newlist = PopulationList([pop.copy(newtitle=newtitle, addtitle=addtitle) for pop in self], title=self.title)
        #possibly change title
        if newtitle is not None:
            newlist.title = newtitle
        if addtitle is not None:
            newlist.title += ' '+addtitle
        #return
        return newlist
    def double(self, newtitle1=None, addtitle1=None, newtitle2=None, addtitle2=None):
        newpoplist2 = self.copy(newtitle=newtitle2, addtitle=addtitle2)
        newpoplist1 = self
        if newtitle1 is not None:
            #change list title
            newpoplist1.title = newtitle1
            #change population titles
            for newpop1 in newpoplist1: newpop1.title = newtitle1
        if addtitle1 is not None:
            #change list title
            newpoplist1.title += ' '+addtitle1
            #change population titles
            for newpop1 in newpoplist1: newpop1.title += ' '+addtitle1
        return newpoplist1, newpoplist2
    @classmethod
    def full(cls, length, population, title=None):
        '''Creates a PopulationList of many copies of the given population'''
        if title is None: title='list of '+population.title
        return cls([population.copy() for i in range(length)], title)
    def to_dataframes(self):
        return [pop.to_dataframe() for pop in self]
    
    def filter(self, copy=False, **kwargs):
        if copy==False: poplist=self
        elif copy==True: poplist=self.copy(addtitle='filtered')
        else: raise ValueError()
        
        poplist.data = [pop.filter(copy=copy, **kwargs) for pop in poplist]
        
        return poplist #may be self or a copy
    
    def plot(self, ax, direction='row', **kwargs):
        '''
        Plots all populations sequentially, using same kwargs as in Population.plot() method
        If ax corresponds to a single axis, all are plotted on it and the title is the PopulationList title
        If ax is an ndarray with an axis per population, then each is plotted on each and direction determines how to put the PopulationList title
        Direction can be "row", "col", or "none" depending on if the axarr represents a row or column of a figure
            and determines where the PopulationList title will go ("none" means the title isn't displayed)
        '''
        axarr = np.array(ax)
        
        if axarr.size == 1: #plot all on the same
            axis = axarr.item()
            for pop in self:
                pop.plot(axis, **kwargs)
            axis.set_title(self.title, wrap=True)
            
        elif axarr.size == len(self): #plot all sequentially
            for (i, pop) in enumerate(self):
                pop.plot(axarr.flat[i], **kwargs)
            #PopulationList title
            ax = axarr[0] #first axis
            if direction == 'col':
                pad=30
                color='blue'
                ax.annotate(self.title, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', 
                                textcoords='offset points', size='large', ha='center', va='baseline', color=color)
            elif direction == 'row':
                pad=15
                ax.annotate(self.title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0), xycoords=ax.yaxis.label, 
                                textcoords='offset points', size='large', ha='right', va='center', color='blue')
            elif direction != 'none': 
                raise ValueError("Unknown direction {}".format(direction))
            
        else:
            raise ValueError(f"{len(axes)} axes are not enough to plot {len(self)} Populations")