#TODO: printing methods
#TODO: use Enums (?)
#TODO return self in population methods for chaining
#TODO remove unnecessary kwargs defaults (eg Population() should return an empty population, not one with size 10000)
#TODO black lines when plotting 'both' with 'day'
#TODO Parameter_generator class for easy printing?
#TODO make sure population can't be re-generated after being sampled
#TODO raise error if generate() is called with generate_parameters=False but no parameters had been generated before either
#TODO don't return filtered Population nor filtered PopulationList if copy=False
#TODO improve plotting syntax and design with seaborn
#TODO replace all numpy data with pandas dataframe
#TODO create fig and axes by default if no axes given for plotting methods
#TODO documentation: https://realpython.com/documenting-python-code/
#TODO possibility of slicing Population with scorename as first element in subscript tuple
#TODO keep track of expected value of parameters for use when plotting regression results
#TODO shape doesn't *need* to be the variable name for random, as long as there is 1


# Standard library imports
from collections import UserList
from copy import copy
from warnings import warn
from abc import ABC, abstractmethod #abstract base class

# Third party imports
import numpy as np
import numpy.ma as ma
from numpy import random
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
import smile.helper; reload(smile.helper)
from smile import helper
import smile.global_params; reload(smile.global_params)
from smile.global_params import *



# # Population


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
            self.parameters[paramname] = self.parameter_generators[paramname]()
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
            warn("visual score in {} has min={}, which is below VMIN={}".format(self.title, minscore, VMIN))
        minsymptomscore = np.min(self.scores['symptom'], initial=SMIN) #initial arg to avoid error of min on empty array
        if minsymptomscore < SMIN: 
            warn("symptom score in {} has  min={}, which is below SMIN={}".format(self.title, minscore, SMIN))
            
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
            
        #regress
        if random_effect == 'intercept':
            model = smf.mixedlm(f' {y}~{x} ', df, groups=df['person']) 
        elif random_effect == 'slope':
            model = smf.mixedlm(f' {y}~{x} ', df, groups=df['person'], re_formula=f' ~{x}+0') 
        elif random_effect == 'both':
            model = smf.mixedlm(f' {y}~{x} ', df, groups=df['person'], re_formula=f' ~{x}') 
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
        newpop.scores = {scorename:np.array(helper.twodarray(newpop.scores[scorename])[subscript]) #slice as twodarray but keep as ndarray
                         for scorename in newpop.scores}
        newpop.days = np.array(helper.twodarray(newpop.days)[subscript]) #slice as twodarray but keep as ndarray
        return newpop
    def to_dataframe(self):
        data_dict = {
            'person': np.broadcast_to(np.arange(self.npersons), (self.ndays, self.npersons)).T, # same shape matrix as days or scores, with values that indicate person index
            'day': self.days,
            **self.scores
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
            if self.scorename == 'visual': recovered_score = VMIN
            elif self.scorename == 'symptom' or self.scorename == 'symptom_noerror': recovered_score = SMIN
        
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
        return RegressionResultList([pop.regress_linear(**kwargs) for pop in self], title=self.title+'\nregressed linear')
    def regress_mixed(self, **kwargs):
        return RegressionResultList([pop.regress_mixed(**kwargs) for pop in self], title=self.title+'\nMixed effects regression')
            
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
    
    def plot(self, axeslist, direction='row', **kwargs):
        '''
        Plots all populations sequentially, using same kwargs as in Population.plot() method
        Direction can be "row" or "col" depending on if the axeslist represents a row or column of a figure
            and determines where the PopulationList title will go
        '''
        #check axes input
        if len(axeslist) != len(self): 
            raise ValueError("{} axes are not enough to plot {} Populations".format(len(axes), len(self)))
        #plot
        for (i, pop) in enumerate(self):
                pop.plot(axeslist[i], **kwargs)
        #PopulationList title
        ax = axeslist[0] #first axis
        if direction=='col':
            pad=30
            color='blue'
            ax.annotate(self.title, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', 
                            textcoords='offset points', size='large', ha='center', va='baseline', color=color)
        elif direction=='row':
            pad=15
            ax.annotate(self.title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0), xycoords=ax.yaxis.label, 
                            textcoords='offset points', size='large', ha='right', va='center', color='blue')
        else: 
            raise ValueError("Unknown direction {}".format(direction))

class RegressionResult:
    '''Wrapper for linear RegressionResults class of statsmodels'''
    def __init__(self, statsmodelRegResult, population):
        self.statsmodelRegResult = statsmodelRegResult
        self.population = population
        
        #slicing to not catch extra params in mixed effects model, e.g. 'Group x visual Cov'
        self.params = statsmodelRegResult.params.iloc[0:2]
        #self.rsquared = statsmodelRegResult.rsquared #Does not exist for linear mixed effects in statsmodels
        
        @property
        def title(self):
            return self.population.title +'\n regression result'
        
    # Statistical methods
    
    def confidence_interval(self, alpha=0.05):
        '''uses a Student t distribution'''
        return self.statsmodelRegResult.conf_int(alpha=alpha)
    
    # Plotting methods
    
    def plot_line(self, ax, alpha=0.05): #TODO make more generate than just visual vs symptom
        x = np.linspace(VMIN, np.max(self.population.scores['visual']), 20)

        y = self.params['visual']*x + self.params['Intercept']

        conf_int = self.confidence_interval(alpha=alpha)
        ylow = conf_int[0]['visual']*x + conf_int[0]['Intercept']
        yhigh = conf_int[1]['visual']*x + conf_int[1]['Intercept']

        ax.plot(x,y, color='b')
        ax.fill_between(x, ylow, yhigh, color='b', alpha=.1)
        
        # Formatting
        title = "Regression of {} \n with {:2.0f}% confidence intervals".format(self.population.title, (1-alpha)*100)
        ax.set_title(title, wrap=True)
        xlabel='visual score'
        ylabel='symptom score'
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.autoscale()
        
    # Other methods
    
    def unwrap(self):
        return self.statsmodelRegResult
        
def assertRegressionResult(obj):
    """Check if object is a RegressionResult"""
    if not isinstance(obj, RegressionResult):
        raise AssertionError("object {} is {} instead of {}".format(obj, type(obj), repr(RegressionResult)))
    else:
        return True #if no error
def assertListlikeOfRegressionResults(listlike):
    """Check if all objects are RegressionResults"""
    if isinstance(listlike, PopulationList):
        return True #short circuit since must contain only Populations
    else:
        for (i, obj) in enumerate(listlike):
            try: 
                assertRegressionResult(obj)
            except AssertionError as e:
                newmessage = "At index {}, {}".format(i, e.args[0])
                #append update error message
                if len(e.args) >= 1:
                    e.args = (newmessage,) + e.args[1:]
                raise #re-raise error with updated message
        return True #if no error

# TODO keep track of number of parameters
class RegressionResultList(UserList):
    def __init__(self, listlike=[], title=''):
        assertListlikeOfRegressionResults(listlike)
        super().__init__(listlike)
        self.title = title
    #overriden methods
    def append(self, other):
        assertRegressionResult(other)
        super().append(other)
    def extend(self, listlike):
        assertListlikeOfRegressionResult(listlike)
        super().extend(listlike)
    def insert(self, i, obj):
        assertRegressionResult(obj)
        super().insert(i, obj)
        
    # Properties which iterate over the respective attributes for all RegressionResults in the list
    
    @property
    def statsmodelRegResults(self): return [regresult.statsmodelRegResult for regresult in self]
    @property
    def populations(self): return [regresult.population for regresult in self]
    @property
    def params(self): return [regresult.params for regresult in self]
    @property
    def params_dataframe(self): return pd.concat(self.params, axis=1).transpose()
    #@property
    #def rsquareds(self): return [regresult.rsquared for regresult in self] #Does not exist for linear mixed effects in statsmodels
    
    # Statistical methods
    
    def get_biases(self, ground_truths):
        '''
        Returns bias of the estimates
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        param_means = self.params_dataframe.mean()
        param_truths = pd.Series(ground_truths)
        #set index if necessary
        if isinstance(ground_truths, (list, tuple, np.ndarray)):
            param_truths.index = param_means.index #same order as params
        #calculate biases
        biases = param_means - param_truths
        biases.index = biases.index.copy(name='Biases') #So it isn't shared between return values of different statistical methods
        return biases
    
    def get_percentage_biases(self, ground_truths):
        '''
        Returns bias of the estimates
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        biases = self.get_biases(ground_truths)
        param_truths = pd.Series(ground_truths)
        #set index if necessary
        if isinstance(ground_truths, (list, tuple, np.ndarray)):
            param_truths.index = biases.index #same order as params
        biases = biases / param_truths * 100
        biases.index = [paramname+" (%)" for paramname in biases.index] #add percentage symbol to parameter text
        biases.index = biases.index.copy(name='Percentage Biases') #So it isn't shared between return values of different statistical methods
        return biases
    
    def get_mses(self, ground_truths):
        '''
        Returns mean square errors of the estimates
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        params = self.params_dataframe
        param_truths = pd.Series(ground_truths)
        #set index if necessary
        if isinstance(ground_truths, (list, tuple, np.ndarray)):
            param_truths.index = params.columns #same order as params
        #calculate mean square errors
        square_errors = params.sub(param_truths, axis='columns')**2
        mses = square_errors.mean()
        mses.index = mses.index.copy(name='MSEs') #So it isn't shared between return values of different statistical methods
        return mses
    
    def get_rmses(self, ground_truths):
        '''
        Returns the root mean square errors of the estimates
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        rmses = self.get_mses(ground_truths)**0.5
        rmses.index = rmses.index.copy(name='RMSEs') #So it isn't shared between return values of different statistical methods
        return rmses
    
    def get_vars(self, ground_truths):
        '''
        Returns the variance of the estimates, calculated using the formula mse = var - bias^2
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        variance = self.get_mses(ground_truths) - self.get_biases(ground_truths)**2
        variance.index = variance.index.copy(name="Variances") #So it isn't shared between return values of different statistical methods
        return variance
    
    def get_stdevs(self, ground_truths):
        '''
        Returns the variance of the estimates, calculated from the variances
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        stdevs = self.get_vars(ground_truths)**0.5
        stdevs.index = stdevs.index.copy(name="Std Devs") #So it isn't shared between return values of different statistical methods
        return stdevs
    
    def get_stderrs(self, ground_truths):
        '''
        Returns the Standard Error of the estimates, calculated using the variance
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        '''
        stderr = self.get_stdevs(ground_truths)/len(self)**0.5
        stderr.index = stderr.index.copy(name="Std Errors") #So it isn't shared between return values of different statistical methods
        return stderr
        
    def get_sample_stdevs(self, ddof=1):
        '''
        Returns the sample standard deviations of the estimates
        ddof is the delta dof (degrees of freedom), defined by the formula dof = n - ddof
        '''
        sample_stdevs = self.params_dataframe.std(ddof=ddof)
        sample_stdevs.index = sample_stdevs.index.copy(name="Sample Std Devs") #So it isn't shared between return values of different statistical methods
        return sample_stdevs    
    
    def get_sample_stderrs(self, ddof=1):
        '''
        Returns the Standard Error of the estimates, calculated using the standard deviations
        ddof is the delta dof (degrees of freedom), defined by the formula dof = n - ddof
        '''
        stderr = self.get_sample_stdevs(ddof=ddof)/len(self)**0.5
        stderr.index = stderr.index.copy(name="Sample Std Errors") #So it isn't shared between return values of different statistical methods
        return stderr
    
        
        
    # Plotting methods
    
    # TODO add title
    def plot_box(self, axeslist, ground_truths=None, direction='row'):
        '''
        Ground truth is either None, or a list of floats (inc. np.Nan for unknown) of same length as number of params
        Direction can be "row" or "col" depending on if the axeslist represents a row or column of a figure
            and determines where the PopulationList title will go
        '''
        #vars
        params_df = self.params_dataframe
        paramnames = params_df.columns
        #check input
        if len(axeslist) != len(paramnames):
            raise ValueError("Not enough axes to plot each parameter.")
        if ground_truths is not None:
            if len(ground_truths) != len(paramnames):
                raise ValueError("Not enough ground truths for each parameter.")
        #plotting
        for i in range(len(paramnames)):
            #Box plot
            boxprops = dict(linewidth=2, color='blue')
            medianprops = dict(linewidth=2, color='blue')
            meanlineprops = dict(linestyle=':', linewidth=2, color='green')
            params_df[paramnames[i]].plot.box(ax=axeslist[i], grid=False, 
                                              boxprops=boxprops, medianprops=medianprops, 
                                              meanprops=meanlineprops, meanline=True, showmeans=True)
            #Ground truth line
            if ground_truths is not None: 
                axeslist[i].axhline(ground_truths[i], xmin=0.0, xmax=1.0, linewidth=1, color='red')
        #RegressionList title
        ax = axeslist[0] #first axis
        if direction=='col':
            pad=30
            color='blue'
            ax.annotate(self.title, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', 
                            textcoords='offset points', size='large', ha='center', va='baseline', color=color)
        elif direction=='row':
            pad=15
            ax.annotate(self.title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0), xycoords=ax.yaxis.label, 
                            textcoords='offset points', size='large', ha='right', va='center', color='blue')
        else: 
            raise ValueError("Unknown direction {}".format(direction))
            
    # Other methods
    
    def unwrap(self): return [result.unwrap() for result in self]
            

# # Study


#TODO check score percentage not only first dip, but multiple consecutive days
        
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
        return PopulationList([self.sample_population(pop, filter_args=filter_args) for pop in poplist], 
                              title='sampled by '+self.title)
            
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
        #TODO simplify retrieval of MINs
        if self.smile_scorename == 'visual': smilescore_lowerbound = VMIN
        elif self.smile_scorename == 'symptom' or self.smile_scorename == 'symptom_noerror': smilescore_lowerbound = SMIN
            
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
    
class MixedMethodology(Methodology):
    '''Sampling at fixed days and at milestones'''
    def __init__(self, traditional_kwargs, smile_kwargs, title=''):
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