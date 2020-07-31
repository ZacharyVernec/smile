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
    def regress_mixed(self, x='visual', y='symptom'):
        '''Mixed effects linear regression on self, with random intercept and slope'''
        # Argument parsing # TODO make into helper function for clutter reduction
        y_possibilities = {'symptom'} #TODO add more possibilities
        x_possibilities = {'visual'} #TODO add more possibilities
        if y not in y_possibilities:
            raise ValueError('Dependent variable {} not recognized. Use one of {} instead.'.format(y, y_possibilities))
        if x not in x_possibilities:
            raise ValueError('Independent variable {} not recognized. Use one of {} instead.'.format(x, yx_possibilities))
            
        df = self.to_dataframe()
        #check for NaN, will decide later if should be dropped when specifying model
        null_count = df.isnull().sum().sum()
        if null_count > 0: 
            warn('Population {} has {} NaN values'.format(self.title, null_count))
            
        #regress
        model = smf.mixedlm(y+" ~ "+x, df, groups=df['person'], re_formula='~'+x) 
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
    def filter(self, recovered_symptom_score=SMIN, firstday=FIRSTVISIT, lastday=NDAYS, copy=False):
        if copy==False: pop=self
        elif copy==True: pop=self.copy(addtitle='filtered')
        else: raise ValueError()
        
        persons_recovered_early = np.any(pop.scores['symptom'][:,:firstday] <= recovered_symptom_score, axis=1)
        persons_recovered_late = np.min(pop.scores['symptom'][:,:lastday], axis=1) > recovered_symptom_score
        persons_excluded = np.logical_or(persons_recovered_early, persons_recovered_late)
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
        return RegressionResultList([pop.regress_mixed(*kwargs) for pop in self], title=self.title+'\nMixed effects regression')
            
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
        if title==None: title='list of '+population.title
        return cls([population.copy() for i in range(length)], title)
    def to_dataframes(self):
        return [pop.to_dataframe() for pop in self]
    
    def filter(self, recovered_symptom_score=SMIN, firstday=FIRSTVISIT, lastday=NDAYS, copy=False):
        if copy==False: poplist=self
        elif copy==True: poplist=self.copy(addtitle='filtered')
        else: raise ValueError()
        
        poplist.data = [pop.filter(recovered_symptom_score=recovered_symptom_score, firstday=firstday, lastday=lastday, copy=copy) for pop in poplist]
        
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
    
    def get_biases(self, ground_truths, magnitude=False, relative=False):
        '''
        Returns bias of the estimates
        Ground truth is either:
            - a list-like of floats (inc. np.NaN for unknown) of same length as number of params,
            - a dict-like of floats (inc. np.NaN) with keys as the param names
        Magnitude determines whether to return the absolute value of the biases
        Relative determinves whether to return the percentage biases
        '''
        param_means = self.params_dataframe.mean()
        param_truths = pd.Series(ground_truths)
        #set index if necessary
        if isinstance(ground_truths, (list, tuple, np.ndarray)):
            param_truths.index = param_means.index #same order as params
        #calculate biases
        biases = param_means - param_truths
        if relative: 
            biases = biases / param_truths * 100 #percent
            biases.index = [paramname+" (%)" for paramname in biases.index] #add percentage symbol to parameter text
        if magnitude: 
            biases = np.abs(biases)
        return biases
        
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
    
class Methodology:
    '''deprecated'''
    def __init__(self, title='', sample_args=[8, 15, 29], smiledelay=0, smilescorename='symptom'):
        '''
        Sample args that are integers are interpreted as a fixed day
        Sample args between 0 and 1 (exclusive) are interpreted as a percentage with which to use SMILE, based on the first fixed day
        Sample args that do not fit the above criteria are ignored
        smiledelay can be an int or a random function with a single 'shape' parameter which will be used to get the value for each person and milestone
        smilescore determines which score the milestone_ratios will be based on (can be symptom, visual, or symptom_noerror)
        '''
        
        #TODO possibility of random smiledelay, random index
        
        self.title=title
        self.smilescorename=smilescorename
        
        #split sample_args into fixed and smile
        sample_args = np.array(sample_args)
        self.fixed_days = sample_args[sample_args == sample_args.astype(int)].astype(int)
        self.milestone_ratios = sample_args[(0 < sample_args) & (sample_args < 1)]
        #checking fixed_days for errors
        if self.fixed_days.size==0: 
            raise ValueError("No fixed days in sample_args, which were {}".format(sample_args))
        if np.max(self.fixed_days) >= NDAYS:
            raise IndexError("There is a fixed sample day in {} that is later than the LASTVISIT of {}".format(self.fixed_days, NDAYS))
        if np.max(self.fixed_days) > LASTVISIT:
            warn("There is a fixed sample day in {} that is later than the LASTVISIT of {}".format(self.fixed_days, LASTVISIT))
        if np.min(self.fixed_days) < FIRSTVISIT:
            warn("There is a fixed sample day in {} that is earlier than the FIRSTVISIT of {}".format(self.fixed_days, FIRSTVISIT))
        
        #set smiledelay_generator as callable #TODO merge into SmileMethodology
        if isinstance(smiledelay, int): 
            self.smiledelay_generator = lambda shape: smiledelay
        elif callable(smiledelay):
            if smiledelay.__code__.co_varnames == ('shape',): 
                self.smiledelay_generator = smiledelay
            else: 
                raise ValueError("The function for smiledelay generation should only have 'shape' as an argument.")
        else: 
            raise ValueError("smiledelay is not an int nor is it callable")
        
        
    # Statistical
    
    def sample(self, pop_or_poplist, filter_args=None):
        #if population is a PopulationList, apply the single-population version to all
        if isinstance(pop_or_poplist, PopulationList):
            poplist = pop_or_poplist #renaming variable
            return PopulationList([self.sample(pop, filter_args=filter_args) for pop in poplist], title='sampled by '+self.title)
        else:
            population = pop_or_poplist #renaming variable
            
            #possible filtering
            if filter_args is not None: population = population.filter(**filter_args, copy=True)

            # MILESTONES        

            smilescores = population.scores[self.smilescorename]

            # Compute the scores which will trigger milestones
            index = int(np.min(self.fixed_days)) #day which milestone_ratios are based on #cast as int rather than np.int for easy type checks
            smilescores_at_index = helper.to_vertical(smilescores[:, index])
            smile_vals = smilescores_at_index*self.milestone_ratios 
            #smile_vals are the score values to reach. Each row is a person, each column is an ordinal, and each value is the score the reach

            #TODO use a masked array (possible complication using take_along_axis)
            # Compute the days where the milestones are triggered, and where we sample
            milestone_days = np.empty_like(smile_vals, dtype=int) #will hold the day each milestone_ratio is reached for each person
            #careful: values of 0 in milestone_days might represent 'day 0' or might represent 'never reached milestone'. The following array will help mitigate this
            milestone_days_real = np.empty_like(milestone_days, dtype=bool) #will hold the days where the milestone_ratios is reached (exc. those stored as 0 meaning 'never reached')
            for milestone_col in range(smile_vals.shape[1]): 
                milestone_vals = helper.to_vertical(smile_vals[:,milestone_col])
                milestone_days_temp = np.argmax(smilescores <= milestone_vals, axis=1) #the day at which the milestone is reached for each person
                milestone_days[:,milestone_col] = milestone_days_temp #the day at which the milestone is reached for each person
                milestone_days_real[:,milestone_col] = np.take_along_axis(smilescores <= milestone_vals, helper.to_vertical(milestone_days_temp), axis=1).flatten()
            if not np.all(milestone_days_real): warn("There is a milestone that was not reached.")
            sampling_days = np.where(milestone_days_real, milestone_days+self.smiledelay, milestone_days)  #fake days stay as 0
            exceed_study_duration = sampling_days > LASTVISIT #Will become fake days since can't be sampled
            if np.any(exceed_study_duration): warn("The smiledelay is pushing a sampling day past the study duration.")
            milestone_days_real = np.where(exceed_study_duration, False, milestone_days_real)
            sampling_days = np.where(exceed_study_duration, 0, sampling_days) #get sent to 0 like other fake days

            #Sample at real sampling days
            milestone_smilescores = np.take_along_axis(smilescores, sampling_days, axis=1) #both real and fake
            milestone_scores = {scorename:np.take_along_axis(population.scores[scorename], sampling_days, axis=1) for scorename in population.scores} #both real and fake
            #replace the 'fake' days and scores with NaN
            milestone_days = np.where(milestone_days_real, sampling_days, NDAYS) #Fake days will give index errors
            milestone_scores = {scorename: np.where(milestone_days_real, milestone_scores[scorename], np.nan) for scorename in milestone_scores} #Fake days will be NaN

            # FIXED

            fixed_days = np.tile(self.fixed_days, (smilescores.shape[0],1)) #same shape as milestone_days
            fixed_scores = {scorename:np.take_along_axis(population.scores[scorename], fixed_days, axis=1) for scorename in population.scores}

            # COMBINE fixed and milestones

            samplepop = population.copy(addtitle='\nsampled by '+self.title)
            samplepop.days = np.hstack([fixed_days, milestone_days])
            samplepop.scores = {scorename:np.hstack([fixed_scores[scorename], milestone_scores[scorename]]) for scorename in samplepop.scores}

            return samplepop
                 
    #TODO compare_analysis classmethod
    
    # Other methods
    
    def plot(self, ax, max_day=NDAYS):
        #titles and labels
        ax.set_title(self.title, wrap=True)
        ax.set(xlabel='fixed days', ylabel='smile ratios')
        
        #plotting #TODO use ax.axhline and ax.axvline
        for fixed_day in self.fixed_days:
            ax.plot([fixed_day, fixed_day], [0.0, 1.0], color='black')
        for milestone_ratio in self.milestone_ratios:
            ax.plot([0, max_day], [milestone_ratio, milestone_ratio], color='black')
        ax.autoscale()
        
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
        
        self.index_day = index_day
        if not isinstance(index_day, int):
            raise TypeError("Index day must be an int")
        if index_day >= NDAYS:
            raise ValueError(f"The index day {index_day} is later than the simulation duration of {NDAYS}")
        if index_day > LASTVISIT:
            warn(f"The index day {index_day} is later than the LASTVISIT of {LASTVISIT}")
        if index_day < FIRSTVISIT:
            warn(f"The index day {index_day} is earlier than the FIRSTVISIT of {FIRSTVISIT}")
            
        self.delay = delay
            
        self.milestone_ratios = milestone_ratios
        if not all(0 < ratio < 1 for ratio in milestone_ratios): 
            warn(f"Some milestone_ratios in {milestone_ratios} may be unobtainable.")
            
        self.smile_scorename = smile_scorename
        if smile_scorename not in {'symptom', 'visual', 'symptom_noerror'}:
            raise ValueError(f"smile_scorename of {smile_scorename} not understood")
            
    #TODO consider VMIN and SMIN
    def sample_population(self, population, filter_args=None):
        #possible filtering
        if filter_args is not None: population = population.filter(**filter_args, copy=True)
        
        smilescores = population.scores[self.smile_scorename] #scores which the ratios refer to
        #TODO simplify retrieval of MINs
        if self.smile_scorename == 'visual': smilescore_lowerbound = VMIN
        elif self.smile_scorename == 'symptom' or self.smile_scorename == 'symptom_noerror': smilescore_lowerbound = SMIN

        # Compute the scores which will trigger milestones
        smilescores_at_index = helper.to_vertical(smilescores[:, self.index_day])
        smile_vals = (smilescores_at_index - smilescore_lowerbound)*self.milestone_ratios + smilescore_lowerbound
        #smile_vals are the score values to reach. Each row is a person, each column is an ordinal, and each value is the score the reach

        #TODO use a masked array (possible complication using take_along_axis)
        # Compute the days where the milestones are triggered, and where we sample
        milestone_days = np.empty_like(smile_vals, dtype=int) #will hold the day each milestone_ratio is reached for each person
        #careful: values of 0 in milestone_days might represent 'day 0' or might represent 'never reached milestone'. The following array will help mitigate this
        milestone_days_real = np.empty_like(milestone_days, dtype=bool) #will hold the days where the milestone_ratios is reached (exc. those stored as 0 meaning 'never reached')
        for milestone_col in range(smile_vals.shape[1]): 
            milestone_vals = helper.to_vertical(smile_vals[:,milestone_col])
            milestone_days_temp = np.argmax(smilescores <= milestone_vals, axis=1) #the day at which the milestone is reached for each person
            milestone_days[:,milestone_col] = milestone_days_temp #the day at which the milestone is reached for each person
            milestone_days_real[:,milestone_col] = np.take_along_axis(smilescores <= milestone_vals, helper.to_vertical(milestone_days_temp), axis=1).flatten()
        if not np.all(milestone_days_real): warn("There is a milestone that was not reached.")
        sampling_days = np.where(milestone_days_real, milestone_days+self.delay, milestone_days)  #fake days stay as 0
        exceed_study_duration = sampling_days > LASTVISIT #Will become fake days since can't be sampled
        if np.any(exceed_study_duration): warn("The delay is pushing a sampling day past the study duration.")
        milestone_days_real = np.where(exceed_study_duration, False, milestone_days_real)
        sampling_days = np.where(exceed_study_duration, 0, sampling_days) #get sent to 0 like other fake days
        
        #include index_days as a real day
        index_days = np.full_like(sampling_days, fill_value=self.index_day, shape=(population.npersons, 1))
        index_days_real = np.full_like(milestone_days_real, fill_value=True, shape=(population.npersons, 1))
        sampling_days = np.hstack([index_days, milestone_days])
        milestone_days_real = np.hstack([index_days_real, milestone_days_real])
        

        #Sample at real sampling days
        milestone_smilescores = np.take_along_axis(smilescores, sampling_days, axis=1) #both real and fake
        milestone_scores = {scorename:np.take_along_axis(population.scores[scorename], sampling_days, axis=1) for scorename in population.scores} #both real and fake
        #replace the 'fake' days and scores with NaN
        milestone_days = np.where(milestone_days_real, sampling_days, NDAYS) #Fake days will give index errors
        milestone_scores = {scorename: np.where(milestone_days_real, milestone_scores[scorename], np.nan) for scorename in milestone_scores} #Fake days will be NaN

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
        samplepop.days = np.hstack(samplepops.days)
        samplepop.scores = {scorename:np.hstack(scorevalues) for (scorename, scorevalues) in samplepops.dict_scores.items()}
        return samplepop