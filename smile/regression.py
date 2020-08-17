# Standard library imports
from collections import UserList

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
#none

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