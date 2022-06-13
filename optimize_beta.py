from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(20220609)
rng = np.random.default_rng(67890)

# To access number of function evals in optim_fun
class Counter:
    def __init__(self):
        self.val = 0
    def inc(self):
        self.val += 1


# Methods to get probabilities of early & late recovery
def get_tails_empir(alpha, beta, eta, gamma, size=10000):
    '''
        Calculate tails by empirical ratio

        Parameters:
            alpha, beta (float > 0, float > 0): parameters of rate rv R
            eta, gamma (float > 0, float > 0): parameters of initial value rv V_0
            size: amount of points in simulation

        Returns:
            early_prob: probability of -R*t+V_0 hitting 6 before day 7
            late_prob: probability of -R*t+V_0 hitting 6 after day 159
    '''
    # Random variables for visual score defined in simulating_and_pickling as -R*t+V_0
    # Have support [loc, loc+scale]
    R = stats.beta(alpha, beta, loc=0, scale=2)
    V_0 = stats.beta(eta, gamma, loc=14, scale=11)
    #Random variates
    r = R.rvs(size=size)
    v_0 = V_0.rvs(size=size)

    # Simulate to get probabilities
    early_prob = np.mean(-r*7 + v_0 <= 6) #TODO use smile.global_params
    late_prob = np.mean(-r*159 + v_0 > 6) #TODO use smile.global_params

    return early_prob, late_prob
def get_tails_integ(alpha, beta, eta, gamma):
    '''
        Calculate tails by integrating joint pdf

        Parameters:
            alpha, beta (float > 0, float > 0): parameters of rate rv R
            eta, gamma (float > 0, float > 0): parameters of initial value rv V_0

        Returns:
            early_prob: probability of -R*t+V_0 hitting 6 before day 7
            late_prob: probability of -R*t+V_0 hitting 6 after day 159
    '''
    # Random variables for visual score defined in simulating_and_pickling as -R*t+V_0
    # Have support [loc, loc+scale]
    R = stats.beta(alpha, beta, loc=0, scale=2)
    V_0 = stats.beta(eta, gamma, loc=14, scale=11)

    # Integrate to get probabilities based on distributions
    # The true formula is a double integral, but the first integral simply calculates a cdf or sf=1-cdf
    early_prob, _ = integrate.quad(lambda x: R.sf((x-6)/7) * V_0.pdf(x), 14, 25)
    late_prob, _ = integrate.quad(lambda x: R.cdf((x-6)/159) * V_0.pdf(x), 14, 25)

    return early_prob, late_prob


# Get which get_tails function to use
parser = argparse.ArgumentParser(description='Optimize beta distributions of visual score parameters')
parser.add_argument('--empirical', type=int, help='instead of integration, simulate with n points')
empirical = parser.parse_args().empirical


# Error function to optimize 
def optim_fun(params, early_target=0.3, late_target=0.1, counter=Counter(), print_frequency=0):
    '''
    Square Error func for fitting previous function

    Keep in mind this problem is underspecified, as len(params)==4 but there are only 2 target probabilities

        Parameters:
            params (tuple): Parameters of the distribution to optimize (alpha, beta, eta, gamma)
            early_target (float): Target for ratio of patients who recover too quickly
            late_target (float): Target for ratio of patients who recover too slowly
            counter (Counter): Keeps track of number of times this function is evaluated
            print_frequency (int): How many function evals between prints. 0 means never print.
    '''
    if empirical is not None:
        early_prob, late_prob = get_tails_empir(*params, size=empirical)
    else:
        early_prob, late_prob = get_tails_integ(*params)

    # Print
    if print_frequency > 0 and counter.val % print_frequency == 0: 
        print(f"Recovery tails: {early_prob, late_prob}")
    counter.inc()

    return (early_prob-early_target)**2 + (late_prob-late_target)**2




initial_params = (5,4,3,2)
bounds = [(1,None)]*4 #so the beta's pdf are unimodal
targets = (0.3, 0.1)
print_frequency = 100
method = 'Nelder-Mead' #Same solver as default in R
res = optimize.minimize(optim_fun, initial_params, bounds=bounds,
                        args=(*targets, Counter(), print_frequency), 
                        method=method, options={'maxiter':8000, 'maxfev':8000})
print(res)
if empirical is not None:
    print(f"Recovery tails: {get_tails_empir(*res.x, size=empirical)}")
else:
    print(f"Recovery tails: {get_tails_empir(*res.x)}")


# Show plots of found distributions

fig, axes = plt.subplots(1,2)
fig.suptitle('Found distribution for visual score recovery, which goes like -R*t + V_0')
#R
x = np.linspace(0,2, 5000)
y = stats.beta(res.x[0], res.x[1], loc=0, scale=2).pdf(x)
axes[0].plot(x, y)
axes[0].set_title("PDF of R")
#V_0
x = np.linspace(14,25, 5000)
y = stats.beta(res.x[2], res.x[3], loc=14, scale=11).pdf(x)
axes[1].plot(x, y)
axes[1].set_title("PDF of V_0")

plt.show()