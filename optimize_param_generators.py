from git import Reference
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(20220609)

# To access number of function evals in optim_fun
class Counter:
    def __init__(self):
        self.val = 0
    def inc(self):
        self.val += 1


# New distribution
class BoundedNormal:
    '''Creates frozen instance of truncnorm but with bounds independent of loc and scale'''
    def __new__(cls, lower, upper, loc=0, scale=1):
        if np.any(np.atleast_1d(lower) > np.atleast_1d(upper)):
            raise ValueError()
        a = (lower - loc) / scale
        b = (upper - loc) / scale
        return stats.truncnorm(a, b, loc=loc, scale=scale)

# New distribution
class Mixture:
    """Mixture of Gaussians, partially imitates stats.rv_continuous"""
    def __init__(self, lower, upper, mix, locs, scales):
        if not self._argcheck(mix, locs, scales):
            raise ValueError("bad parameters")
        self.mix = np.array([*np.atleast_1d(mix), 1-np.sum(mix)])
        self.distribs = [BoundedNormal(lower, upper, loc=loc, scale=scale) for loc, scale in zip(locs, scales)]
        
    def _argcheck(self, mix, locs, scales):
        mix = np.atleast_1d(mix)
        dims_ok = (mix.ndim == 1) and (len(mix)+1 == len(locs) == len(scales))
        mix_ok = np.all(mix >= 0) and np.sum(mix) <= 1
        locs_ok = np.all(np.isfinite(locs))
        scales_ok = np.all(scales > 0) and np.all(np.isfinite(scales))
        return dims_ok and mix_ok and locs_ok and scales_ok
    
    def rvs(self, size=1, random_state=None):
        #flatten size but store as 'shape' for returning reshaped
        shape = size
        size = np.prod(shape)
        
        indices = stats.rv_discrete(values=(range(len(self.mix)), self.mix)).rvs(size=size, random_state=random_state)
        norm_variates = [distrib.rvs(size=size, random_state=random_state) for distrib in self.distribs]
        return np.choose(indices, norm_variates).reshape(shape)
        
    def pdf(self, x):
        return np.average([distrib.pdf(x) for distrib in self.distribs], axis=0, weights=self.mix)
    def cdf(self, x):
        return np.average([distrib.cdf(x) for distrib in self.distribs], axis=0, weights=self.mix)
    def sf(self, x):
        return np.average([distrib.sf(x) for distrib in self.distribs], axis=0, weights=self.mix)

class ReferenceDistrib:
    '''Create frozen instance of discrete distribution based on real-world data'''
    def __new__(cls):
        unique = np.arange(20+1)
        counts = np.array([0, 34, 38, 26, 32, 30, 12, 14, 12, 12, 12, 7, 8, 5, 5, 5, 3, 1, 0, 3, 3], dtype=int)

        unique = unique[2:] #exclude 0 and 1 values as already healthy
        counts = counts[2:] #correspond
        probs = counts / np.sum(counts)

        return stats.rv_discrete(values=(unique, probs))
    
# Methods to get probabilities of early & late recovery
def get_tails_empir(R, V_0, size=10000):
    '''
        Calculate tails by empirical ratio

        Parameters:
            R: frozen distribution for rate random variable
            V_0: frozen distribution for initial value random variable
            size: amount of points in simulation

        Returns:
            early_prob: probability of -R*t+V_0 hitting 1 before day 7
            late_prob: probability of -R*t+V_0 hitting 1 after day 159
    '''
    #Random variates
    r = R.rvs(size=size)
    v_0 = V_0.rvs(size=size)

    # Simulate to get probabilities
    early_prob = np.mean(-r*7 + v_0 <= 1) #TODO use smile.global_params
    late_prob = np.mean(-r*159 + v_0 > 1) #TODO use smile.global_params

    return early_prob, late_prob
def get_tails_integ(R, V_0):
    '''
        Calculate tails by integrating joint pdf

        Parameters:
            R: frozen distribution for rate random variable
            V_0: frozen distribution for initial value random variable
            size: amount of points in simulation

        Returns:
            early_prob: probability of -R*t+V_0 hitting 1 before day 7
            late_prob: probability of -R*t+V_0 hitting 1 after day 159
    '''

    # Integrate to get probabilities based on distributions
    # The true formula is a weighted sum of integrals, but the integrals simply calculates a cdf or sf=1-cdf
    V0_supported_vals = np.arange(V_0.a, V_0.b+1)
    V0_probs = V_0.pmf(V0_supported_vals)
    early_prob = np.sum(R.sf((V0_supported_vals - 1)/7) * V0_probs)
    late_prob = np.sum(R.cdf((V0_supported_vals - 1)/159) * V0_probs)

    return early_prob, late_prob

def error_fun(probs, targets):
    ''' Square Error'''
    return np.sum((np.array(probs) - np.array(targets))**2)


if __name__ == '__main__':

    # Optimization parameters
    parser = argparse.ArgumentParser(description='Optimize beta distributions R and V_0')
    parser.add_argument('method', choices=['integration', 'simulation'],
                        help='How the probabilities should be calculated (default: integration)')
    parser.add_argument('-n', type=int, default=10000,
                        help='Number of points to use for simulation (default 10 000)')
    parser.add_argument('--printfreq', type=int, default=20,
                        help='How often to print the current probabilities in numb of function evals (default: 20).\n' +
                            '0 means never print')
    parser.add_argument('--targets', type=float, nargs=2, default=(0.3, 0.1),
                        help='Tail probabilities to target. (default: 0.3 0.1)')
    parser.add_argument('--supportR', type=float, nargs=2, default=(0, 2),
                        help='Support interval for the rate random var R (default 0.0 2.0)')
    parser.add_argument('--initparamsR', type=float, nargs=5, default=(0.5, 0.3, 1.7, 0.6, 0.6),
                        help='Initial mixture distribution parameters (default: 0.5 0.3 1.7 0.6 0.6)')
    parser.add_argument('--lowerboundsR', type=float, nargs=5, default=(1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
                        help='Lower bounds for mixture parameters during optimization '
                             '(default: 1e-8 1e-8 1e-8 1e-8 1e-8)')
    parser.add_argument('--upperboundsR', type=float, nargs=5, default=(1, np.inf, np.inf, np.inf, np.inf),
                        help='Lower bounds for mixture parameters during optimization '
                             '(default: 1.0, inf inf inf inf)')
    parser.add_argument('--maxiter', type=int, default=5*200, #Numb_of_params * 200
                        help='Max number of solver iterations (default: 1000)')
    parser.add_argument('--tol', type=float, default=1e-8,
                        help='Solver tolerance for convergence (default: 1e-8)')
    parser.add_argument('--solver', choices=['Nelder-Mead'], default='Nelder-Mead')
    
    # Parse for ease of use in further code
    args = parser.parse_args()
    # Method
    if args.method == 'integration':
        get_tails = get_tails_integ
    elif args.method == 'simulation':
        get_tails = lambda R, V_0: get_tails_empir(R, V_0, size=args.n)
    # Bounds
    args.bounds = optimize.Bounds(np.array(args.lowerboundsR), 
                                  np.array(args.upperboundsR))
    # Initparams to array
    args.initparams = np.array(args.initparamsR)


    # Function to optimize 
    def optim_fun(params, counter):
        # Random variables for visual score defined in simulating_and_pickling as -R*t+V_0
        R = Mixture(*args.supportR, params[0], params[1:3], params[3:5])
        V_0 = ReferenceDistrib()

        tails = get_tails(R, V_0)

        # Print
        if args.printfreq > 0 and counter.val % args.printfreq == 0: 
            print(f"Recovery tails: {tails}")
        counter.inc()

        return error_fun(tails, args.targets)

    #Minimize
    if args.maxiter > 0:
        counter = Counter()
        res = optimize.minimize(optim_fun, args.initparams, bounds=args.bounds,
            args=(counter,), 
            method=args.solver, options={'maxiter':args.maxiter},
            tol=args.tol)
        resultparams = res.x
        print(res)
    else:
        resultparams = args.initparams
        

    #Result
    R = Mixture(*args.supportR, resultparams[0], resultparams[1:3], resultparams[3:5])
    V_0 = ReferenceDistrib()
    tails = get_tails(R, V_0)
    print(f"Recovery tails: {tails}")


    # Show plots of found distributions

    fig, axes = plt.subplots(1,2)
    fig.suptitle('Found distribution for visual score recovery, which is used like -R*t + V_0\n' +
                f'obtaining recovery time tails of {tails[0]:g}, {tails[1]:g}')
    #R
    x = np.linspace(*args.supportR, 5000)
    y = R.pdf(x)
    axes[0].plot(x, y)
    axes[0].set_title("PDF of R \nParams: " + ", ".join(f"{param:g}" for param in resultparams))
    #V_0
    x = np.arange(21)
    y = V_0.pmf(x)
    axes[1].bar(x, y, width=1)
    axes[1].set_title("PMF of V_0")

    plt.tight_layout()
    plt.show()