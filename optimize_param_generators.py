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



# Methods to get probabilities of early & late recovery
def get_tails_empir(R, V_0, size=10000):
    '''
        Calculate tails by empirical ratio

        Parameters:
            R: frozen distribution for rate random variable
            V_0: frozen distribution for initial value random variable
            size: amount of points in simulation

        Returns:
            early_prob: probability of -R*t+V_0 hitting 6 before day 7
            late_prob: probability of -R*t+V_0 hitting 6 after day 159
    '''
    #Random variates
    r = R.rvs(size=size)
    v_0 = V_0.rvs(size=size)

    # Simulate to get probabilities
    early_prob = np.mean(-r*7 + v_0 <= 6) #TODO use smile.global_params
    late_prob = np.mean(-r*159 + v_0 > 6) #TODO use smile.global_params

    return early_prob, late_prob
def get_tails_integ(R, V_0):
    '''
        Calculate tails by integrating joint pdf

        Parameters:
            R: frozen distribution for rate random variable
            V_0: frozen distribution for initial value random variable
            size: amount of points in simulation

        Returns:
            early_prob: probability of -R*t+V_0 hitting 6 before day 7
            late_prob: probability of -R*t+V_0 hitting 6 after day 159
    '''

    # Integrate to get probabilities based on distributions
    # The true formula is a double integral, but the first integral simply calculates a cdf or sf=1-cdf
    early_prob, _ = integrate.quad(lambda x: R.sf((x-6)/7) * V_0.pdf(x), 14, 25)
    late_prob, _ = integrate.quad(lambda x: R.cdf((x-6)/159) * V_0.pdf(x), 14, 25)

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
    parser.add_argument('--supportV0', type=float, nargs=2, default=(14, 25),
                        help='Support interval for the init value random var V_0 (default 14.0 25.0)')
    parser.add_argument('--initparamsR', type=float, nargs=5, default=(0.5, 0.3, 1.7, 0.6, 0.6),
                        help='Initial mixture distribution parameters (default: 0.5 0.3 1.7 0.6 0.6)')
    parser.add_argument('--initparamsV0', type=float, nargs=2, default=(18, 6),
                        help='Initial beta distribution parameters (default: 18.0 6.0)')
    parser.add_argument('--lowerboundsR', type=float, nargs=5, default=(1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
                        help='Lower bounds for mixture parameters during optimization '
                             '(default: 1e-8 1e-8 1e-8 1e-8 1e-8)')
    parser.add_argument('--lowerboundsV0', type=float, nargs=2, default=(1e-8, 1e-8),
                        help='Lower bounds for beta parameters during optimization (default: 1e-8 1e-8)')
    parser.add_argument('--upperboundsR', type=float, nargs=5, default=(1, np.inf, np.inf, np.inf, np.inf),
                        help='Lower bounds for mixture parameters during optimization '
                             '(default: 1.0, np.inf np.inf np.inf np.inf)')
    parser.add_argument('--upperboundsV0', type=float, 
                        nargs=2, default=(np.inf, np.inf),
                        help='Lower bounds for beta parameters during optimization (default: np.inf np.inf)')
    parser.add_argument('--maxiter', type=int, default=7*200, #Numb_of_params * 200
                        help='Max number of solver iterations (default: 1400)')
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
    args.bounds = optimize.Bounds(np.array([*args.lowerboundsR, *args.lowerboundsV0]), 
                                  np.array([*args.upperboundsR, *args.upperboundsV0]))
    # Initparams to array
    args.initparams = np.array([*args.initparamsR, *args.initparamsV0])


    # Function to optimize 
    def optim_fun(params, counter):
        # Random variables for visual score defined in simulating_and_pickling as -R*t+V_0
        R = Mixture(*args.supportR, params[0], params[1:3], params[3:5])
        V_0 = BoundedNormal(*args.supportV0, *params[-2:])

        tails = get_tails(R, V_0)

        # Print
        if args.printfreq > 0 and counter.val % args.printfreq == 0: 
            print(f"Recovery tails: {tails}")
        counter.inc()

        return error_fun(tails, args.targets)

    #Minimize
    counter = Counter()
    res = optimize.minimize(optim_fun, args.initparams, bounds=args.bounds,
        args=(counter,), 
        method=args.solver, options={'maxiter':args.maxiter},
        tol=args.tol)

    #Result
    print(res)
    R = Mixture(*args.supportR, res.x[0], res.x[1:3], res.x[3:5])
    V_0 = BoundedNormal(*args.supportV0, *res.x[-2:])
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
    axes[0].set_title("PDF of R \nParams: " + ", ".join(f"{param:g}" for param in res.x[:-2]))
    #V_0
    x = np.linspace(*args.supportV0, 5000)
    y = V_0.pdf(x)
    axes[1].plot(x, y)
    axes[1].set_title("PDF of V_0 \nParams: " + ", ".join(f"{param:g}" for param in res.x[-2:]))

    plt.tight_layout()
    plt.show()