from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(20220609)
rng = np.random.default_rng(67890) #TODO USE

# To access number of function evals in optim_fun
class Counter:
    def __init__(self):
        self.val = 0
    def inc(self):
        self.val += 1




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
    parser.add_argument('--locscaleR', type=float, nargs=2, default=(0, 2),
                        help='location-scale pair for the rate random var R (default 0.0 2.0)')
    parser.add_argument('--locscaleV0', type=float, nargs=2, default=(14, 11),
                        help='location-scale pair for the init value random var V_0 (default 14.0 11.0)')
    parser.add_argument('--initparams', type=float, nargs=4, default=(1.3, 1.05, 2, 6),
                        help='Initial beta distribution parameters (default: 1.3 1.05 2.0 6.0')
    parser.add_argument('--lowerbounds', type=float, nargs=4, default=(1e-8, 1e-8, 1e-8, 1e-8),
                        help='Lower bounds for beta parameters during optimization (default: 1e-8 1e-8 1e-8 1e-8)')
    parser.add_argument('--upperbounds', type=lambda s: None if s is None else float(s), 
                        nargs=4, default=(None, None, None, None),
                        help='Lower bounds for beta parameters during optimization (default: None None None None)')
    parser.add_argument('--maxiter', type=int, default=4*200, #Numb_of_params * 200
                        help='Max number of solver iterations (default: 800)')
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
    # Location and scales
    args.locR, args.scaleR = args.locscaleR
    args.locV0, args.scaleV0 = args.locscaleV0
    # Bounds
    args.bounds = zip(args.lowerbounds, args.upperbounds)
    # Initparams to array
    args.initparams = np.array(args.initparams)


    # Function to optimize 
    def optim_fun(params, counter):
        # Random variables for visual score defined in simulating_and_pickling as -R*t+V_0
        # Have support [loc, loc+scale]
        R = stats.beta(*params[:2], loc=args.locR, scale=args.scaleR)
        V_0 = stats.beta(*params[2:], loc=args.locV0, scale=args.scaleV0)

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
    R = stats.beta(*res.x[:2], loc=args.locR, scale=args.scaleR)
    V_0 = stats.beta(*res.x[2:], loc=args.locV0, scale=args.scaleV0)
    tails = get_tails(R, V_0)
    print(f"Recovery tails: {tails}")


    # Show plots of found distributions

    fig, axes = plt.subplots(1,2)
    fig.suptitle('Found distribution for visual score recovery, which is used like -R*t + V_0\n' +
                f'obtaining recovery time tails of {tails[0]:g}, {tails[1]:g}')
    #R
    x = np.linspace(args.locR, args.locR+args.scaleR, 5000)
    y = R.pdf(x)
    axes[0].plot(x, y)
    axes[0].set_title(f"PDF of R \nParams: {res.x[0]:g}, {res.x[1]:g}")
    #V_0
    x = np.linspace(args.locV0, args.locV0+args.scaleV0, 5000)
    y = V_0.pdf(x)
    axes[1].plot(x, y)
    axes[1].set_title(f"PDF of V_0 \nParams: {res.x[2]:g}, {res.x[3]:g}")

    plt.tight_layout()
    plt.show()