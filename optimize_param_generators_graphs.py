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

    #Result Fig 3
    R = Mixture(0, 2, 0.676866, np.array([0.151503, 6.31151]), np.array([0.211144, 0.548655]))
    V_0 = BoundedNormal(14, 25, 16.6942, 2.04876)

    x = np.linspace(0, 8, 10000)

    yR3 = R.rvs(size=10000)
    yV3 = V_0.rvs(size=10000)

    #Result Fig 5
    R = Mixture(0, 8, 0.7035, np.array([0.15, 6.3]), np.array([0.21, 0.54]))
    V_0 = BoundedNormal(14, 25, 16, 2)

    yR5 = R.rvs(size=10000)
    yV5 = V_0.rvs(size=10000)

    #Result Fig 7
    R = Mixture(0, 3, 0.67722, np.array([0.118413, 2.60797]), np.array([0.353518, 0.130295]))
    V_0 = BoundedNormal(14, 25, 19.1255, 9.6916)

    yR7 = R.rvs(size=10000)
    yV7 = V_0.rvs(size=10000)

    # Show plots of found distributions

    fig, axes = plt.subplots(ncols=2, sharey=True)
    fig.suptitle('Found distribution for visual score recovery, which is used like -R*t + V_0\n' +
                f'obtaining recovery time tails of around 0.3, 0.1')
    #R
    axes[0].hist(yR3, bins=20, range=(0,8), alpha=0.4, edgecolor='k', label='A')
    axes[0].hist(yR5, bins=20, range=(0,8), alpha=0.4, edgecolor='k', label='B')
    axes[0].hist(yR7, bins=20, range=(0,8), alpha=0.4, edgecolor='k', label='C')
    axes[0].set_title("Histograms of 10,000 samples of R")
    #V_0
    axes[1].hist(yV3, bins=20, range=(14,25), alpha=0.4, edgecolor='k', label='A')
    axes[1].hist(yV5, bins=20, range=(14,25), alpha=0.4, edgecolor='k', label='B')
    axes[1].hist(yV7, bins=20, range=(14,25), alpha=0.4, edgecolor='k', label='C')
    axes[1].set_title("Histograms of 10,000 samples of V_0")


    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__2':

    #Result Fig 3
    R = Mixture(0, 2, 0.676866, np.array([0.151503, 6.31151]), np.array([0.211144, 0.548655]))
    V_0 = BoundedNormal(14, 25, 16.6942, 2.04876)

    dec = np.linspace(0, 1, 10)

    yR3 = R.ppf(dec)
    yV3 = V_0.ppf(dec)

    #Result Fig 5
    R_ = Mixture(0, 8, 0.7035, np.array([0.15, 6.3]), np.array([0.21, 0.54]))
    V_0 = BoundedNormal(14, 25, 16, 2)

    yR5 = R.ppf(dec)
    yV5 = V_0.ppf(dec)

    # Show plots of found distributions

    fig, axes = plt.subplots(ncols=2, sharey=True)
    fig.suptitle('Found distribution for visual score recovery, which is used like -R*t + V_0\n' +
                f'obtaining recovery time tails of around 0.3, 0.1')
    #R
    axes[0].plot(yR3, dec)
    axes[0].plot(yR5, dec)
    axes[0].set_title("Deciles of Rs")
    axes[0].xlabel('values of R')
    axes[1].ylabel('probability of having at most such value')
    #V_0
    axes[1].plot(yV3, dec)
    axes[1].plot(yV5, dec)
    axes[1].set_title("Deciles of V_0s")
    axes[1].xlabel('values of V_0')
    axes[1].ylabel('probability of having at most such values')

    plt.tight_layout()
    plt.show()