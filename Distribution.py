import numpy as np
from scipy.stats import norm
import random

# make normal distributions for all arms, where best arm has mean "best_mean" and all others "rest_means",
# all arms have std=0.1
def make_normal_dists(n_arms, best_mean, rest_means):
    dists = []
    for arm in range(n_arms):
        dists.append(DistNormal.forMean(rest_means))
    best_arm = random.choice(np.arange(n_arms))
    dists[best_arm] = DistNormal.forMean(best_mean)
    return dists, best_arm

# make normal distributions for all arms, where all non-optimal arms have random mean in [lower_bound, upper_bound] with
# std in [lower_scale_bound, upper_scale_bound]
# and best arm has mean = max mean of other arms + diff and random std in [lower_scale_bound, upper_scale_bound]
def make_random_normal_dists(n_arms, diff, lower_bound, upper_bound, lower_scale_bound, upper_scale_bound):
    dists = []
    max_mean = lower_bound
    for arm in range(n_arms):
        mean = np.random.uniform(lower_bound, upper_bound)
        scale = np.random.uniform(lower_scale_bound, upper_scale_bound)
        dists.append(DistNormal.forMean(mean, scale=scale))
        if mean > max_mean:
            max_mean = mean
    best_arm = random.choice(np.arange(n_arms))
    scale = np.random.uniform(lower_scale_bound, upper_scale_bound)
    dists[best_arm] = DistNormal.forMean(max_mean + diff, scale=scale)
    borda_winner = best_arm
    while borda_winner == best_arm:
        borda_winner = random.choice(np.arange(n_arms))
        borda_scale = np.random.uniform(lower_scale_bound, upper_scale_bound)
    return dists, best_arm, borda_winner, max_mean, borda_scale

# make normal distributions for arms in one query set q
# all arms have random mean in [lower_bound, upper_bound] with std in [lower_scale_bound, upper_scale_bound]
# best arm has mean = max mean of other arms in q + diff and random std in [lower_scale_bound, upper_scale_bound]
def make_random_normal_dists_for_q(partition, dists, best_arm, diff, lower_bound, upper_bound, lower_scale_bound, upper_scale_bound):
    max_mean = lower_bound
    for arm in partition:
        mean = np.random.uniform(lower_bound, upper_bound)
        scale = np.random.uniform(lower_scale_bound, upper_scale_bound)
        dists[arm] = DistNormal.forMean(mean, scale=scale)
        if mean > max_mean:
            max_mean = mean
    scale = np.random.uniform(lower_scale_bound, upper_scale_bound)
    dists[best_arm] = DistNormal.forMean(max_mean + diff, scale=scale)
    return dists, max_mean

class Distribution:

    def sample(self, n=1):
        raise Exception("Sample method is not overwritten by class.")

    def getMean(self):
        return self.mean

    def getVar(self):
        return self.var

    def getLabel(self):
        return self.label

    def getDensity(self, X):
        raise NotImplementedError("getDensity is not implemented.")

    def getQuantile(self, q):  # default implementation: check this empirically
        densities = self.getDensity(np.linspace(-10, 10, 10000))
        totalDensity = sum(densities)
        cum = 0
        for i, p in enumerate(np.linspace(-10, 10, 10000)):
            cum += densities[i]
            if cum / totalDensity >= q:
                return p

    def getGeneralizedMean(self, X):
        gMeans = []
        sample_sequence = self.sample(1000000)
        for power in X:
            gMeans.append(np.power(np.mean(np.power(sample_sequence, power)), 1 / power))
        return gMeans

    def plotDensity(self, ax=None, color=None, start=0, end=1):
        if ax is None:
            fig, ax = plt.subplots()
        X = np.linspace(start, end, 1000)
        Y = self.getDensity(X)
        ax.plot(X, Y, color=color)
        return self.label

    def plotQuantiles(self, ax=None, color=None, start=0, end=1):
        if ax is None:
            fig, ax = plt.subplots()
        X = np.linspace(start, end, 1000)
        Y = self.getQuantiles(X)
        ax.plot(X, Y, color=color)
        return self.label


class DistNormal(Distribution):
    def forMean(mean, scale=0.1):
        loc = mean
        return DistNormal(loc, scale)

    def __init__(self, loc, scale):
        self.loc = loc
        self.mean = loc
        self.scale = scale
        self.var = scale ** 2
        self.label = "norm_" + str(loc) + "_" + str(scale)

    def getDensity(self, X):
        return norm.pdf(X, loc=self.loc, scale=self.scale)

    def getQuantiles(self, X):
        return norm.ppf(X, loc=self.loc, scale=self.scale)

    def sample(self, n=1, unit=True):
        val = norm.rvs(loc=self.loc, scale=self.scale, size=n).flatten()
        return val if not unit else np.maximum(0, np.minimum(1, val))