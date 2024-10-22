import numpy as np
import random
import scipy
import itertools
from CSFramework import pull_query, pull_query_AC
import Distribution


def RoundRobin(n_arms, k, B, dists, setting, winner_setting, best_arm, borda_winner, second_best_mean, eps,
               borda_winner_scale, dependence_on_q, lower_bound, upper_bound, lower_scale_bound, upper_scale_bound,
               experiment, rts):
    # initialize internal statistics
    p = 2
    r = 0
    n_pulls = 0
    if experiment == "AC":
        runtime = 0
    # at least one pull for each set
    binom = scipy.special.binom(n_arms, k)
    win_probs = [[] for a in range(n_arms)]
    b = int(B/binom)
    if b == 0:
        b = 1
    for q in itertools.combinations(np.arange(n_arms), k):
        # get subset
        q = np.array(q)
        if dependence_on_q:
            dists, second_best_mean = Distribution.make_random_normal_dists_for_q(q, dists, best_arm, eps,
                                                                                  lower_bound, upper_bound,
                                                                                  lower_scale_bound, upper_scale_bound)
        wins = np.zeros(len(q))
        if setting == "median" or setting == "std":
            wins = [[] for a in q]
        pulls = np.zeros(len(q))
        # pull b times
        for i in range(b):
            if experiment == "AC":
                s = pull_query_AC(n_pulls, q, rts)
                runtime += np.max(s)
            else:
                s = pull_query(q, dists, borda_winner, best_arm, second_best_mean, eps,
                               borda_winner_scale)
            # update statistics
            if setting == "pb": # preference-based setting
                for arm_id, arm in enumerate(q):
                    pulls[arm_id] += 1
                    if s[arm_id] == np.max(s):
                        wins[arm_id] += 1
            if setting == "reward": # reward setting
                for arm_id, arm in enumerate(q):
                    pulls[arm_id] += 1
                    wins[arm_id] += s[arm_id]
            # reward setting with other statistics
            if setting == "median":
                for arm_id, arm in enumerate(q):
                    pulls[arm_id] += 1
                    wins[arm_id].append(s[arm_id])
            if setting == "std":
                for arm_id, arm in enumerate(q):
                    pulls[arm_id] += 1
                    wins[arm_id].append(s[arm_id])
            if setting == "power-mean":
                for arm_id, arm in enumerate(q):
                    pulls[arm_id] += 1
                    wins[arm_id] += s[arm_id] ** p
            n_pulls += 1
            # stop if Budget is reached
            if n_pulls > B:
                if winner_setting == "borda": # borda winner
                    for arm in range(n_arms):
                        if not win_probs[arm]:
                            win_probs[arm].append(0)
                    mean = [np.mean(win_probs[arm]) for arm in range(n_arms)]
                    if experiment == "AC":
                        return random.choice(np.where(mean == np.max(mean))[0]), runtime
                    return random.choice(np.where(mean == np.max(mean))[0])
        for arm_id, arm in enumerate(q):
            if setting == "median":
                win_probs[arm].append(np.median(wins[arm_id]))
            elif setting == "std":
                win_probs[arm].append(np.std(wins[arm_id]))
            elif setting == "power-mean":
                win_probs[arm].append((wins[arm_id] / pulls[arm_id]) ** (1/p))
            else:
                win_probs[arm].append(wins[arm_id] / pulls[arm_id])
        r += 1
    # return arm with best statistics
    if winner_setting == "borda":
        mean = [np.mean(win_probs[arm]) for arm in range(n_arms)]
        if experiment == "AC":
            return random.choice(np.where(mean == np.max(mean))[0]), runtime
        return random.choice(np.where(mean == np.max(mean))[0])